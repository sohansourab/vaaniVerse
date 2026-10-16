import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from groq import Groq

from core.config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)
from core.exceptions import LLMInferenceError
from reasoning.prompter import PromptPackage

logger = logging.getLogger(__name__)

# Provider names accepted by the API / config.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "groq")


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class GeneratedResponse:
    answer:          str
    query:           str
    model:           str
    provider:        str         # provider actually used ("ollama" | "groq")
    backend:         str         # legacy alias of `provider`, kept for compatibility
    sources_used:    list[dict]  # [{chunk_id, source, page_number, score}, ...]
    prompt_tokens:   int | None  # None when the provider does not report usage
    completion_tokens: int | None


# ══════════════════════════════════════════════════════════════════════════════
# Provider interface
# ══════════════════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """
    Interface for a text-generation backend.

    Implementations take a user prompt (plus an optional system prompt) and
    return the model's completion as a plain string.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Return the completion string for `prompt`."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self.name


# ── Ollama provider (default) ──────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    LLM inference via a local Ollama instance.
    Offline-friendly default — no data leaves the machine.
    Requires: `ollama serve` and a pulled model (e.g. `ollama pull qwen2.5:3b`).
    """

    name = "ollama"

    def __init__(
        self,
        base_url:    str   = OLLAMA_BASE_URL,
        model:       str   = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE,
    ) -> None:
        self._base_url    = base_url.rstrip("/")
        self._chat_url    = f"{self._base_url}/api/chat"
        self._model       = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt:        str,
        *,
        system_prompt: str | None = None,
        temperature:   float | None = None,
        max_tokens:    int   = 1024,
        timeout:       int   = 120,   # local inference can be slow on CPU/small GPUs
        **kwargs,
    ) -> str:
        """
        Sends a chat request to the local Ollama `/api/chat` endpoint.

        Raises:
            LLMInferenceError: On connection failure, timeout, or empty response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":  self._model,
            "stream": False,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": max_tokens,
            },
            "messages": messages,
        }

        try:
            resp = httpx.post(self._chat_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMInferenceError(
                f"Ollama request timed out after {timeout}s.",
                details={"model": self._model, "base_url": self._base_url},
            ) from exc
        except Exception as exc:
            raise LLMInferenceError(
                "Ollama API call failed.",
                details={"model": self._model, "base_url": self._base_url, "error": str(exc)},
            ) from exc

        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise LLMInferenceError(
                "Ollama returned an empty response.",
                details={"model": self._model, "raw_response": str(data)[:200]},
            )

        logger.info(
            f"Ollama [{self._model}] — "
            f"prompt_tokens={data.get('prompt_eval_count')}, "
            f"completion_tokens={data.get('eval_count')}"
        )
        return answer


# ── Groq provider (opt-in) ─────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    LLM inference via the Groq API (opt-in).

    The API key is read from the environment at construction time. Construction
    never raises: a missing key fails only when `generate()` is actually called,
    so an unconfigured Groq backend cannot crash the server at startup.
    """

    name = "groq"

    def __init__(
        self,
        api_key:     str   = GROQ_API_KEY,
        model:       str   = GROQ_MODEL,
        temperature: float = GROQ_TEMPERATURE,
        max_tokens:  int   = GROQ_MAX_TOKENS,
    ) -> None:
        self._api_key     = api_key
        self._model       = model
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._client: Groq | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _ensure_client(self) -> Groq:
        if not self._api_key:
            raise LLMInferenceError(
                "GROQ_API_KEY is not set. Configure it to use provider 'groq'.",
                details={"provider": "groq"},
            )
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt:        str,
        *,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Sends a chat request to the Groq API.

        Raises:
            LLMInferenceError: If the key is missing, on API error/timeout, or on
                an empty response.
        """
        client = self._ensure_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model       = self._model,
                messages    = messages,
                max_tokens  = self._max_tokens,
                temperature = self._temperature,
            )
        except Exception as exc:
            raise LLMInferenceError(
                "Groq API call failed.",
                details={"model": self._model, "error": str(exc)},
            ) from exc

        choice = response.choices[0]
        answer = choice.message.content

        if not answer or not answer.strip():
            raise LLMInferenceError(
                "Groq returned an empty response.",
                details={"model": self._model, "finish_reason": choice.finish_reason},
            )

        usage = getattr(response, "usage", None)
        logger.info(
            f"Groq [{self._model}] — "
            f"prompt_tokens={getattr(usage, 'prompt_tokens', None)}, "
            f"completion_tokens={getattr(usage, 'completion_tokens', None)}"
        )
        return answer.strip()


# ── Shared utilities ───────────────────────────────────────────────────────────

def _extract_source_metadata(package: PromptPackage) -> list[dict]:
    """
    Extracts citation-ready metadata from context chunks.
    Returned in GeneratedResponse.sources_used for API consumers to render
    source attribution without re-parsing the answer text.
    """
    return [
        {
            "chunk_id":    chunk.chunk_id,
            "source":      chunk.metadata.get("source", "unknown"),
            "page_number": chunk.metadata.get("page_number", "?"),
            "score":       round(chunk.score, 4),
            "script":      chunk.metadata.get("script", "unknown"),
        }
        for chunk in package.context_chunks
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Unified Generator (provider-agnostic facade)
# ══════════════════════════════════════════════════════════════════════════════

_PROVIDER_FACTORIES: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "groq":   GroqProvider,
}


class Generator:
    """
    Provider-agnostic LLM generator.

    The environment-configured provider (`LLM_PROVIDER`, default "ollama") is used
    unless a per-call `provider` is supplied. Providers are constructed lazily and
    never touch the network at construction time, so startup cannot fail just
    because a backend is unconfigured or offline.

    Usage:
        generator = Generator()
        response  = generator.generate(prompt_package)                  # env default
        response  = generator.generate(prompt_package, provider="groq") # override
    """

    def __init__(self, provider: str = LLM_PROVIDER) -> None:
        self.default_provider = (provider or "ollama").lower()
        self._providers: dict[str, LLMProvider] = {}
        logger.info(f"Generator initialized. Default provider: '{self.default_provider}'")

    @property
    def backend(self) -> str:
        """Legacy alias for the default provider name."""
        return self.default_provider

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """
        Returns a cached provider instance for `name` (defaults to the env default).

        Raises:
            LLMInferenceError: If `name` is not a supported provider.
        """
        provider_name = (name or self.default_provider).lower()
        if provider_name not in _PROVIDER_FACTORIES:
            raise LLMInferenceError(
                f"Unknown LLM provider '{provider_name}'. "
                f"Choose one of: {', '.join(_PROVIDER_FACTORIES)}.",
                details={"provider": provider_name, "supported": list(_PROVIDER_FACTORIES)},
            )
        if provider_name not in self._providers:
            self._providers[provider_name] = _PROVIDER_FACTORIES[provider_name]()
        return self._providers[provider_name]

    def generate(
        self,
        package:  PromptPackage,
        provider: str | None = None,
    ) -> GeneratedResponse:
        """
        Generates an answer for `package` using the selected provider.

        Args:
            package:  PromptPackage from reasoning.prompter.
            provider: Optional per-call override ("ollama" | "groq").
                      None uses the environment default.

        Returns:
            GeneratedResponse (includes the `provider` that was used).
        """
        active = self.get_provider(provider)
        answer = active.generate(
            package.user_prompt,
            system_prompt=package.system_prompt,
        )
        return GeneratedResponse(
            answer            = answer,
            query             = package.query,
            model             = active.model_name,
            provider          = active.name,
            backend           = active.name,
            sources_used      = _extract_source_metadata(package),
            prompt_tokens     = None,
            completion_tokens = None,
        )
