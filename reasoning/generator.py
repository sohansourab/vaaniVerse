import logging
from dataclasses import dataclass

import httpx
from groq import Groq

from core.config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)
from core.exceptions import LLMInferenceError
from reasoning.prompter import PromptPackage

logger = logging.getLogger(__name__)


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class GeneratedResponse:
    answer:          str
    query:           str
    model:           str
    backend:         str
    sources_used:    list[dict]   # [{chunk_id, source, page_number, score}, ...]
    prompt_tokens:   int | None   # None for Ollama (not always returned)
    completion_tokens: int | None


# ══════════════════════════════════════════════════════════════════════════════
# Groq Backend
# ══════════════════════════════════════════════════════════════════════════════

class GroqGenerator:
    """
    LLM inference via Groq API.
    Uses llama3-70b-8192 by default — fastest available model with
    sufficient context window for 5 retrieved chunks + system prompt.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise LLMInferenceError(
                "GROQ_API_KEY is not set. Add it to your .env file.",
                details={"backend": "groq"},
            )
        self._client = Groq(api_key=GROQ_API_KEY)

    def generate(self, package: PromptPackage) -> GeneratedResponse:
        """
        Sends a PromptPackage to Groq and returns a GeneratedResponse.

        Args:
            package: PromptPackage from reasoning.prompter.build_prompt()

        Returns:
            GeneratedResponse with answer, source metadata, and token counts.

        Raises:
            LLMInferenceError: On API error, timeout, or empty response.
        """
        messages = [
            {"role": "system", "content": package.system_prompt},
            {"role": "user",   "content": package.user_prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = messages,
                max_tokens  = GROQ_MAX_TOKENS,
                temperature = GROQ_TEMPERATURE,
            )
        except Exception as exc:
            raise LLMInferenceError(
                "Groq API call failed.",
                details={
                    "model":         GROQ_MODEL,
                    "error":         str(exc),
                    "query_snippet": package.query[:80],
                },
            )

        choice = response.choices[0]
        answer = choice.message.content

        if not answer or not answer.strip():
            raise LLMInferenceError(
                "Groq returned an empty response.",
                details={"model": GROQ_MODEL, "finish_reason": choice.finish_reason},
            )

        usage = response.usage

        logger.info(
            f"Groq [{GROQ_MODEL}] — "
            f"prompt_tokens={usage.prompt_tokens}, "
            f"completion_tokens={usage.completion_tokens}"
        )

        return GeneratedResponse(
            answer            = answer.strip(),
            query             = package.query,
            model             = GROQ_MODEL,
            backend           = "groq",
            sources_used      = _extract_source_metadata(package),
            prompt_tokens     = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Ollama Backend
# ══════════════════════════════════════════════════════════════════════════════

class OllamaGenerator:
    """
    LLM inference via local Ollama instance.
    Offline fallback — no data leaves the machine.
    Requires: `ollama serve` + `ollama pull llama3`

    Uses /api/chat endpoint (supports system/user message separation)
    rather than /api/generate (prompt string only) for consistency
    with the PromptPackage structure.
    """

    def __init__(self) -> None:
        self._base_url = OLLAMA_BASE_URL.rstrip("/")
        self._chat_url = f"{self._base_url}/api/chat"
        self._health_url = f"{self._base_url}/api/tags"
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Fails fast if Ollama is not reachable before the first query."""
        try:
            resp = httpx.get(self._health_url, timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMInferenceError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Ensure `ollama serve` is running.",
                details={"base_url": self._base_url, "error": str(exc)},
            )

    def generate(self, package: PromptPackage) -> GeneratedResponse:
        """
        Sends a PromptPackage to the local Ollama /api/chat endpoint.

        Args:
            package: PromptPackage from reasoning.prompter.build_prompt()

        Returns:
            GeneratedResponse with answer and source metadata.
            prompt_tokens / completion_tokens may be None if Ollama
            does not return eval_count for the selected model.

        Raises:
            LLMInferenceError: On connection failure, timeout, or empty response.
        """
        payload = {
            "model":  OLLAMA_MODEL,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": 1024,
            },
            "messages": [
                {"role": "system", "content": package.system_prompt},
                {"role": "user",   "content": package.user_prompt},
            ],
        }

        try:
            resp = httpx.post(
                self._chat_url,
                json    = payload,
                timeout = 120,   # local inference can be slow on CPU
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            raise LLMInferenceError(
                "Ollama request timed out after 120s.",
                details={"model": OLLAMA_MODEL, "base_url": self._base_url},
            )
        except Exception as exc:
            raise LLMInferenceError(
                "Ollama API call failed.",
                details={"model": OLLAMA_MODEL, "error": str(exc)},
            )

        answer = data.get("message", {}).get("content", "").strip()

        if not answer:
            raise LLMInferenceError(
                "Ollama returned an empty response.",
                details={"model": OLLAMA_MODEL, "raw_response": str(data)[:200]},
            )

        prompt_tokens     = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        logger.info(
            f"Ollama [{OLLAMA_MODEL}] — "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}"
        )

        return GeneratedResponse(
            answer            = answer,
            query             = package.query,
            model             = OLLAMA_MODEL,
            backend           = "ollama",
            sources_used      = _extract_source_metadata(package),
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Shared Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _extract_source_metadata(package: PromptPackage) -> list[dict]:
    """
    Extracts citation-ready metadata from context chunks.
    Returned in GeneratedResponse.sources_used for API consumers
    to render source attribution in the UI without re-parsing the answer text.
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
# Unified Interface
# ══════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Backend-agnostic LLM generator.
    Reads LLM_BACKEND from config and delegates to Groq or Ollama.
    All downstream code (api/main.py) imports only this class.

    Usage:
        generator = Generator()
        response  = generator.generate(prompt_package)
        print(response.answer)
        print(response.sources_used)
    """

    def __init__(self, backend: str = LLM_BACKEND) -> None:
        self.backend = backend.lower()
        if self.backend == "groq":
            self._generator = GroqGenerator()
        elif self.backend == "ollama":
            self._generator = OllamaGenerator()
        else:
            raise LLMInferenceError(
                f"Unknown LLM_BACKEND '{backend}'. Choose 'groq' or 'ollama'.",
                details={"backend": backend},
            )
        logger.info(f"Generator initialized with backend: '{self.backend}'")

    def generate(self, package: PromptPackage) -> GeneratedResponse:
        """
        Delegates to the configured backend generator.

        Args:
            package: PromptPackage from prompter.build_prompt()
                     or prompter.build_no_result_prompt()

        Returns:
            GeneratedResponse
        """
        return self._generator.generate(package)