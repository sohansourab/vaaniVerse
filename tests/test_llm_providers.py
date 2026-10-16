"""Unit tests for the LLM provider layer (no real services).

HTTP/SDK calls are monkeypatched, so these tests run fully offline.
"""
from __future__ import annotations

import pytest

from core.exceptions import LLMInferenceError
from reasoning.generator import (
    SUPPORTED_PROVIDERS,
    Generator,
    GroqProvider,
    OllamaProvider,
)
from reasoning.prompter import PromptPackage
from retrieval.vector_store import RetrievedChunk


# ── helpers ────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._data


def _package() -> PromptPackage:
    chunk = RetrievedChunk(
        chunk_id="doc_p1_0",
        text="Rama is the hero of the Ramayana.",
        score=0.91,
        metadata={"source": "9.nenu-evarini.pdf", "page_number": 1, "script": "roman"},
    )
    return PromptPackage(
        system_prompt="SYS",
        user_prompt="USER",
        context_chunks=[chunk],
        query="who is rama",
    )


# ── OllamaProvider ──────────────────────────────────────────────────────────────

def test_supported_providers_are_ollama_and_groq():
    assert SUPPORTED_PROVIDERS == ("ollama", "groq")


def test_ollama_provider_generate(monkeypatch):
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp({"message": {"content": "  hello from ollama  "}})

    monkeypatch.setattr("reasoning.generator.httpx.post", fake_post)

    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:3b")
    out = provider.generate("who is rama?", system_prompt="SYS")

    assert out == "hello from ollama"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["stream"] is False
    assert captured["json"]["model"] == "qwen2.5:3b"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "who is rama?"}


def test_ollama_provider_empty_response_raises(monkeypatch):
    monkeypatch.setattr(
        "reasoning.generator.httpx.post",
        lambda *a, **k: _FakeResp({"message": {"content": ""}}),
    )
    with pytest.raises(LLMInferenceError):
        OllamaProvider(base_url="http://localhost:11434", model="m").generate("x")


def test_ollama_provider_connection_error_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("reasoning.generator.httpx.post", boom)
    with pytest.raises(LLMInferenceError):
        OllamaProvider(base_url="http://localhost:11434", model="m").generate("x")


# ── GroqProvider ────────────────────────────────────────────────────────────────

def test_groq_provider_without_key_raises_on_generate():
    provider = GroqProvider(api_key="")
    assert provider.configured is False
    with pytest.raises(LLMInferenceError):
        provider.generate("hi")


def test_groq_provider_generate(monkeypatch):
    class _Msg:
        content = "groq answer"

    class _Choice:
        message = _Msg()

    class _Completions:
        def create(self, **kwargs):
            return type("_Resp", (), {"choices": [_Choice()]})()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr("reasoning.generator.Groq", lambda api_key=None: _Client())

    provider = GroqProvider(api_key="dummy-key", model="llama-3.3-70b-versatile")
    assert provider.generate("hi", system_prompt="S") == "groq answer"


# ── Generator façade ────────────────────────────────────────────────────────────

class _FakeProvider:
    def __init__(self, name: str):
        self.name = name
        self.model_name = f"fake-{name}"

    def generate(self, prompt, **kwargs):
        return f"answer-from-{self.name}"


def test_generator_uses_env_default_provider(monkeypatch):
    gen = Generator(provider="ollama")
    monkeypatch.setattr(gen, "get_provider", lambda name=None: _FakeProvider(name or "ollama"))

    resp = gen.generate(_package())
    assert resp.provider == "ollama"
    assert resp.backend == "ollama"
    assert resp.answer == "answer-from-ollama"
    assert resp.sources_used[0]["source"] == "9.nenu-evarini.pdf"


def test_generator_per_call_override(monkeypatch):
    gen = Generator(provider="ollama")
    monkeypatch.setattr(gen, "get_provider", lambda name=None: _FakeProvider(name or "ollama"))

    resp = gen.generate(_package(), provider="groq")
    assert resp.provider == "groq"
    assert resp.answer == "answer-from-groq"


def test_generator_rejects_unknown_provider():
    gen = Generator(provider="ollama")
    with pytest.raises(LLMInferenceError):
        gen.get_provider("bogus")
