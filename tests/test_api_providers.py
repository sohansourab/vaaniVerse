"""API-level provider tests using FastAPI TestClient with fakes (no real services).

The vector store and generator singletons in `api.main` are replaced with fakes so
that no embedding model, database, Ollama, or Groq call is made.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as m
from reasoning.generator import Generator
from retrieval.vector_store import RetrievedChunk


class FakeStore:
    backend = "faiss"
    total_vectors = 5
    # Mirrors VectorStore.reranking_active, which QueryResponse/SearchResponse report.
    reranking_active = False

    def query(self, query_text, top_k, score_threshold, filter_by=None):
        return [
            RetrievedChunk(
                chunk_id="9.nenu-evarini_p1_0",
                text="Rama is the hero of the Ramayana.",
                score=0.9,
                metadata={"source": "9.nenu-evarini.pdf", "page_number": 1, "script": "roman"},
            )
        ]


class FakeProvider:
    def __init__(self, name: str):
        self.name = name
        self.model_name = f"fake-{name}"

    def generate(self, prompt, **kwargs):
        return "mocked answer"


class RecordingProvider(FakeProvider):
    """Keeps the prompt it was handed so tests can assert on language routing."""

    def __init__(self, name: str):
        super().__init__(name)
        self.user_prompt: str | None = None
        self.system_prompt: str | None = None

    def generate(self, prompt, **kwargs):
        self.user_prompt = prompt
        self.system_prompt = kwargs.get("system_prompt")
        return "mocked answer"


def _client(monkeypatch, *, default: str = "ollama", groq_configured: bool = False) -> TestClient:
    gen = Generator(provider=default)
    monkeypatch.setattr(gen, "get_provider", lambda name=None: FakeProvider(name or default))
    monkeypatch.setattr(m, "_vector_store", FakeStore(), raising=False)
    monkeypatch.setattr(m, "_generator", gen, raising=False)
    monkeypatch.setattr(m, "GROQ_CONFIGURED", groq_configured, raising=False)
    monkeypatch.setattr(m, "LLM_PROVIDER", default, raising=False)
    return TestClient(m.app)


def _client_with_recorder(monkeypatch, *, default: str = "ollama") -> tuple[TestClient, RecordingProvider]:
    recorder = RecordingProvider(default)
    gen = Generator(provider=default)
    monkeypatch.setattr(gen, "get_provider", lambda name=None: recorder)
    monkeypatch.setattr(m, "_vector_store", FakeStore(), raising=False)
    monkeypatch.setattr(m, "_generator", gen, raising=False)
    monkeypatch.setattr(m, "GROQ_CONFIGURED", False, raising=False)
    monkeypatch.setattr(m, "LLM_PROVIDER", default, raising=False)
    return TestClient(m.app), recorder


def test_default_provider_is_ollama(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/query", json={"query": "who is rama"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "ollama"
    assert body["answer"] == "mocked answer"
    assert body["sources_used"][0]["source"] == "9.nenu-evarini.pdf"


def test_invalid_provider_returns_400(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/query", json={"query": "who is rama", "provider": "bogus"})
    assert resp.status_code == 400
    assert "Invalid provider" in resp.json()["detail"]


def test_groq_without_key_returns_503(monkeypatch):
    client = _client(monkeypatch, groq_configured=False)
    resp = client.post("/query", json={"query": "who is rama", "provider": "groq"})
    assert resp.status_code == 503
    assert "GROQ_API_KEY" in resp.json()["detail"]


def test_response_includes_provider_used_groq(monkeypatch):
    client = _client(monkeypatch, groq_configured=True)
    resp = client.post("/query", json={"query": "who is rama", "provider": "groq"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "groq"


def test_health_reports_llm_default_and_groq_configured(monkeypatch):
    client = _client(monkeypatch, default="ollama", groq_configured=False)
    body = client.get("/health").json()
    assert body["llm_default"] == "ollama"
    assert body["groq_configured"] is False


def test_health_groq_configured_true(monkeypatch):
    client = _client(monkeypatch, default="ollama", groq_configured=True)
    body = client.get("/health").json()
    assert body["groq_configured"] is True


# ── Answer-language routing (regression) ──────────────────────────────────────
#
# detect_script reports "roman" for both English and Romanized Telugu. Every
# "roman" query used to be routed to the English system prompt, whose rule 2
# forbids Telugu, so a Romanized Telugu query was answered in English even
# though the Telugu prompt carries a rule for exactly that case.

def test_romanized_telugu_query_reaches_model_with_telugu_prompt(monkeypatch):
    client, recorder = _client_with_recorder(monkeypatch)
    resp = client.post("/query", json={"query": "nenu evaru?"})
    assert resp.status_code == 200
    assert recorder.system_prompt is not None
    assert "specializing in Telugu" in recorder.system_prompt
    # The model is shown the transliterated query, not the Roman input.
    assert "నేను ఎవరు?" in recorder.user_prompt


def test_english_query_reaches_model_with_english_prompt(monkeypatch):
    client, recorder = _client_with_recorder(monkeypatch)
    resp = client.post("/query", json={"query": "who am i?"})
    assert resp.status_code == 200
    assert "specializing in Telugu" not in recorder.system_prompt
    assert "who am i?" in recorder.user_prompt


def test_native_telugu_query_reaches_model_with_telugu_prompt(monkeypatch):
    client, recorder = _client_with_recorder(monkeypatch)
    resp = client.post("/query", json={"query": "నేను ఎవరు?"})
    assert resp.status_code == 200
    assert "specializing in Telugu" in recorder.system_prompt
