"""API-level tests for /ingest and the destructive-route guards.

The parser, chunker and vector store are replaced with fakes, so the whole ingest
endpoint runs without a model, a PDF, or a real index. Rate limiting is disabled
here (its own behaviour is covered in test_security) so repeated calls in this
module do not trip the hourly ingest budget.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api.main as m
from core import security
from ingestion.chunker import TextChunk
from ingestion.parser import ParsedDocument


class FakeIngestStore:
    backend = "faiss"

    def __init__(self, removed: int = 0, total: int = 5):
        self._removed = removed
        self._total = total
        self.indexed: list | None = None
        self.removed_sources: list[str] = []

    def remove_source(self, source, persist=True):
        self.removed_sources.append(source)
        return self._removed

    def index_chunks(self, chunks, show_progress=True, persist=True):
        self.indexed = chunks
        self._total += len(chunks)

    def sources(self):
        return ["x.pdf"]

    @property
    def total_vectors(self):
        return self._total


def _parsed(pages: int = 1, **meta) -> ParsedDocument:
    return ParsedDocument(
        source_path  = "x.pdf",
        total_pages  = pages,
        pages        = [],
        doc_metadata = {"source": "x.pdf", **meta},
    )


def _chunks() -> list[TextChunk]:
    return [TextChunk(
        chunk_id = "x.pdf_p1_0",
        text     = "hello",
        token_len = 1,
        metadata = {"source": "x.pdf", "page_number": 1, "script": "roman"},
    )]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "")
    # Keep the upload from being written into the repo's data directory.
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)


def _client(monkeypatch, store: FakeIngestStore | None = None) -> TestClient:
    monkeypatch.setattr(m, "_vector_store", store or FakeIngestStore(), raising=False)
    monkeypatch.setattr(m, "parse_document", lambda *a, **k: _parsed())
    monkeypatch.setattr(m, "chunk_document", lambda *a, **k: _chunks())
    return TestClient(m.app)


def _pdf(content: bytes = b"%PDF-1.4 whatever"):
    return {"file": ("x.pdf", content, "application/pdf")}


# ── happy path ─────────────────────────────────────────────────────────────────

def test_ingest_success_returns_counts_and_hash(monkeypatch):
    store = FakeIngestStore(removed=0, total=5)
    resp = _client(monkeypatch, store).post("/ingest", files=_pdf())

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["source"] == "x.pdf"
    assert body["total_chunks"] == 1
    assert body["file_hash"]
    assert body["replaced"] is False
    # The stored chunks carry the hash and the canonical source name.
    assert store.indexed[0].metadata["file_hash"] == body["file_hash"]
    assert store.indexed[0].metadata["source"] == "x.pdf"


def test_reingest_reports_replacement(monkeypatch):
    store = FakeIngestStore(removed=3)
    body = _client(monkeypatch, store).post("/ingest", files=_pdf()).json()

    assert body["replaced"] is True
    assert body["replaced_chunks"] == 3
    assert store.removed_sources == ["x.pdf"]


# ── input validation ───────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf(monkeypatch):
    resp = _client(monkeypatch).post(
        "/ingest", files={"file": ("notes.txt", b"hi", "text/plain")}
    )
    assert resp.status_code == 422
    assert "PDF" in resp.json()["detail"]


def test_ingest_rejects_an_oversized_upload(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(m, "MAX_UPLOAD_MB", 0)
    assert client.post("/ingest", files=_pdf(b"some bytes")).status_code == 413


def test_ingest_rejects_a_document_over_the_page_limit(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(m, "parse_document", lambda *a, **k: _parsed(pages=10_000))
    assert client.post("/ingest", files=_pdf()).status_code == 413


def test_ingest_distinguishes_a_missing_ocr_engine(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(m, "chunk_document", lambda *a, **k: [])
    monkeypatch.setattr(
        m, "parse_document", lambda *a, **k: _parsed(ocr_unavailable="tesseract missing")
    )
    resp = client.post("/ingest", files=_pdf())
    assert resp.status_code == 422
    assert "tesseract missing" in resp.json()["detail"]


def test_ingest_reports_an_empty_document(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(m, "chunk_document", lambda *a, **k: [])
    resp = client.post("/ingest", files=_pdf())
    assert resp.status_code == 422
    assert "zero chunks" in resp.json()["detail"]


# ── auth on destructive routes ─────────────────────────────────────────────────

def test_ingest_requires_a_key_when_configured(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    client = _client(monkeypatch)

    assert client.post("/ingest", files=_pdf()).status_code == 401
    ok = client.post("/ingest", files=_pdf(), headers={"X-API-Key": "sekret"})
    assert ok.status_code == 200


def test_reset_is_guarded_and_clears_the_index(monkeypatch):
    created = {}

    class FakeVectorStore:
        def __init__(self):
            created["store"] = self
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(m, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    client = _client(monkeypatch)

    assert client.delete("/index/reset").status_code == 401
    resp = client.delete("/index/reset", headers={"X-API-Key": "sekret"})
    assert resp.status_code == 200
    assert created["store"].reset_called is True


def test_admin_requests_is_guarded(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    client = _client(monkeypatch)

    assert client.get("/admin/requests").status_code == 401
    assert client.get("/admin/requests", headers={"X-API-Key": "sekret"}).status_code == 200


def test_health_reports_whether_auth_is_configured(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    monkeypatch.setattr(m, "_vector_store", FakeIngestStore(), raising=False)
    monkeypatch.setattr(m, "_generator", None, raising=False)
    monkeypatch.setattr(m, "ollama_reachable", lambda: True)
    monkeypatch.setattr(m, "_ollama_probe_cache", None)

    body = TestClient(m.app).get("/health").json()
    assert body["auth_configured"] is True


# ── feedback ───────────────────────────────────────────────────────────────────

def _feedback_payload(verdict: str = "down") -> dict:
    return {
        "verdict":  verdict,
        "query":    "ఎరుక అంటే ఏమిటి?",
        "answer":   "an answer the reader judged",
        "provider": "groq",
        "model":    "qwen/qwen3.8-27b",
        "sources":  [],
    }


def test_feedback_appends_a_json_line(monkeypatch, tmp_path):
    log = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(m, "_FEEDBACK_PATH", log)
    resp = _client(monkeypatch).post("/feedback", json=_feedback_payload())

    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["verdict"] == "down"
    assert record["query"] == "ఎరుక అంటే ఏమిటి?"


def test_feedback_rejects_an_unknown_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    resp = _client(monkeypatch).post("/feedback", json=_feedback_payload("maybe"))
    assert resp.status_code == 422


def test_feedback_requires_a_query_and_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    client = _client(monkeypatch)
    assert client.post("/feedback", json={"verdict": "up"}).status_code == 422
