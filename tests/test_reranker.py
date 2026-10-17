"""
Unit tests for the reranking stage.

Fully offline: no reranker weights are downloaded and no embeddings are
computed. The model is replaced with a fake, which is what makes it possible to
assert the things that actually matter — score alignment, error surfacing,
ranking, thresholding, and the explicit CPU device that keeps the reranker off
the 4 GB GPU.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from core.exceptions import RerankError
from retrieval import reranker
from retrieval import vector_store
from retrieval.vector_store import RetrievedChunk, VectorStore


# ── fixtures ───────────────────────────────────────────────────────────────────

def _chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id = chunk_id,
        text     = text,
        score    = score,
        metadata = {"source": "book.pdf", "page_number": 1, "script": "telugu"},
    )


class _FakeReranker:
    """Stands in for FlagReranker, recording the pairs it was asked to score."""

    def __init__(self, scores):
        self._scores = scores
        self.seen: list = []

    def compute_score(self, pairs, **kwargs):
        self.seen.append(pairs)
        return self._scores


@pytest.fixture
def fake_reranker(monkeypatch):
    """Route reranker.score_pairs through a controllable fake model."""
    def install(scores):
        fake = _FakeReranker(scores)
        monkeypatch.setattr(reranker, "get_reranker", lambda *a, **k: fake)
        return fake
    return install


# ── output normalisation ───────────────────────────────────────────────────────

def test_as_float_list_handles_scalars_and_sequences():
    assert reranker._as_float_list(0.7) == [0.7]
    assert reranker._as_float_list(np.float32(0.7)) == pytest.approx([0.7])
    assert reranker._as_float_list([0.1, 0.2]) == pytest.approx([0.1, 0.2])
    assert reranker._as_float_list(np.array([0.1, 0.2], dtype=np.float32)) == pytest.approx([0.1, 0.2])


# ── score_pairs ────────────────────────────────────────────────────────────────

def test_empty_passages_return_no_scores(fake_reranker):
    fake_reranker([0.9])
    assert reranker.score_pairs("q", []) == []


def test_empty_query_is_rejected(fake_reranker):
    fake_reranker([0.9])
    with pytest.raises(RerankError):
        reranker.score_pairs("   ", ["passage"])


def test_scores_stay_aligned_with_input_order(fake_reranker):
    fake = fake_reranker([0.11, 0.22, 0.33])
    scores = reranker.score_pairs("ఎవరు", ["a", "b", "c"])

    assert scores == pytest.approx([0.11, 0.22, 0.33])
    pairs = fake.seen[0]
    assert [passage for _query, passage in pairs] == ["a", "b", "c"]
    assert {query for query, _passage in pairs} == {"ఎవరు"}


def test_scoring_failure_is_wrapped(monkeypatch):
    class Exploding:
        def compute_score(self, pairs, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(reranker, "get_reranker", lambda *a, **k: Exploding())

    with pytest.raises(RerankError) as exc:
        reranker.score_pairs("q", ["a"])
    assert "CUDA out of memory" in str(exc.value.details)


def test_score_count_mismatch_is_rejected(fake_reranker):
    fake_reranker([0.5])  # one score for three passages
    with pytest.raises(RerankError):
        reranker.score_pairs("q", ["a", "b", "c"])


# ── model loading ──────────────────────────────────────────────────────────────

def test_reranker_is_loaded_once_and_pinned_to_cpu(monkeypatch):
    """The device must be explicit: FlagEmbedding otherwise claims every GPU."""
    created: list[dict] = []

    class FakeFlagReranker:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    monkeypatch.setattr(reranker, "_RERANKER_CACHE", {})

    first  = reranker.get_reranker("fake/model", "cpu")
    second = reranker.get_reranker("fake/model", "cpu")

    assert first is second
    assert len(created) == 1, "the model must be cached, not reloaded per query"
    assert created[0]["devices"] == "cpu"
    assert created[0]["normalize"] is True, "0..1 sigmoid scores keep the threshold contract"
    assert created[0]["use_fp16"] is False


def test_load_failure_raises_rerank_error(monkeypatch):
    class ExplodingFlagReranker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights not found")

    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.FlagReranker = ExplodingFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    monkeypatch.setattr(reranker, "_RERANKER_CACHE", {})

    with pytest.raises(RerankError) as exc:
        reranker.get_reranker("fake/model", "cpu")
    assert "weights not found" in str(exc.value.details)


# ── activation policy ──────────────────────────────────────────────────────────

def test_reranking_off_by_config(monkeypatch):
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", False)
    assert VectorStore._reranking_active() is False


def test_reranking_on_when_available(monkeypatch):
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", True)
    monkeypatch.setattr(vector_store, "reranker_available", lambda: True)
    assert VectorStore._reranking_active() is True


def test_enabled_but_unavailable_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", True)
    monkeypatch.setattr(vector_store, "reranker_available", lambda: False)
    monkeypatch.setattr(vector_store, "reranker_probe", lambda: "not installed")

    assert VectorStore._reranking_active() is False


# ── reranking the candidate set ────────────────────────────────────────────────

def test_rerank_reorders_preserves_dense_scores_and_thresholds(monkeypatch):
    candidates = [
        _chunk("a", "low dense, best rerank", 0.60),
        _chunk("b", "high dense, worst rerank", 0.90),
        _chunk("c", "middling", 0.70),
    ]
    # Deliberately anti-correlated with the dense scores.
    monkeypatch.setattr(vector_store, "score_pairs", lambda q, texts: [0.9, 0.2, 0.5])

    kept = VectorStore._rerank("q", candidates, top_k=5, score_threshold=0.3)

    assert [c.chunk_id for c in kept] == ["a", "c"]      # b fell below 0.3
    assert [c.score for c in kept] == pytest.approx([0.9, 0.5])
    # The dense score is still readable, not overwritten.
    assert [c.retrieval_score for c in kept] == pytest.approx([0.60, 0.70])


def test_rerank_returns_nothing_when_everything_is_below_threshold(monkeypatch):
    monkeypatch.setattr(vector_store, "score_pairs", lambda q, texts: [0.1, 0.2])
    kept = VectorStore._rerank("q", [_chunk("a", "x", 0.9), _chunk("b", "y", 0.8)],
                               top_k=5, score_threshold=0.5)
    assert kept == []


def test_retrieval_score_defaults_to_score():
    chunk = _chunk("a", "text", 0.42)
    assert chunk.retrieval_score == 0.42


# ── two-stage wiring in query() ────────────────────────────────────────────────

def _capture_store(monkeypatch, chunks):
    captured: dict = {}
    store = VectorStore(backend="faiss")

    def fake_search(query_vector, top_k, score_threshold, **kwargs):
        captured["top_k"] = top_k
        captured["score_threshold"] = score_threshold
        return list(chunks)

    monkeypatch.setattr(store._store, "search", fake_search)
    monkeypatch.setattr(vector_store, "embed_query", lambda q: np.zeros(1024, dtype=np.float32))
    return store, captured


def test_query_widens_the_candidate_set_and_drops_the_cosine_cutoff(monkeypatch):
    candidates = [_chunk(str(i), f"t{i}", 0.9 - i * 0.1) for i in range(3)]
    store, captured = _capture_store(monkeypatch, candidates)

    monkeypatch.setattr(vector_store, "RERANK_ENABLED", True)
    monkeypatch.setattr(vector_store, "reranker_available", lambda: True)
    monkeypatch.setattr(vector_store, "RERANK_CANDIDATES", 20)
    monkeypatch.setattr(vector_store, "score_pairs", lambda q, texts: [0.4, 0.8, 0.6])

    results = store.query("q", top_k=2, score_threshold=0.5)

    assert captured["top_k"] == 20, "reranking needs a wider net than top_k"
    assert captured["score_threshold"] == 0.0, "the cosine cutoff must not pre-filter candidates"
    assert [c.chunk_id for c in results] == ["1", "2"]
    assert [c.retrieval_score for c in results] == pytest.approx([0.8, 0.7])


def test_query_is_unchanged_when_reranking_is_off(monkeypatch):
    candidates = [_chunk("a", "x", 0.9), _chunk("b", "y", 0.8)]
    store, captured = _capture_store(monkeypatch, candidates)

    monkeypatch.setattr(vector_store, "RERANK_ENABLED", False)

    results = store.query("q", top_k=5, score_threshold=0.45)

    assert captured["top_k"] == 5
    assert captured["score_threshold"] == 0.45
    assert [c.score for c in results] == pytest.approx([0.9, 0.8])
    assert all(c.retrieval_score == c.score for c in results)


def test_query_truncates_to_top_k_without_reranking(monkeypatch):
    candidates = [_chunk(str(i), f"t{i}", 0.9 - i * 0.01) for i in range(10)]
    store, _ = _capture_store(monkeypatch, candidates)
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", False)

    assert len(store.query("q", top_k=5, score_threshold=0.0)) == 5
