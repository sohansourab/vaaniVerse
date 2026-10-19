"""Unit tests for the vector-store layer.

The FAISS backend is exercised directly with synthetic vectors, so no embedding
model is loaded. The hybrid path is driven through a fake backend so the fusion
and the abstention gate can be asserted without any model at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.exceptions import IndexNotFoundError, VectorStoreError
from ingestion.chunker import TextChunk
from retrieval import vector_store
from retrieval.vector_store import FAISSVectorStore, RetrievedChunk, VectorStore


# ── helpers ────────────────────────────────────────────────────────────────────

def _chunk(cid: str, text: str = "t", source: str = "book.pdf",
           page: int = 1, script: str = "telugu") -> TextChunk:
    return TextChunk(
        chunk_id = cid,
        text     = text,
        token_len = 1,
        metadata = {"source": source, "page_number": page, "script": script},
    )


def _store(tmp_path, dim: int = 3) -> FAISSVectorStore:
    return FAISSVectorStore(
        index_path    = str(tmp_path / "index.faiss"),
        meta_path     = str(tmp_path / "meta.pkl"),
        embedding_dim = dim,
    )


def _vec(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


# ── FAISS add / search ─────────────────────────────────────────────────────────

def test_add_then_search_ranks_by_similarity(tmp_path):
    store = _store(tmp_path)
    store.add(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        _vec([1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]),
    )
    assert store.total_vectors == 3

    results = store.search(_vec([1, 0, 0]), top_k=3, score_threshold=0.0)
    assert [r.chunk_id for r in results] == ["a", "b", "c"]
    assert results[0].score == pytest.approx(1.0)
    # Metadata excludes the keys that are promoted to fields on the chunk.
    assert results[0].metadata["source"] == "book.pdf"
    assert "chunk_id" not in results[0].metadata


def test_search_applies_the_score_threshold(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a"), _chunk("b"), _chunk("c")],
              _vec([1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]))
    results = store.search(_vec([1, 0, 0]), top_k=3, score_threshold=0.6)
    assert [r.chunk_id for r in results] == ["a"]


def test_search_truncates_to_top_k(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a"), _chunk("b")], _vec([1, 0, 0], [0.9, 0.1, 0]))
    assert len(store.search(_vec([1, 0, 0]), top_k=1, score_threshold=0.0)) == 1


def test_filter_by_source_over_fetches_to_still_fill_top_k(tmp_path):
    store = _store(tmp_path)
    store.add(
        [_chunk("a", source="A"), _chunk("b", source="B"), _chunk("c", source="B")],
        _vec([1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]),
    )
    results = store.search(_vec([1, 0, 0]), top_k=2, score_threshold=0.0,
                           filter_by={"source": "B"})
    assert [r.chunk_id for r in results] == ["b", "c"]


def test_searching_an_empty_index_raises(tmp_path):
    with pytest.raises(IndexNotFoundError):
        _store(tmp_path).search(_vec([1, 0, 0]))


# ── removal / catalog / sources ────────────────────────────────────────────────

def test_remove_by_source_rebuilds_the_index(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a", source="A"), _chunk("b", source="B")],
              _vec([1, 0, 0], [0, 1, 0]))

    assert store.remove_by_source("A") == 1
    assert store.total_vectors == 1
    assert store.sources() == ["B"]
    assert [r.chunk_id for r in store.search(_vec([1, 0, 0]), top_k=5,
                                             score_threshold=0.0)] == ["b"]
    # Removing a source that is not present is a no-op, not an error.
    assert store.remove_by_source("missing") == 0


def test_catalog_and_sources(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a", text="alpha", source="A"), _chunk("b", text="beta", source="B")],
              _vec([1, 0, 0], [0, 1, 0]))

    catalog = store.catalog()
    assert [r["chunk_id"] for r in catalog] == ["a", "b"]
    assert catalog[0]["text"] == "alpha"
    assert store.sources() == ["A", "B"]


# ── persistence ────────────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a"), _chunk("b")], _vec([1, 0, 0], [0, 1, 0]))
    store.save()

    reloaded = _store(tmp_path)
    reloaded.load()
    assert reloaded.total_vectors == 2
    assert [r.chunk_id for r in reloaded.search(_vec([1, 0, 0]), top_k=5,
                                                score_threshold=0.0)][0] == "a"


def test_reset_clears_memory_and_disk(tmp_path):
    store = _store(tmp_path)
    store.add([_chunk("a")], _vec([1, 0, 0]))
    store.save()
    assert Path(store.index_path).exists()

    store.reset()

    assert store.total_vectors == 0
    assert not Path(store.index_path).exists()
    assert not Path(store.meta_path).exists()


# ── validation ─────────────────────────────────────────────────────────────────

def test_dimension_mismatch_is_rejected(tmp_path):
    with pytest.raises(VectorStoreError):
        _store(tmp_path, dim=3).add([_chunk("a")], _vec([1, 0]))


def test_chunk_and_vector_count_mismatch_is_rejected(tmp_path):
    with pytest.raises(VectorStoreError):
        _store(tmp_path).add([_chunk("a"), _chunk("b")], _vec([1, 0, 0]))


def test_saving_an_empty_index_is_rejected(tmp_path):
    with pytest.raises(VectorStoreError):
        _store(tmp_path).save()


# ── facade ─────────────────────────────────────────────────────────────────────

def test_unknown_backend_is_rejected():
    with pytest.raises(VectorStoreError):
        VectorStore(backend="memory")


def test_facade_remove_source_invalidates_the_bm25_cache():
    store = VectorStore(backend="faiss")

    class _Backend:
        def remove_by_source(self, source):
            return 2

    store._store = _Backend()
    store._bm25 = object()  # pretend a stale index exists

    assert store.remove_source("x.pdf", persist=False) == 2
    assert store._bm25 is None


# ── hybrid search ──────────────────────────────────────────────────────────────

def _rc(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id = cid,
        text     = text,
        score    = score,
        metadata = {"source": "book.pdf", "page_number": 1, "script": "telugu"},
    )


class _HybridBackend:
    def __init__(self, dense, records):
        self._dense = dense
        self._records = records
        self.last_threshold = None
        self.last_top_k = None

    def search(self, query_vector, top_k, score_threshold, filter_by=None):
        self.last_top_k = top_k
        self.last_threshold = score_threshold
        # Emulate the real backends, which drop anything below the threshold.
        return [c for c in self._dense if c.score >= score_threshold][:top_k]

    def catalog(self):
        return list(self._records)


def _hybrid_store(monkeypatch, dense, records) -> tuple[VectorStore, _HybridBackend]:
    monkeypatch.setattr(vector_store, "HYBRID_ENABLED", True)
    monkeypatch.setattr(vector_store, "embed_query", lambda q: np.zeros(3, dtype=np.float32))
    store = VectorStore(backend="faiss")
    backend = _HybridBackend(dense, records)
    store._store = backend
    return store, backend


def test_hybrid_does_not_open_the_abstention_gate(monkeypatch):
    """A query dense retrieval rejects must stay rejected in hybrid mode."""
    store, _ = _hybrid_store(
        monkeypatch,
        dense    = [_rc("d1", "alpha beta", 0.2)],
        records  = [{"chunk_id": "d1", "text": "alpha beta", "source": "book.pdf"}],
    )
    assert store.query("alpha", top_k=5, score_threshold=0.45) == []


def test_hybrid_surfaces_a_lexical_only_hit(monkeypatch):
    store, backend = _hybrid_store(
        monkeypatch,
        dense = [_rc("d1", "alpha beta", 0.9)],
        records = [
            {"chunk_id": "d1",  "text": "alpha beta",     "source": "book.pdf"},
            {"chunk_id": "lex", "text": "kosha rare term", "source": "book.pdf"},
        ],
    )

    results = store.query("kosha", top_k=5, score_threshold=0.45)
    ids = [r.chunk_id for r in results]

    assert set(ids) == {"d1", "lex"}
    lexical = next(r for r in results if r.chunk_id == "lex")
    assert lexical.score == 0.0
    assert lexical.retrieval_score == 0.0
    assert "hybrid_score" in lexical.metadata
    # Hybrid widens the dense candidate pool (no cosine cutoff at that stage).
    assert backend.last_threshold == 0.0


def test_hybrid_drops_a_dense_only_chunk_below_the_floor(monkeypatch):
    store, _ = _hybrid_store(
        monkeypatch,
        dense = [_rc("d1", "alpha", 0.9), _rc("d2", "beta", 0.3)],
        records = [
            {"chunk_id": "d1", "text": "alpha", "source": "book.pdf"},
            {"chunk_id": "d2", "text": "beta",  "source": "book.pdf"},
        ],
    )
    results = store.query("alpha", top_k=5, score_threshold=0.45)
    assert [r.chunk_id for r in results] == ["d1"]


def test_dense_path_is_used_when_hybrid_is_off(monkeypatch):
    monkeypatch.setattr(vector_store, "HYBRID_ENABLED", False)
    monkeypatch.setattr(vector_store, "embed_query", lambda q: np.zeros(3, dtype=np.float32))
    store = VectorStore(backend="faiss")
    backend = _HybridBackend(
        [_rc("d1", "alpha", 0.9), _rc("d2", "beta", 0.3)], []
    )
    store._store = backend

    results = store.query("alpha", top_k=5, score_threshold=0.45)
    assert [r.chunk_id for r in results] == ["d1"]
    assert backend.last_threshold == 0.45  # the cosine cutoff is applied
