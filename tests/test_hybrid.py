"""Unit tests for BM25 lexical retrieval and Reciprocal Rank Fusion.

Fully offline and model-free — this is the point of keeping hybrid search in its
own module: the fusion maths can be asserted without a vector store.
"""
from __future__ import annotations

from retrieval.hybrid import BM25Index, reciprocal_rank_fusion, tokenize


# ── tokenization ──────────────────────────────────────────────────────────────

def test_tokenizer_keeps_telugu_words_whole():
    tokens = tokenize("ఎరుక అంటే ఏమిటి?")
    assert "ఎరుక" in tokens
    assert "ఏమిటి" in tokens
    # Punctuation is dropped, not attached to the word.
    assert "ఏమిటి?" not in tokens


def test_tokenizer_lowercases_roman_text():
    assert tokenize("What Is Awareness?") == ["what", "is", "awareness"]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _index() -> BM25Index:
    index = BM25Index()
    index.build(
        ["a", "b", "c"],
        [
            "awareness is the witness in deep sleep",
            "the ocean and the waves are one water",
            "how do I bake a chocolate cake",
        ],
    )
    return index


def test_bm25_finds_a_literal_term_match():
    results = dict(_index().search("awareness", top_k=3))
    assert "a" in results
    assert results["a"] > 0


def test_bm25_returns_nothing_for_unmatched_terms():
    assert _index().search("quantum chromodynamics", top_k=3) == []


def test_bm25_ranks_the_on_topic_document_first():
    ranked = [doc_id for doc_id, _ in _index().search("ocean waves", top_k=3)]
    assert ranked[0] == "b"


def test_bm25_add_and_remove_keep_the_index_in_sync():
    index = _index()
    index.add(["d"], ["awareness of awareness is awareness"])
    assert "d" in dict(index.search("awareness", top_k=5))

    index.remove({"d"})
    assert "d" not in dict(index.search("awareness", top_k=5))
    assert index.size == 3


# ── reciprocal rank fusion ────────────────────────────────────────────────────

def test_rrf_rewards_agreement_between_the_two_rankers():
    # "b" is second on both lists; "a" tops dense but is absent from lexical.
    fused = dict(reciprocal_rank_fusion(["a", "b"], ["c", "b"], alpha=0.5, k=60))
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_rrf_alpha_one_is_dense_only():
    fused = [doc_id for doc_id, _ in reciprocal_rank_fusion(["a", "b"], ["b", "a"], alpha=1.0, k=60)]
    assert fused == ["a", "b"]


def test_rrf_alpha_zero_is_lexical_only():
    fused = [doc_id for doc_id, _ in reciprocal_rank_fusion(["a", "b"], ["b", "a"], alpha=0.0, k=60)]
    assert fused == ["b", "a"]


def test_rrf_surfaces_a_document_dense_missed():
    """The whole reason hybrid exists: a lexical-only hit must be reachable."""
    fused = [doc_id for doc_id, _ in reciprocal_rank_fusion(["a"], ["z"], alpha=0.5, k=60)]
    assert set(fused) == {"a", "z"}


def test_rrf_is_empty_for_two_empty_lists():
    assert reciprocal_rank_fusion([], [], alpha=0.5, k=60) == []
