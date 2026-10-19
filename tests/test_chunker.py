"""Unit tests for the sentence-aware chunker.

Offline: the Telugu branch is rule-based and the English branch falls back to a
blank spaCy sentencizer when `en_core_web_sm` is not installed, so no model is
downloaded to run these.
"""
from __future__ import annotations

import pytest

from core.exceptions import ChunkingError
from ingestion import chunker
from ingestion.chunker import (
    _build_chunks,
    _estimate_tokens,
    _split_telugu_sentences,
    chunk_document,
    chunk_page,
)
from ingestion.parser import ParsedDocument, ParsedPage


# ── helpers ────────────────────────────────────────────────────────────────────

def _page(number: int, text: str, **meta) -> ParsedPage:
    return ParsedPage(
        page_number     = number,
        raw_text        = text,
        normalized_text = text,
        images          = [],
        metadata        = {"source": "book.pdf", "page_number": number, **meta},
    )


def _doc(*pages: ParsedPage) -> ParsedDocument:
    return ParsedDocument(
        source_path  = "book.pdf",
        total_pages  = len(pages),
        pages        = list(pages),
        doc_metadata = {"source": "book.pdf"},
    )


# ── token estimation ───────────────────────────────────────────────────────────

def test_telugu_tokens_are_weighted_higher_than_english():
    """BGE-M3's SentencePiece tokenizer produces more tokens per Telugu word."""
    telugu  = " ".join(["ఎరుక"] * 10)
    english = " ".join(["word"] * 10)

    assert _estimate_tokens(telugu) == 14   # 10 words * 1.4
    assert _estimate_tokens(english) == 11  # 10 words * 1.1


# ── Telugu sentence splitting ───────────────────────────────────────────────────

def test_telugu_splitter_keeps_the_delimiter_with_its_sentence():
    sentences = _split_telugu_sentences("ఎరుక అంటే ఏమిటి? నిజం అనేది సత్యం.")
    assert sentences == ["ఎరుక అంటే ఏమిటి?", "నిజం అనేది సత్యం."]


def test_telugu_splitter_handles_dandas():
    sentences = _split_telugu_sentences("మొదటి వాక్యం। రెండవ వాక్యం॥")
    assert sentences == ["మొదటి వాక్యం।", "రెండవ వాక్యం॥"]


# ── greedy packing ──────────────────────────────────────────────────────────────

def test_chunks_are_packed_until_the_size_limit():
    sentences = ["aaaa bbbb", "cccc dddd", "eeee ffff"]  # 2 tokens each
    chunks = _build_chunks(sentences, chunk_size=4, chunk_overlap=0)
    assert chunks == ["aaaa bbbb cccc dddd", "eeee ffff"]


def test_overlap_window_reseeds_the_next_chunk():
    sentences = ["aaaa bbbb", "cccc dddd", "eeee ffff"]
    chunks = _build_chunks(sentences, chunk_size=4, chunk_overlap=2)
    # The last sentence of chunk 1 reappears at the head of chunk 2.
    assert chunks == ["aaaa bbbb cccc dddd", "cccc dddd eeee ffff"]


def test_oversized_sentence_is_emitted_on_its_own():
    sentences = ["x", "one two three four"]  # 4 tokens, over the limit of 3
    chunks = _build_chunks(sentences, chunk_size=3, chunk_overlap=0)
    assert chunks == ["x", "one two three four"]


def test_no_sentences_yields_no_chunks():
    assert _build_chunks([], chunk_size=100, chunk_overlap=0) == []


# ── document walk ───────────────────────────────────────────────────────────────

def test_chunk_document_ids_and_metadata():
    telugu = "ఎరుక అంటే ఏమిటి? నిజం అనేది సత్యం."
    chunks = chunk_document(_doc(_page(1, telugu)))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "book.pdf_p1_0"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["page_number"] == 1
    assert chunk.metadata["script"] == "telugu"
    assert chunk.token_len > 0


def test_chunk_document_skips_blank_pages():
    chunks = chunk_document(_doc(_page(1, "   \n  "), _page(2, "ఎరుక ఉంది.")))
    assert [c.metadata["page_number"] for c in chunks] == [2]


def test_chunk_document_numbers_chunks_per_page():
    long_page = " ".join(["ఎరుక ఉంది."] * 200)  # forces multiple chunks
    chunks = chunk_document(_doc(_page(1, long_page)), chunk_size=20, chunk_overlap=0)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id.endswith("_p1_0")


def test_chunk_document_wraps_split_failures(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(chunker, "_split_into_sentences", boom)
    with pytest.raises(ChunkingError) as exc:
        chunk_document(_doc(_page(1, "ఏదో")))
    assert exc.value.details["page"] == 1


# ── single-page helper ──────────────────────────────────────────────────────────

def test_chunk_page_returns_nothing_for_blank_text():
    assert chunk_page(_page(1, "   "), source_name="x.pdf") == []


def test_chunk_page_uses_the_supplied_source_name():
    chunks = chunk_page(_page(3, "ఎరుక ఉంది."), source_name="given.pdf")
    assert chunks[0].chunk_id == "given.pdf_p3_0"
