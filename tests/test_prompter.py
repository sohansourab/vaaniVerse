"""
Tests for prompt-side answer-language selection.

Regression locked down here: detect_script() reports "roman" for two different
things — genuine English, and Romanized Telugu. Every "roman" query used to be
routed to _SYSTEM_PROMPT_ENGLISH, whose rule 2 forbids Telugu, which made
_SYSTEM_PROMPT_TELUGU's own rule for Romanized input unreachable. The visible
bug was that "nenu evaru?" was answered in English.

These tests deliberately drive the real normalizer rather than hand-built
strings, so they exercise the same flow the API does: detect the script of the
raw query, then choose the prompt from the *normalized* one.
"""
import pytest

from ingestion.normalizer import detect_script, normalize
from reasoning.prompter import (
    _SYSTEM_PROMPT_ENGLISH,
    _SYSTEM_PROMPT_MIXED,
    _SYSTEM_PROMPT_TELUGU,
    _answer_language,
    _select_system_prompt,
    build_no_result_prompt,
    build_prompt,
)
from retrieval.vector_store import RetrievedChunk


def _flow(raw: str) -> tuple[str, str]:
    """Mirror api.main._normalize_query: detect on the raw query, prompt on the normalized one."""
    return normalize(raw, transliterate=True), detect_script(raw)


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id = "book_p1_0",
        text     = "జీవుడు దేహమును నిజమని భ్రమించుచున్నాడు.",
        score    = 0.71,
        metadata = {"source": "book.pdf", "page_number": 1},
    )


# ── _answer_language ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["nenu evaru?", "eruka ante emiti?", "oka", "jivudu ante emiti"])
def test_romanized_telugu_resolves_to_telugu(raw):
    query, script = _flow(raw)
    assert script == "roman", "precondition: the script detector cannot tell these apart"
    assert _answer_language(script, query) == "telugu"


@pytest.mark.parametrize("raw", ["who am i?", "what is dharma?", "Explain the ocean and waves analogy"])
def test_english_resolves_to_english(raw):
    query, script = _flow(raw)
    assert _answer_language(script, query) == "english"


def test_native_telugu_resolves_to_telugu():
    assert _answer_language("telugu", "నేను ఎవరు?") == "telugu"


def test_mixed_script_resolves_to_mixed():
    assert _answer_language("mixed", "నేను evaru?") == "mixed"


def test_unknown_script_resolves_to_english():
    """No alphabetic characters leaves nothing to infer from; English is the safe default."""
    assert _answer_language("unknown", "???") == "english"


# ── _select_system_prompt ─────────────────────────────────────────────────────

def test_select_system_prompt_romanized_gets_telugu_prompt():
    query, script = _flow("nenu evaru?")
    assert _select_system_prompt(script, query) is _SYSTEM_PROMPT_TELUGU


def test_select_system_prompt_english_gets_english_prompt():
    query, script = _flow("who am i?")
    assert _select_system_prompt(script, query) is _SYSTEM_PROMPT_ENGLISH


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_uses_telugu_prompt_for_romanized_query():
    query, script = _flow("nenu evaru?")
    package = build_prompt(query, [_chunk()], script)
    assert package.system_prompt is _SYSTEM_PROMPT_TELUGU
    # The model is shown the transliterated query, so it is not asked to answer
    # Romanized input on the strength of rule 5 alone.
    assert "నేను ఎవరు?" in package.user_prompt


def test_build_prompt_uses_english_prompt_for_english_query():
    query, script = _flow("who am i?")
    package = build_prompt(query, [_chunk()], script)
    assert package.system_prompt is _SYSTEM_PROMPT_ENGLISH
    assert "who am i?" in package.user_prompt


# ── build_no_result_prompt ────────────────────────────────────────────────────

def test_no_result_prompt_is_telugu_for_romanized_query():
    query, script = _flow("nenu evaru?")
    package = build_no_result_prompt(query, script)
    assert package.system_prompt is _SYSTEM_PROMPT_TELUGU
    assert "ఈ సమాచారం అందుబాటులో లేదు." in package.user_prompt
    assert "Respond in English only." not in package.user_prompt


def test_no_result_prompt_is_english_for_english_query():
    query, script = _flow("who am i?")
    package = build_no_result_prompt(query, script)
    assert package.system_prompt is _SYSTEM_PROMPT_ENGLISH
    assert "The requested information is not available" in package.user_prompt
    assert "Respond in English only." in package.user_prompt


def test_no_result_prompt_for_mixed_query_does_not_demand_english():
    """A mixed query previously got the MIXED prompt but an English-only suffix — a contradiction."""
    package = build_no_result_prompt("నేను evaru?", "mixed")
    assert package.system_prompt is _SYSTEM_PROMPT_MIXED
    assert "Respond in English only." not in package.user_prompt
    # _SYSTEM_PROMPT_MIXED says to match the query's language mix, so the forced
    # fallback sentence must itself be bilingual rather than English-only.
    assert "ఈ సమాచారం అందుబాటులో లేదు." in package.user_prompt
    assert "The requested information is not available" in package.user_prompt
