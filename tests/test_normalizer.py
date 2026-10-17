"""Unit tests for normalization, script detection and transliteration.

Fully offline — `ingestion.normalizer` is pure Python and touches no model,
no network and no GPU.
"""
from __future__ import annotations

import pytest

from core.exceptions import NormalizationError
from ingestion.normalizer import (
    detect_script,
    is_english_query,
    normalize,
    repair_telugu_unicode,
    transliterate_roman_to_telugu,
    unicode_normalize,
)


# ── script detection ──────────────────────────────────────────────────────────

def test_detect_script_classifies_the_three_query_modes():
    assert detect_script("నేను ఎవరు?") == "telugu"
    assert detect_script("nenu evaru?") == "roman"
    assert detect_script("నేను evaru?") == "mixed"
    assert detect_script("12345 !!!") == "unknown"


def test_is_english_query_needs_two_stopwords():
    assert is_english_query("What is awareness?") is True
    assert is_english_query("Who am I?") is True
    # Zero stopwords — treated as Romanized Telugu, not English.
    assert is_english_query("eruka ante emiti?") is False


# ── abugida transliteration ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "roman,expected",
    [
        # consonant + vowel must produce a MATRA, not an independent vowel
        ("eruka", "ఎరుక"),
        ("nija", "నిజ"),
        ("mariyu", "మరియు"),
        ("alalu", "అలలు"),
        # word-final "m" becomes anusvara
        ("samudram", "సముద్రం"),
        # consonant cluster takes a virama
        ("svarupam", "స్వరుపం"),
    ],
)
def test_transliterate_uses_dependent_vowel_signs(roman: str, expected: str):
    assert transliterate_roman_to_telugu(roman) == expected


def test_independent_vowels_when_no_consonant_precedes():
    # "a" at the start has no consonant to attach to, so it stays independent.
    assert transliterate_roman_to_telugu("alalu").startswith("అ")


def test_regression_old_flat_substitution_is_gone():
    """The old table emitted independent vowels after consonants.

    "eruka" used to become "ఎరఉకఅ" — three separate vowel letters instead of
    two dependent signs. Guard against that shape returning.
    """
    assert "ఎరఉకఅ" not in transliterate_roman_to_telugu("eruka")
    assert transliterate_roman_to_telugu("eruka") == "ఎరుక"


def test_plain_t_d_n_map_to_the_dental_series():
    """ISO 15919 convention — retroflex is not expressible in ASCII here."""
    assert transliterate_roman_to_telugu("samudram") == "సముద్రం"
    assert "డ" not in transliterate_roman_to_telugu("samudram")


# ── normalize() routing ──────────────────────────────────────────────────────

def test_pure_roman_telugu_is_now_transliterated():
    """Regression: this path previously only ran the phrase map.

    Four of the five Romanized questions in `eval/gold_qa.jsonl` reached the
    embedder as raw ASCII because the syllable walk was gated to mixed script.
    """
    out = normalize("eruka ante emiti?")
    assert out != "eruka ante emiti?"
    assert "ఎరుక" in out


def test_english_queries_are_left_alone():
    for english in (
        "What is the purpose of self-inquiry?",
        "How do I install FAISS with pip?",
        "Who is the current Chief Minister of Andhra Pradesh?",
    ):
        assert normalize(english) == english


def test_telugu_passes_through_unchanged():
    assert normalize("నేను ఎవరు?") == "నేను ఎవరు?"


def test_transliterate_flag_can_be_disabled():
    assert normalize("eruka", transliterate=False) == "eruka"


def test_known_limitation_stopword_poor_english_is_transliterated():
    """Documented failure mode, not desired behaviour.

    The English guard counts stopwords, so a stopword-free English string is
    indistinguishable from Romanized Telugu and gets transliterated. Fixing
    this properly needs a language identifier, not a rule.
    """
    assert normalize("hello") == "హెల్లొ"


def test_known_limitation_phonetic_spellings_are_not_recoverable():
    """Ad-hoc spellings are not a consistent scheme, so no rule set fixes them.

    "emiti" is spelled for ఏమిటి and "ante" for అంటే; both render differently.
    Asserted so that any future change to these outputs is noticed.
    """
    assert transliterate_roman_to_telugu("emiti") == "ఎమితి"
    assert transliterate_roman_to_telugu("ante") == "అన్తె"


# ── unicode repair ───────────────────────────────────────────────────────────

def test_unicode_repair_fixes_pdf_artifacts():
    assert repair_telugu_unicode("స్ వభావం") == "స్వభావం"
    assert repair_telugu_unicode("స్వభావం" + "\u200b") == "స్వభావం"
    assert repair_telugu_unicode("ాా") == "ా"


def test_nfc_normalization_is_idempotent_and_returns_str():
    once = unicode_normalize("నేను ెఎ")
    assert isinstance(once, str)
    assert unicode_normalize(once) == once


# ── error handling ───────────────────────────────────────────────────────────

def test_normalize_rejects_non_strings():
    with pytest.raises(NormalizationError):
        normalize(None)  # type: ignore[arg-type]
    with pytest.raises(NormalizationError):
        normalize(42)  # type: ignore[arg-type]
