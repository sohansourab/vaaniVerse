"""Unit tests for the claim ↔ citation verifier.

Offline and deterministic: the checks are lexical, so no model is involved.
"""
from __future__ import annotations

from reasoning.verify import verify_answer

_TELUGU_SOURCES = [
    "ఎరుక అనేది శరీరానికి బయట ఉంది మరియు అంతటా వ్యాపించి ఉంది.",
    "సముద్రం మరియు అలలు ఒకటే నీరు, వేరు కాదు.",
    "గాఢ నిద్రలో కూడా ఎరుక ఉంటుంది.",
]


def test_in_range_citation_with_overlap_passes():
    answer = "ఎరుక శరీరానికి బయట ఉంది [SOURCE 1]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.invalid_citations == []
    assert result.unsupported_claims == 0
    assert result.passed is True


def test_out_of_range_citation_is_always_an_error():
    answer = "Something invented [SOURCE 7]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.invalid_citations == [7]
    assert result.passed is False


def test_unsupported_claim_is_flagged():
    # Cites source 1, but the claim is about source 2's content and shares no
    # content tokens with source 1.
    answer = "చాక్లెట్ కేక్ కాల్చడం సులభం [SOURCE 1]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.unsupported_claims == 1
    assert result.passed is False
    assert result.to_dict()["flagged"][0]["reason"] == "unsupported"


def test_cross_script_claim_is_not_failed():
    """An English answer translating a Telugu passage cannot be judged lexically."""
    answer = "Awareness extends beyond the body [SOURCE 1]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.unsupported_claims == 0
    assert result.comparable_claims == 0
    assert result.passed is True


def test_uncited_claims_are_counted_but_not_invalid():
    answer = "ఎరుక అంటే ఏమిటి? ఇది ఒక ప్రశ్న [SOURCE 1]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.uncited_claims >= 1
    assert result.invalid_citations == []


def test_multi_citation_claim_uses_the_union_of_cited_sources():
    answer = "గాఢ నిద్రలో ఎరుక ఉంటుంది [SOURCE 3]."
    result = verify_answer(answer, _TELUGU_SOURCES)
    assert result.unsupported_claims == 0


def test_no_sources_at_all_marks_every_citation_invalid():
    result = verify_answer("An answer [SOURCE 1].", [])
    assert result.invalid_citations == [1]
    assert result.passed is False
