"""
Automatic claim ↔ citation checks on a generated answer.

The prompt requires every claim to end with a ``[SOURCE N]`` whose passage
supports it. Nothing verified that, so a model could cite a page that does not
mention the thing it just said, or cite ``[SOURCE 7]`` against six sources, and
the API would return it as a grounded answer.

Two checks, in increasing strictness:

  1. STRUCTURAL — every ``[SOURCE N]`` must point at a source that was actually
     supplied (1 ≤ N ≤ provided). This is exact, language-independent, and the
     one check that catches a real hallucination class on its own.

  2. SUPPORT — the text of a cited claim should overlap the text of the passage
     it cites. This is lexical (token containment), so it is only run when the
     claim and the passage are in a comparable script. An English answer that
     *translates* a Telugu passage cannot be checked this way, and is reported
     as ``comparable=False`` rather than silently failed or silently passed.

This is a smoke alarm, not a judge: it catches the obvious failure and its
output is surfaced to the client, not used to rewrite the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.config import CITATION_SUPPORT_MIN
from ingestion.normalizer import detect_script, normalize

# [SOURCE 3] — the only citation form the prompt asks for. Tolerant of spacing.
_CITATION_RE = re.compile(r"\[\s*SOURCE\s+(\d+)\s*\]", re.IGNORECASE)

# Sentence-ish split. Kept simple on purpose: over-segmenting a Telugu sentence
# would split a claim away from its citation.
_SENTENCE_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")

# Word tokenizer that keeps Indic words whole: Telugu vowel signs and the virama
# are combining marks (category Mn), which ``\w`` does not match.
_WORD_RE = re.compile(r"[\w\u0300-\u036f\u0900-\u0dff]+", re.UNICODE)

# Very common words carry no evidence of support; ignoring them stops "the",
# "is", "and" from inflating a weak match.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "which", "with",
})


@dataclass
class ClaimCheck:
    text: str
    citations: list[int] = field(default_factory=list)
    support: float = 0.0
    comparable: bool = True
    supported: bool = True

    @property
    def unsupported(self) -> bool:
        return self.comparable and not self.supported


@dataclass
class VerificationResult:
    claims: list[ClaimCheck] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_claims: int = 0
    comparable_claims: int = 0
    unsupported_claims: int = 0

    @property
    def passed(self) -> bool:
        return not self.invalid_citations and self.unsupported_claims == 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_claims": len(self.claims),
            "uncited_claims": self.uncited_claims,
            "invalid_citations": self.invalid_citations,
            "comparable_claims": self.comparable_claims,
            "unsupported_claims": self.unsupported_claims,
            "min_support": CITATION_SUPPORT_MIN,
            "flagged": [
                {
                    "claim": c.text[:300],
                    "citations": c.citations,
                    "support": round(c.support, 4),
                    "reason": "unsupported",
                }
                for c in self.claims if c.unsupported
            ],
        }


def _content_tokens(text: str) -> set[str]:
    return {
        tok for tok in _WORD_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }


def _split_claims(answer: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(answer or "") if p.strip()]
    return parts


def verify_answer(
    answer: str,
    source_texts: list[str],
    *,
    min_support: float = CITATION_SUPPORT_MIN,
) -> VerificationResult:
    """
    Check ``answer`` against the ``source_texts`` it was allowed to cite.

    ``source_texts[i]`` is the passage behind ``[SOURCE i+1]``.
    """
    result = VerificationResult()
    n_sources = len(source_texts)

    # Tokenize the sources once. Script is detected on the text *as written*:
    # the transliterator is deliberately not applied here, because running
    # English through it produces Telugu-script noise, which would make an
    # English claim look comparable to a Telugu source and fail it falsely.
    normalized_sources = [normalize(t, transliterate=False) for t in source_texts]
    source_tokens = [_content_tokens(t) for t in normalized_sources]
    source_scripts = [detect_script(t) for t in normalized_sources]

    for claim_text in _split_claims(answer):
        citations = [int(m) for m in _CITATION_RE.findall(claim_text)]
        check = ClaimCheck(text=claim_text, citations=citations)

        if not citations:
            result.uncited_claims += 1
            result.claims.append(check)
            continue

        # Structural check — an out-of-range source number is always an error.
        for n in citations:
            if n < 1 or n > n_sources:
                if n not in result.invalid_citations:
                    result.invalid_citations.append(n)

        # Support check — over the union of the cited, in-range passages.
        usable = [n for n in citations if 1 <= n <= n_sources]
        if not usable:
            check.comparable = False
            check.supported = False
            result.claims.append(check)
            continue

        raw_claim = _CITATION_RE.sub(" ", claim_text)
        claim_script = detect_script(raw_claim)
        claim_tokens = _content_tokens(normalize(raw_claim, transliterate=False))

        cited_scripts = {source_scripts[n - 1] for n in usable}
        # Comparable only when both sides share a script family. A Latin-script
        # claim against a Telugu passage is a translation (or an English answer),
        # which lexical overlap cannot judge. Romanized Telugu is reported as
        # comparable=False too, rather than silently passed or silently failed.
        comparable = bool(claim_tokens) and (
            claim_script in cited_scripts
            or "mixed" in cited_scripts
            or claim_script == "mixed"
        )
        check.comparable = comparable
        result.comparable_claims += int(comparable)

        if not comparable or not claim_tokens:
            # Nothing to assert; do not call it unsupported.
            check.supported = True
            result.claims.append(check)
            continue

        cited_tokens: set[str] = set()
        for n in usable:
            cited_tokens |= source_tokens[n - 1]
        overlap = len(claim_tokens & cited_tokens) / len(claim_tokens)
        check.support = overlap
        check.supported = overlap >= min_support
        if check.unsupported:
            result.unsupported_claims += 1

        result.claims.append(check)

    return result
