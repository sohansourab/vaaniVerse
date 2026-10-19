"""
Lexical (BM25) retrieval and rank fusion for hybrid search.

Why this exists: dense retrieval alone is weakest exactly where a heritage
corpus is hardest. BGE-M3 embeds *meaning*, so a rare proper name, a
transliterated title, or a term like ``emiti``/``ఏమిటి`` that the model has
seen rarely lands in a fuzzy neighbourhood and can be out-ranked by a
topically-similar passage about something else. BM25 is blind to meaning and
exact about tokens — it is the complement, not a replacement.

The two produce scores on incomparable scales (cosine similarity vs. an
unbounded BM25 sum), so fusing them by adding scores would need calibration
nobody has. Reciprocal Rank Fusion sidesteps that by fusing *ranks*:

    score(d) = α / (k + rank_dense(d)) + (1 − α) / (k + rank_bm25(d))

with ``k`` = RRF_K (60, the standard from Cormack et al. 2009) and α =
HYBRID_ALPHA. A document missing from one list simply contributes nothing from
it, which is how a name-match that dense retrieval missed can still surface.

No external dependency: a corpus of a few thousand chunks does not justify
``rank_bm25``, and this keeps the retrieval path importable without it.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Unicode-aware word tokenizer. ``\w`` alone is not enough: Telugu vowel signs
# and the virama are combining marks (category Mn), which ``\w`` excludes, so
# "ఎరుక" would be shredded into "ఎర"/"క". The explicit Mark ranges keep an Indic
# word whole.
_TOKEN_RE = re.compile(r"[\w\u0300-\u036f\u0900-\u0dff]+", re.UNICODE)

# Standard BM25 constants. k1 controls term-frequency saturation, b controls
# length normalization.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lower-cased Unicode word tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    In-memory Okapi BM25 over a set of documents keyed by an id.

    Built once and cached by the caller; ``add``/``remove`` keep it in sync with
    the vector store as documents are ingested.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()     # term → number of docs
        self._term_freqs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._avg_len: float = 0.0

    # ── construction ───────────────────────────────────────────────────────────

    def build(self, ids: list[str], texts: list[str]) -> None:
        """Replace the whole index."""
        self._ids = list(ids)
        self._doc_tokens = [tokenize(t or "") for t in texts]
        self._reindex()

    def add(self, ids: list[str], texts: list[str]) -> None:
        """Append documents."""
        self._ids.extend(ids)
        self._doc_tokens.extend(tokenize(t or "") for t in texts)
        self._reindex()

    def remove(self, ids: set[str]) -> None:
        """Drop documents whose id is in ``ids``."""
        keep = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids]
        self._ids = [self._ids[i] for i in keep]
        self._doc_tokens = [self._doc_tokens[i] for i in keep]
        self._reindex()

    def _reindex(self) -> None:
        self._doc_freq = Counter()
        self._term_freqs = []
        self._doc_len = []
        for tokens in self._doc_tokens:
            counts = Counter(tokens)
            self._term_freqs.append(counts)
            self._doc_len.append(len(tokens))
            for term in counts:
                self._doc_freq[term] += 1
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    # ── queries ────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """
        Return ``[(doc_id, score), …]`` for the best BM25 matches, best first.

        Documents scoring 0 (no query term present) are omitted: RRF rewards
        presence in a list, and padding a list with zeros would be noise.
        """
        terms = tokenize(query)
        if not terms or not self._ids:
            return []

        n = len(self._ids)
        scores: dict[str, float] = {}
        for term in set(terms):
            df = self._doc_freq.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc_id in enumerate(self._ids):
                tf = self._term_freqs[i].get(term)
                if not tf:
                    continue
                dl = self._doc_len[i] or 1
                denom = tf + _K1 * (1 - _B + _B * dl / (self._avg_len or 1.0))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (_K1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


def reciprocal_rank_fusion(
    dense_ids: list[str],
    lexical_ids: list[str],
    *,
    alpha: float = 0.5,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse two ranked id lists into one, by rank rather than by score.

    Args:
        dense_ids:   Ids ordered best-first by dense retrieval.
        lexical_ids: Ids ordered best-first by BM25.
        alpha:       Weight on the dense ranking in [0, 1].
        k:           RRF damping constant.

    Returns:
        ``[(id, fused_score), …]`` sorted best-first.
    """
    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + alpha / (k + rank + 1)
    for rank, doc_id in enumerate(lexical_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + (1.0 - alpha) / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
