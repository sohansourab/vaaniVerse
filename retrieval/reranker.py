"""
Second-stage cross-encoder reranking.

Dense retrieval scores a query and a passage *independently* and compares the
resulting vectors. That is what makes it fast enough to search a whole corpus,
and also what limits it: the two texts never see each other, so a passage can
score highly on topic while being the wrong passage.

A cross-encoder reads the query and the passage *together*, with full attention
between them. It is far more accurate and far too slow to run over a corpus —
so it runs over the ~20 candidates dense retrieval already narrowed down. That
is the standard retrieve-then-rerank shape, and it is the fix for the specific
failure documented in docs/EVALUATION.md: in-corpus and out-of-domain score
distributions that overlap, which no fixed cosine cutoff can separate.

Scores are sigmoid-normalized to (0, 1), so `score_threshold` keeps meaning
what it meant before — a relevance floor on a 0..1 scale.

This module is deliberately decoupled from `vector_store`: it scores text and
returns numbers. Selecting, ordering and thresholding chunks stays the caller's
job, which keeps this testable without an index or an embedding model.
"""
from __future__ import annotations

import logging
from typing import Any

from core.config import RERANK_DEVICE, RERANK_MODEL
from core.exceptions import RerankError

logger = logging.getLogger(__name__)

# Loaded once per process, keyed by model + device, exactly like the embedder.
_RERANKER_CACHE: dict[str, Any] = {}

# A cross-encoder over ~20 candidates does not need a large batch.
_DEFAULT_BATCH_SIZE = 16
_MAX_LENGTH = 512


# ── availability ───────────────────────────────────────────────────────────────

def probe() -> str:
    """Return "" when the reranker is usable, otherwise the reason it is not."""
    try:
        from FlagEmbedding import FlagReranker  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"FlagEmbedding is not importable: {exc}"
    return ""


def available() -> bool:
    """True when the reranker can be loaded in this environment."""
    return probe() == ""


# ── model singleton ────────────────────────────────────────────────────────────

def get_reranker(
    model_name: str = RERANK_MODEL,
    device: str = RERANK_DEVICE,
):
    """
    Load the cross-encoder once per process and cache it.

    Args:
        model_name: HuggingFace id of a sequence-classification reranker.
        device:     "cpu" or "cuda[:N]".

    Returns:
        The cached FlagReranker.

    Raises:
        RerankError: If FlagEmbedding is missing or the model cannot be loaded.
    """
    cache_key = f"{model_name}::{device}"
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    reason = probe()
    if reason:
        raise RerankError("Reranker is unavailable.", details={"reason": reason})

    from FlagEmbedding import FlagReranker

    logger.info(f"Loading reranker '{model_name}' on device '{device}'...")
    try:
        _RERANKER_CACHE[cache_key] = FlagReranker(
            model_name,
            use_fp16=(device != "cpu"),
            # Must be passed explicitly. With devices=None, FlagEmbedding claims
            # *every* visible CUDA device — which would put the reranker on the
            # same 4 GB GPU as BGE-M3 and the LLM. Defaulting to CPU here is a
            # deliberate resource decision, not an oversight.
            devices=device,
            normalize=True,   # sigmoid → (0, 1), preserving the 0..1 threshold contract
            max_length=_MAX_LENGTH,
            batch_size=_DEFAULT_BATCH_SIZE,
        )
    except Exception as exc:
        raise RerankError(
            f"Failed to load reranker '{model_name}'.",
            details={"model": model_name, "device": device, "error": str(exc)},
        )
    logger.info("Reranker loaded.")
    return _RERANKER_CACHE[cache_key]


# ── scoring ────────────────────────────────────────────────────────────────────

def _as_float_list(raw: Any) -> list[float]:
    """Normalize whatever the library returns (array, tensor, scalar) to a list."""
    if hasattr(raw, "tolist"):          # numpy array / torch tensor
        raw = raw.tolist()
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return [float(value) for value in raw]


def score_pairs(
    query: str,
    passages: list[str],
    model_name: str = RERANK_MODEL,
    device: str = RERANK_DEVICE,
) -> list[float]:
    """
    Score every (query, passage) pair with the cross-encoder.

    Each pair is scored independently, so the result is positionally aligned
    with `passages` and contains no ordering information.

    Args:
        query:    The normalized query string.
        passages: Candidate passage texts.

    Returns:
        List of floats in (0, 1) — higher is more relevant. Empty in, empty out.

    Raises:
        RerankError: On an empty query, or if scoring fails.
    """
    if not passages:
        return []

    query = (query or "").strip()
    if not query:
        raise RerankError("Cannot rerank with an empty query.", details={"passages": len(passages)})

    reranker = get_reranker(model_name, device)
    pairs = [(query, passage or "") for passage in passages]

    try:
        scores = _as_float_list(reranker.compute_score(pairs))
    except RerankError:
        raise
    except Exception as exc:
        raise RerankError(
            "Reranker scoring failed.",
            details={"pairs": len(pairs), "model": model_name, "error": str(exc)},
        )

    if len(scores) != len(pairs):
        raise RerankError(
            "Reranker returned a score count that does not match the candidate count.",
            details={"passages": len(pairs), "scores": len(scores)},
        )

    return scores
