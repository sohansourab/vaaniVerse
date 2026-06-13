import logging
from typing import Union

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from core.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)
from core.exceptions import EmbeddingError
from ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

# ── Model singleton ────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, BGEM3FlagModel] = {}


def _load_model(model_name: str, device: str) -> BGEM3FlagModel:
    """
    Loads BGE-M3 once per process and caches it.
    BGE-M3 produces three embedding types simultaneously:
      - dense   : standard 1024-dim semantic vector (used for RAG retrieval)
      - sparse  : BM25-style lexical weights        (used for hybrid search)
      - colbert : late-interaction token embeddings  (used for reranking)
    We load all three but expose dense as default for FAISS/Qdrant indexing.
    """
    cache_key = f"{model_name}::{device}"
    if cache_key not in _MODEL_CACHE:
        logger.info(f"Loading BGE-M3 model '{model_name}' on device '{device}'...")
        try:
            _MODEL_CACHE[cache_key] = BGEM3FlagModel(
                model_name,
                use_fp16=(device != "cpu"),   # fp16 on GPU, fp32 on CPU
                device=device,
            )
            logger.info("BGE-M3 model loaded successfully.")
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load BGE-M3 model '{model_name}'.",
                details={"model": model_name, "device": device, "error": str(exc)},
            )
    return _MODEL_CACHE[cache_key]


def get_model() -> BGEM3FlagModel:
    """Public accessor — returns the cached BGE-M3 model instance."""
    return _load_model(EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)


# ── Core embedding functions ───────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
    return_sparse: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, list[dict]]]:
    """
    Embeds a list of raw strings using BGE-M3 dense vectors.

    Args:
        texts:          List of normalized strings to embed.
        batch_size:     Number of texts per forward pass.
        show_progress:  Display tqdm progress bar.
        return_sparse:  If True, also returns sparse lexical weights
                        alongside dense vectors (for hybrid retrieval).

    Returns:
        If return_sparse=False: np.ndarray of shape (N, 1024), dtype float32
        If return_sparse=True:  tuple(dense_array, list of sparse weight dicts)

    Raises:
        EmbeddingError: On model inference failure or shape mismatch.
    """
    if not texts:
        raise EmbeddingError(
            "embed_texts() received an empty list.",
            details={"texts_count": 0},
        )

    model = get_model()

    all_dense:  list[np.ndarray] = []
    all_sparse: list[dict]       = []

    batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    iterator = tqdm(batches, desc="Embedding batches", unit="batch") if show_progress else batches

    for batch in iterator:
        try:
            output = model.encode(
                batch,
                batch_size=len(batch),
                max_length=512,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,   # colbert reserved for reranking phase
            )
        except Exception as exc:
            raise EmbeddingError(
                "BGE-M3 inference failed during batch encoding.",
                details={
                    "batch_size":    len(batch),
                    "sample_text":   batch[0][:80] if batch else "",
                    "error":         str(exc),
                },
            )

        dense_batch = output["dense_vecs"]

        # Normalize to numpy float32 regardless of torch/numpy backend
        if isinstance(dense_batch, torch.Tensor):
            dense_batch = dense_batch.cpu().float().numpy()
        else:
            dense_batch = np.array(dense_batch, dtype=np.float32)

        all_dense.append(dense_batch)

        if return_sparse:
            # sparse_vecs is a list of {token_id: weight} dicts
            all_sparse.extend(output.get("lexical_weights", [{}] * len(batch)))

    dense_matrix = np.vstack(all_dense)  # shape: (N, 1024)

    if dense_matrix.shape[0] != len(texts):
        raise EmbeddingError(
            "Embedding output count does not match input count.",
            details={
                "expected": len(texts),
                "received": dense_matrix.shape[0],
            },
        )

    if return_sparse:
        return dense_matrix, all_sparse

    return dense_matrix


def embed_chunks(
    chunks: list[TextChunk],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
    return_sparse: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, list[dict]]]:
    """
    Convenience wrapper — extracts `.text` from TextChunk objects
    and delegates to embed_texts().

    Args:
        chunks:         Output of ingestion.chunker.chunk_document()
        batch_size:     Texts per forward pass.
        show_progress:  tqdm progress bar.
        return_sparse:  Also return sparse lexical weights.

    Returns:
        Same as embed_texts() — dense array or (dense, sparse) tuple.
        Row i corresponds to chunks[i].
    """
    if not chunks:
        raise EmbeddingError(
            "embed_chunks() received an empty chunk list.",
            details={"chunks_count": 0},
        )

    texts = [chunk.text for chunk in chunks]
    return embed_texts(texts, batch_size=batch_size,
                       show_progress=show_progress, return_sparse=return_sparse)


def embed_query(
    query: str,
    return_sparse: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """
    Embeds a single query string.
    BGE-M3 applies a different instruction prefix internally for queries
    vs. passages — this is handled automatically by FlagEmbedding when
    encode() is called with a single string vs. a list.

    Args:
        query:          Normalized query string (post normalizer.normalize()).
        return_sparse:  Also return sparse lexical weights for hybrid search.

    Returns:
        If return_sparse=False: np.ndarray of shape (1024,), dtype float32
        If return_sparse=True:  tuple(vector_1024, sparse_weight_dict)

    Raises:
        EmbeddingError: On empty query or inference failure.
    """
    query = query.strip()
    if not query:
        raise EmbeddingError(
            "embed_query() received an empty string.",
            details={"query": query},
        )

    model = get_model()

    try:
        output = model.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
    except Exception as exc:
        raise EmbeddingError(
            "BGE-M3 query encoding failed.",
            details={"query_snippet": query[:80], "error": str(exc)},
        )

    dense = output["dense_vecs"]

    if isinstance(dense, torch.Tensor):
        dense = dense.cpu().float().numpy()
    else:
        dense = np.array(dense, dtype=np.float32)

    dense = dense[0]  # shape: (1024,)

    if return_sparse:
        sparse = output.get("lexical_weights", [{}])[0]
        return dense, sparse

    return dense


# ── Similarity utilities ───────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two L2-normalized vectors.
    BGE-M3 dense outputs are already L2-normalized — dot product == cosine sim.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def batch_cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity of one query against N corpus vectors.

    Args:
        query_vec:      shape (D,)
        corpus_matrix:  shape (N, D)

    Returns:
        np.ndarray of shape (N,) — similarity scores in [0, 1]
    """
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norm = corpus_matrix / (
        np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10
    )
    return corpus_norm @ query_norm   # shape: (N,)


EMBEDDING_DIM: int = 1024   # BGE-M3 dense vector dimensionality — consumed by vector_store.py