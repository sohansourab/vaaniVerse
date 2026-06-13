import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from core.config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    SCORE_THRESHOLD,
    TOP_K,
    VECTOR_BACKEND,
)
from core.exceptions import IndexNotFoundError, VectorStoreError
from ingestion.chunker import TextChunk
from retrieval.embeddings import EMBEDDING_DIM, embed_chunks, embed_query

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

class RetrievedChunk:
    __slots__ = ("chunk_id", "text", "score", "metadata")

    def __init__(self, chunk_id: str, text: str, score: float, metadata: dict):
        self.chunk_id = chunk_id
        self.text     = text
        self.score    = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(chunk_id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"source={self.metadata.get('source', '?')!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FAISS Backend
# ══════════════════════════════════════════════════════════════════════════════

class FAISSVectorStore:
    """
    Local FAISS vector store using IndexFlatIP (inner product).
    BGE-M3 dense vectors are L2-normalized — inner product == cosine similarity.

    Persistence:
        - index    → .faiss  (FAISS binary format)
        - metadata → .pkl    (list of dicts, positionally aligned with index)
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        meta_path:  str = FAISS_META_PATH,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.index_path    = index_path
        self.meta_path     = meta_path
        self.embedding_dim = embedding_dim
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict]            = []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_or_create_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        return self._index

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Vector dimension mismatch. Expected {self.embedding_dim}, "
                f"got {vectors.shape[1]}.",
                details={"expected": self.embedding_dim, "received": vectors.shape[1]},
            )
        return vectors

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        """
        Adds pre-computed embeddings + their source chunks to the index.

        Args:
            chunks:  TextChunk list — positionally aligned with vectors.
            vectors: np.ndarray of shape (N, 1024).

        Raises:
            VectorStoreError: On dimension mismatch or FAISS failure.
        """
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                "chunks and vectors count mismatch.",
                details={"chunks": len(chunks), "vectors": vectors.shape[0]},
            )

        vectors = self._validate_vectors(vectors)
        index   = self._get_or_create_index()

        try:
            index.add(vectors)
        except Exception as exc:
            raise VectorStoreError(
                "FAISS index.add() failed.",
                details={"error": str(exc), "vector_shape": vectors.shape},
            )

        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "text":     chunk.text,
                **chunk.metadata,
            })

        logger.info(f"FAISS: added {len(chunks)} vectors. Total: {index.ntotal}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k:            int   = TOP_K,
        score_threshold:  float = SCORE_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """
        Searches the FAISS index for the top-k nearest vectors.

        Args:
            query_vector:    shape (1024,) — output of embed_query().
            top_k:           Number of results to return.
            score_threshold: Minimum cosine similarity score to include.

        Returns:
            List of RetrievedChunk sorted by descending score.

        Raises:
            IndexNotFoundError: If the index is empty.
            VectorStoreError:   On FAISS search failure.
        """
        if self._index is None or self._index.ntotal == 0:
            raise IndexNotFoundError(
                "FAISS index is empty. Run ingestion before querying.",
                details={"index_path": self.index_path},
            )

        query_vector = self._validate_vectors(query_vector)

        try:
            scores, indices = self._index.search(query_vector, top_k)
        except Exception as exc:
            raise VectorStoreError(
                "FAISS search failed.",
                details={"error": str(exc)},
            )

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < score_threshold:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id = meta.get("chunk_id", str(idx)),
                    text     = meta.get("text", ""),
                    score    = float(score),
                    metadata = {k: v for k, v in meta.items()
                                if k not in ("chunk_id", "text")},
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def save(self) -> None:
        """Persists the FAISS index and metadata to disk."""
        if self._index is None or self._index.ntotal == 0:
            raise VectorStoreError(
                "Cannot save an empty FAISS index.",
                details={"index_path": self.index_path},
            )
        try:
            faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"FAISS index saved → {self.index_path}")
            logger.info(f"Metadata saved    → {self.meta_path}")
        except Exception as exc:
            raise VectorStoreError(
                "Failed to save FAISS index to disk.",
                details={"error": str(exc), "index_path": self.index_path},
            )

    def load(self) -> None:
        """Loads a persisted FAISS index and metadata from disk."""
        if not Path(self.index_path).exists():
            raise IndexNotFoundError(
                f"FAISS index file not found: {self.index_path}",
                details={"index_path": self.index_path},
            )
        if not Path(self.meta_path).exists():
            raise IndexNotFoundError(
                f"FAISS metadata file not found: {self.meta_path}",
                details={"meta_path": self.meta_path},
            )
        try:
            self._index    = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            logger.info(
                f"FAISS index loaded ← {self.index_path} "
                f"({self._index.ntotal} vectors)"
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to load FAISS index from disk.",
                details={"error": str(exc), "index_path": self.index_path},
            )

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0


# ══════════════════════════════════════════════════════════════════════════════
# Qdrant Backend
# ══════════════════════════════════════════════════════════════════════════════

class QdrantVectorStore:
    """
    Qdrant vector store — production alternative to FAISS.
    Supports filtering by metadata fields (source, page_number, script)
    which FAISS cannot do natively.

    Requires a running Qdrant instance:
        docker run -p 6333:6333 qdrant/qdrant
    """

    def __init__(
        self,
        host:           str = QDRANT_HOST,
        port:           int = QDRANT_PORT,
        collection:     str = QDRANT_COLLECTION,
        embedding_dim:  int = EMBEDDING_DIM,
    ):
        self.collection    = collection
        self.embedding_dim = embedding_dim
        try:
            self._client = QdrantClient(host=host, port=port, timeout=10)
        except Exception as exc:
            raise VectorStoreError(
                f"Cannot connect to Qdrant at {host}:{port}.",
                details={"host": host, "port": port, "error": str(exc)},
            )

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            self._client.create_collection(
                collection_name = self.collection,
                vectors_config  = qdrant_models.VectorParams(
                    size     = self.embedding_dim,
                    distance = qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"Qdrant collection '{self.collection}' created.")

    def add(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        """
        Upserts chunk embeddings into Qdrant.

        Args:
            chunks:  TextChunk list aligned with vectors.
            vectors: np.ndarray shape (N, 1024).

        Raises:
            VectorStoreError: On Qdrant upsert failure.
        """
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                "chunks and vectors count mismatch.",
                details={"chunks": len(chunks), "vectors": vectors.shape[0]},
            )

        self._ensure_collection()

        points: list[qdrant_models.PointStruct] = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                qdrant_models.PointStruct(
                    id      = abs(hash(chunk.chunk_id)) % (2**63),
                    vector  = vector.tolist(),
                    payload = {
                        "chunk_id": chunk.chunk_id,
                        "text":     chunk.text,
                        **chunk.metadata,
                    },
                )
            )

        try:
            self._client.upsert(
                collection_name = self.collection,
                points          = points,
                wait            = True,
            )
            logger.info(
                f"Qdrant: upserted {len(points)} points "
                f"into '{self.collection}'."
            )
        except Exception as exc:
            raise VectorStoreError(
                "Qdrant upsert failed.",
                details={"error": str(exc), "collection": self.collection},
            )

    def search(
        self,
        query_vector:    np.ndarray,
        top_k:           int   = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        filter_by:       dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Searches Qdrant with optional metadata filtering.

        Args:
            query_vector:    shape (1024,).
            top_k:           Max results.
            score_threshold: Min cosine similarity.
            filter_by:       e.g. {"source": "ramayanam.pdf", "script": "telugu"}
                             Maps to Qdrant FieldCondition filters.

        Returns:
            List of RetrievedChunk sorted by descending score.
        """
        qdrant_filter = None
        if filter_by:
            must_conditions = [
                qdrant_models.FieldCondition(
                    key   = key,
                    match = qdrant_models.MatchValue(value=value),
                )
                for key, value in filter_by.items()
            ]
            qdrant_filter = qdrant_models.Filter(must=must_conditions)

        try:
            hits = self._client.search(
                collection_name = self.collection,
                query_vector    = query_vector.tolist(),
                limit           = top_k,
                score_threshold = score_threshold,
                query_filter    = qdrant_filter,
                with_payload    = True,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Qdrant search failed.",
                details={"error": str(exc), "collection": self.collection},
            )

        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id = payload.get("chunk_id", str(hit.id)),
                    text     = payload.get("text", ""),
                    score    = float(hit.score),
                    metadata = {k: v for k, v in payload.items()
                                if k not in ("chunk_id", "text")},
                )
            )

        return results

    @property
    def total_vectors(self) -> int:
        try:
            info = self._client.get_collection(self.collection)
            return info.points_count
        except Exception:
            return 0


# ══════════════════════════════════════════════════════════════════════════════
# Unified Interface
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Backend-agnostic vector store.
    Reads VECTOR_BACKEND from config and delegates to FAISS or Qdrant.
    All downstream code (api, scripts) imports only this class.

    Usage:
        store = VectorStore()
        store.index_chunks(chunks)          # embed + add + save
        results = store.query("రామ ఎవరు")  # normalize → embed → search
    """

    def __init__(self, backend: str = VECTOR_BACKEND):
        self.backend = backend.lower()
        if self.backend == "faiss":
            self._store = FAISSVectorStore()
        elif self.backend == "qdrant":
            self._store = QdrantVectorStore()
        else:
            raise VectorStoreError(
                f"Unknown VECTOR_BACKEND '{backend}'. Choose 'faiss' or 'qdrant'.",
                details={"backend": backend},
            )
        logger.info(f"VectorStore initialized with backend: '{self.backend}'")

    def index_chunks(
        self,
        chunks:        list[TextChunk],
        batch_size:    int  = 32,
        show_progress: bool = True,
        persist:       bool = True,
    ) -> None:
        """
        Full pipeline: embed chunks → add to store → persist (FAISS only).

        Args:
            chunks:        Output of chunk_document().
            batch_size:    Embedding batch size.
            show_progress: tqdm progress bar during embedding.
            persist:       If True and backend is FAISS, saves index to disk.
        """
        if not chunks:
            raise VectorStoreError(
                "index_chunks() received an empty chunk list.",
                details={"chunks_count": 0},
            )

        vectors = embed_chunks(
            chunks,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        self._store.add(chunks, vectors)

        if persist and self.backend == "faiss":
            self._store.save()

    def load(self) -> None:
        """
        Loads a persisted index from disk (FAISS only).
        Qdrant is stateful — no load step required.
        """
        if self.backend == "faiss":
            self._store.load()
        else:
            logger.info("Qdrant backend is stateful — no load step required.")

    def query(
        self,
        query_text:      str,
        top_k:           int   = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        filter_by:       dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        End-to-end query pipeline:
            raw query string → embed → search → RetrievedChunk list

        Args:
            query_text:      Raw query (Telugu, Romanized, or English).
                             Normalization happens inside embed_query() via
                             normalizer.normalize() applied upstream in the
                             API layer before this call.
            top_k:           Max results.
            score_threshold: Min cosine similarity.
            filter_by:       Metadata filters (Qdrant only; ignored for FAISS).

        Returns:
            List of RetrievedChunk sorted by descending score.
        """
        query_vector = embed_query(query_text)

        if self.backend == "faiss":
            return self._store.search(
                query_vector    = query_vector,
                top_k           = top_k,
                score_threshold = score_threshold,
            )
        else:
            return self._store.search(
                query_vector    = query_vector,
                top_k           = top_k,
                score_threshold = score_threshold,
                filter_by       = filter_by,
            )

    @property
    def total_vectors(self) -> int:
        return self._store.total_vectors