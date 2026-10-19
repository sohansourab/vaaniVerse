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
    HYBRID_ALPHA,
    HYBRID_CANDIDATES,
    HYBRID_ENABLED,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RRF_K,
    SCORE_THRESHOLD,
    TOP_K,
    VECTOR_BACKEND,
)
from core.exceptions import IndexNotFoundError, VectorStoreError
from ingestion.chunker import TextChunk
from retrieval.embeddings import EMBEDDING_DIM, embed_chunks, embed_query
from retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from retrieval.reranker import available as reranker_available
from retrieval.reranker import probe as reranker_probe
from retrieval.reranker import score_pairs

logger = logging.getLogger(__name__)

# Warn once, not once per query, when reranking is configured but unusable.
_WARNED_NO_RERANKER = False


# ── Result dataclass ───────────────────────────────────────────────────────────

class RetrievedChunk:
    __slots__ = ("chunk_id", "text", "score", "retrieval_score", "metadata")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        score: float,
        metadata: dict,
        retrieval_score: float | None = None,
    ):
        self.chunk_id = chunk_id
        self.text     = text
        # `score` is whatever ranked this chunk — cosine similarity from dense
        # retrieval, or the cross-encoder score once a reranker has run. It is
        # the value `score_threshold` applies to.
        self.score    = score
        # The original dense-retrieval similarity, kept separately so reranking
        # never hides the first stage.
        self.retrieval_score = score if retrieval_score is None else retrieval_score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(chunk_id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"retrieval_score={self.retrieval_score:.4f}, "
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
        filter_by:        dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Searches the FAISS index for the top-k nearest vectors.

        FAISS has no native metadata index, so ``filter_by`` is applied in
        Python. To avoid returning fewer than ``top_k`` after filtering, a
        wider candidate window is searched when a filter is present — the same
        over-fetch trick the Qdrant path never needs.

        Args:
            query_vector:    shape (1024,) — output of embed_query().
            top_k:           Number of results to return.
            score_threshold: Minimum cosine similarity score to include.
            filter_by:       Metadata equality filters, e.g. {"source": "book.pdf"}.

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

        # Over-fetch so post-filtering can still fill top_k. The window is
        # bounded by the corpus size, so it never over-reads.
        candidate_k = top_k
        if filter_by:
            candidate_k = min(self._index.ntotal, max(top_k * 5 + 50, top_k))

        try:
            scores, indices = self._index.search(query_vector, candidate_k)
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
            if filter_by and any(meta.get(k) != v for k, v in filter_by.items()):
                continue
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
        return results[:top_k]

    def remove_by_source(self, source: str) -> int:
        """
        Drop every vector whose ``source`` matches, and rebuild the index.

        FAISS IndexFlatIP has no delete, but it can be reconstructed, and a
        rebuild of a few thousand vectors is milliseconds. Returns the number
        of chunks removed (0 when the source was not present).
        """
        if self._index is None or self._index.ntotal == 0:
            return 0

        keep = [i for i, m in enumerate(self._metadata) if m.get("source") != source]
        removed = len(self._metadata) - len(keep)
        if removed == 0:
            return 0

        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        if keep:
            new_index.add(np.ascontiguousarray(vectors[keep], dtype=np.float32))
        self._index = new_index
        self._metadata = [self._metadata[i] for i in keep]
        logger.info(f"FAISS: removed {removed} vectors for source '{source}'.")
        return removed

    def catalog(self) -> list[dict]:
        """All stored chunk records (chunk_id, text, metadata), order-preserved."""
        return list(self._metadata)

    def sources(self) -> list[str]:
        """Distinct source filenames currently indexed."""
        return sorted({m.get("source", "unknown") for m in self._metadata})

    def reset(self) -> None:
        """
        Clear the in-memory index and remove the persisted files.

        The old behaviour cleared memory only, so a reset followed by a restart
        silently resurrected the "deleted" index — confusing at best. Deleting
        the files makes the reset mean the same thing in both states.
        """
        self._index = None
        self._metadata = []
        for path in (self.index_path, self.meta_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                logger.warning(f"Could not remove {path}: {exc}")

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

    def remove_by_source(self, source: str) -> int:
        """Delete every point whose payload ``source`` matches."""
        self._ensure_collection()
        try:
            self._client.delete(
                collection_name = self.collection,
                points_selector = qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="source",
                            match=qdrant_models.MatchValue(value=source),
                        )]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Qdrant delete-by-source failed.",
                details={"error": str(exc), "collection": self.collection, "source": source},
            )
        # Qdrant reports deletes asynchronously-ish; report what remains.
        return 0

    def catalog(self) -> list[dict]:
        """Scroll every point's payload — used to build the BM25 side of hybrid search."""
        self._ensure_collection()
        records: list[dict] = []
        offset = None
        try:
            while True:
                points, offset = self._client.scroll(
                    collection_name = self.collection,
                    limit           = 256,
                    offset          = offset,
                    with_payload    = True,
                    with_vectors    = False,
                )
                for point in points:
                    payload = point.payload or {}
                    records.append({
                        "chunk_id": payload.get("chunk_id", str(point.id)),
                        "text":     payload.get("text", ""),
                        **{k: v for k, v in payload.items() if k not in ("chunk_id", "text")},
                    })
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError(
                "Qdrant scroll failed.",
                details={"error": str(exc), "collection": self.collection},
            )
        return records

    def sources(self) -> list[str]:
        return sorted({r.get("source", "unknown") for r in self.catalog()})

    def reset(self) -> None:
        """Drop and recreate the collection, so the index is empty and usable."""
        try:
            self._client.delete_collection(self.collection)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(f"Qdrant delete_collection failed for {self.collection}: {exc}")
        self._ensure_collection()
        logger.info(f"Qdrant collection '{self.collection}' reset.")

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
        # BM25 side of hybrid search, built lazily and invalidated on writes.
        self._bm25: BM25Index | None = None
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
        self._bm25 = None   # catalog changed; rebuild on next hybrid query

        if persist and self.backend == "faiss":
            self._store.save()

    def remove_source(self, source: str, persist: bool = True) -> int:
        """
        Delete every chunk originating from ``source``.

        This is what makes re-ingesting a PDF idempotent: the ingest path removes
        the source's old vectors before adding the new ones, so the same file
        uploaded twice does not double the index.
        """
        removed = self._store.remove_by_source(source)
        self._bm25 = None
        if removed and persist and self.backend == "faiss":
            try:
                self._store.save()
            except VectorStoreError:
                # An emptied index cannot be saved; the in-memory state is still
                # correct, and the next ingest will persist.
                logger.warning("Index became empty after removal; not persisting.")
        return removed

    def sources(self) -> list[str]:
        """Distinct indexed source filenames."""
        return self._store.sources()

    def catalog(self) -> list[dict]:
        """All chunk records, for building the BM25 index or listing sources."""
        return self._store.catalog()

    def reset(self) -> None:
        """Drop the index in memory *and* on disk (FAISS) / the collection (Qdrant)."""
        self._bm25 = None
        self._store.reset()

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
            raw query text → embed → search → [rerank] → RetrievedChunk list

        With reranking enabled this becomes two-stage:

            embed → top RERANK_CANDIDATES by cosine, *no* cutoff
                  → cross-encoder scores each (query, passage) pair
                  → sort by that score, apply `score_threshold`, keep `top_k`

        The cosine cutoff is deliberately not applied to the candidate set: a
        passage the reranker would promote may sit below it, and that overlap is
        precisely the failure the reranker exists to fix.

        `score_threshold` always applies to the score that *ranks* the results —
        cosine similarity with reranking off (unchanged behaviour), the
        cross-encoder's sigmoid score with it on. Both are on a 0..1 scale, and
        `RetrievedChunk.retrieval_score` always carries the cosine value.

        Args:
            query_text:      Query text (Telugu, Romanized, or English).
                             Normalization happens upstream in the API layer,
                             not here.
            top_k:           Max results to return.
            score_threshold: Relevance floor on the ranking score.
            filter_by:       Metadata filters (Qdrant only; ignored for FAISS).

        Returns:
            List of RetrievedChunk sorted by descending score.

        Raises:
            IndexNotFoundError: If the index is empty.
            VectorStoreError:   On search failure.
            RerankError:        If reranking is enabled and scoring fails.
        """
        query_vector = embed_query(query_text)
        reranking    = self._reranking_active()

        # Hybrid and the reranker are alternative second stages, not a stack:
        # both exist to fix the same failure (dense ranking), and running BM25
        # fusion before a cross-encoder that costs ~34 s/query only slows the
        # measured path. Reranking wins when explicitly enabled.
        if HYBRID_ENABLED and not reranking:
            return self._hybrid_search(
                query_text, query_vector, top_k, score_threshold, filter_by
            )

        # Only widen the net when something is going to re-order it.
        candidate_k         = max(top_k, RERANK_CANDIDATES) if reranking else top_k
        candidate_threshold = 0.0 if reranking else score_threshold

        candidates = self._store.search(
            query_vector    = query_vector,
            top_k           = candidate_k,
            score_threshold = candidate_threshold,
            filter_by       = filter_by,
        )

        if not reranking or not candidates:
            return candidates[:top_k]

        return self._rerank(query_text, candidates, top_k, score_threshold)

    # ── Hybrid search ──────────────────────────────────────────────────────────

    def _bm25_index(self) -> BM25Index:
        """Lazily build (and cache) the BM25 index over every stored passage."""
        if self._bm25 is None:
            records = self._store.catalog()
            index = BM25Index()
            index.build(
                [r.get("chunk_id", "") for r in records],
                [r.get("text", "") for r in records],
            )
            self._bm25 = index
        return self._bm25

    def _hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int,
        score_threshold: float,
        filter_by: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """
        Dense ∪ BM25, fused by Reciprocal Rank Fusion.

        Threshold semantics are kept deliberately narrow: the gate still applies
        to the *cosine* score, so a hybrid run cannot open the abstention gate
        that dense retrieval closed. A passage that only BM25 matched is kept
        despite a low cosine — an exact term match is evidence dense retrieval
        cannot produce — while a dense-only passage below the floor is dropped
        exactly as before. Ordering is by fused rank, not by cosine, so `score`
        may not descend monotonically in hybrid mode; `retrieval_score` always
        carries the cosine.
        """
        dense = self._store.search(
            query_vector    = query_vector,
            top_k           = max(HYBRID_CANDIDATES, top_k),
            score_threshold = 0.0,
            filter_by       = filter_by,
        )

        # Hybrid must not open a gate dense retrieval closed. If nothing clears
        # the calibrated floor, the query is treated as out-of-domain and the
        # caller abstains — otherwise a single shared token ("book" in "what is
        # the price of this book?") would be enough to force an answer, which is
        # exactly the hallucination the abstention path exists to prevent.
        if not dense or dense[0].score < score_threshold:
            return []

        dense_by_id = {chunk.chunk_id: chunk for chunk in dense}

        lexical = self._bm25_index().search(query_text, HYBRID_CANDIDATES)
        lexical_ids = [chunk_id for chunk_id, _ in lexical]
        lexical_set = set(lexical_ids)

        fused = reciprocal_rank_fusion(
            [chunk.chunk_id for chunk in dense],
            lexical_ids,
            alpha=HYBRID_ALPHA,
            k=RRF_K,
        )

        # Only needed when a lexical hit was not also a dense hit.
        catalog: dict[str, dict] | None = None

        results: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            chunk = dense_by_id.get(chunk_id)
            if chunk is None:
                if catalog is None:
                    catalog = {r.get("chunk_id", ""): r for r in self._store.catalog()}
                record = catalog.get(chunk_id)
                if record is None:
                    continue
                if filter_by and any(record.get(k) != v for k, v in filter_by.items()):
                    continue
                chunk = RetrievedChunk(
                    chunk_id        = chunk_id,
                    text            = record.get("text", ""),
                    score           = 0.0,
                    metadata        = {k: v for k, v in record.items()
                                       if k not in ("chunk_id", "text")},
                    retrieval_score = 0.0,
                )
                chunk.metadata["hybrid_score"] = round(rrf_score, 6)
            else:
                # Dense hit: keep the cosine as the displayed/scored value, but
                # record the fused rank that actually ordered it.
                chunk.metadata["hybrid_score"] = round(rrf_score, 6)

            if (
                chunk.retrieval_score
                and chunk.retrieval_score < score_threshold
                and chunk_id not in lexical_set
            ):
                continue  # dense-only and below the calibrated floor

            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _reranking_active() -> bool:
        """
        True when reranking is switched on *and* actually loadable.

        A missing dependency degrades to dense-only retrieval with a single
        warning: an operator's config flag should not be able to break every
        query. A model that loads but then fails to score is a real error and is
        allowed to propagate as RerankError.
        """
        global _WARNED_NO_RERANKER

        if not RERANK_ENABLED:
            return False
        if reranker_available():
            return True

        if not _WARNED_NO_RERANKER:
            logger.warning(
                "RERANK_ENABLED is set but the reranker is unusable (%s); "
                "falling back to dense-only retrieval.",
                reranker_probe(),
            )
            _WARNED_NO_RERANKER = True
        return False

    @staticmethod
    def _rerank(
        query_text: str,
        candidates: list[RetrievedChunk],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        """Re-score candidates with the cross-encoder, then rank and threshold."""
        scores = score_pairs(query_text, [chunk.text for chunk in candidates])

        for chunk, rerank_score in zip(candidates, scores):
            chunk.retrieval_score = chunk.score    # preserve the dense score
            chunk.score           = rerank_score   # this is what ranks now

        ranked = sorted(candidates, key=lambda chunk: chunk.score, reverse=True)
        return [chunk for chunk in ranked if chunk.score >= score_threshold][:top_k]

    @property
    def total_vectors(self) -> int:
        return self._store.total_vectors

    @property
    def reranking_active(self) -> bool:
        """
        Whether this store will rerank. Mirrors the per-query decision so API
        responses can tell the client which scale `score` is on.
        """
        return self._reranking_active()