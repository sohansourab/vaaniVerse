import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import API_HOST, API_PORT, DATA_DIR, SCORE_THRESHOLD, TOP_K
from core.exceptions import (
    IndexNotFoundError,
    InvalidRequestError,
    LLMInferenceError,
    NoContextRetrievedError,
    PromptConstructionError,
    VectorStoreError,
)
from ingestion.chunker import chunk_document
from ingestion.normalizer import detect_script, normalize
from ingestion.parser import parse_document
from reasoning.generator import Generator
from reasoning.prompter import build_no_result_prompt, build_prompt
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Singletons ─────────────────────────────────────────────────────────────────
# Initialized once at startup via lifespan — not at import time.
_vector_store: VectorStore | None = None
_generator:    Generator   | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load vector index + initialize LLM generator.
    Shutdown: nothing to teardown (FAISS is in-memory, Groq/Ollama are stateless).
    """
    global _vector_store, _generator

    logger.info("VāṇiVerse API starting up...")

    _vector_store = VectorStore()
    try:
        _vector_store.load()
        logger.info(
            f"Vector index loaded. "
            f"Total vectors: {_vector_store.total_vectors}"
        )
    except IndexNotFoundError:
        logger.warning(
            "No vector index found on disk. "
            "Ingest documents via POST /ingest before querying."
        )

    _generator = Generator()
    logger.info("LLM generator initialized.")

    yield

    logger.info("VāṇiVerse API shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "VāṇiVerse API",
    description = (
        "Multimodal RAG API for Telugu linguistic and cultural heritage. "
        "Supports document ingestion, semantic search, and source-grounded Q&A."
    ),
    version  = "0.1.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:           str   = Field(..., min_length=1, max_length=1000,
                                   description="Query in Telugu, Romanized Telugu, or English.")
    top_k:           int   = Field(default=TOP_K, ge=1, le=20)
    score_threshold: float = Field(default=SCORE_THRESHOLD, ge=0.0, le=1.0)
    filter_source:   str | None = Field(default=None,
                                        description="Filter results to a specific source PDF filename.")


class SourceReference(BaseModel):
    chunk_id:    str
    source:      str
    page_number: int | str
    score:       float
    script:      str


class QueryResponse(BaseModel):
    answer:        str
    query:         str
    query_script:  str
    model:         str
    backend:       str
    sources_used:  list[SourceReference]
    total_chunks_retrieved: int


class IngestResponse(BaseModel):
    status:        str
    source:        str
    total_pages:   int
    total_chunks:  int
    total_vectors: int


class SearchResult(BaseModel):
    chunk_id:    str
    text:        str
    score:       float
    source:      str
    page_number: int | str
    script:      str


class SearchResponse(BaseModel):
    query:        str
    query_script: str
    results:      list[SearchResult]
    total_results: int


class HealthResponse(BaseModel):
    status:        str
    total_vectors: int
    vector_backend: str
    llm_backend:   str


# ══════════════════════════════════════════════════════════════════════════════
# Dependency helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_store() -> VectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")
    return _vector_store


def _get_generator() -> Generator:
    if _generator is None:
        raise HTTPException(status_code=503, detail="LLM generator not initialized.")
    return _generator


def _normalize_query(query: str) -> tuple[str, str]:
    """
    Normalizes a raw query and detects its script.
    Returns (normalized_query, script).
    script ∈ {"telugu", "roman", "mixed", "unknown"}
    """
    normalized = normalize(query, transliterate=True)
    script     = detect_script(normalized)
    return normalized, script


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """
    Returns API health status and index statistics.
    Use this to verify the service is live and an index is loaded.
    """
    store = _get_store()
    return HealthResponse(
        status         = "ok",
        total_vectors  = store.total_vectors,
        vector_backend = store.backend,
        llm_backend    = _generator.backend if _generator else "uninitialized",
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Ingests a PDF document into the vector index.

    Pipeline:
        upload → save to disk → parse → chunk → embed → index → persist

    Accepts: PDF files only.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code = 422,
            detail      = "Only PDF files are supported. Upload a .pdf file.",
        )

    store       = _get_store()
    destination = DATA_DIR / file.filename

    try:
        content = await file.read()
        with open(destination, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file → {destination}")
    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to save uploaded file: {exc}",
        )

    try:
        parsed_doc = parse_document(destination, extract_images=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Document parsing failed: {exc}")

    try:
        chunks = chunk_document(parsed_doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}")

    if not chunks:
        raise HTTPException(
            status_code = 422,
            detail      = "Document produced zero chunks. It may be empty or image-only.",
        )

    try:
        store.index_chunks(chunks, show_progress=False, persist=True)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc.message}")

    return IngestResponse(
        status        = "success",
        source        = file.filename,
        total_pages   = parsed_doc.total_pages,
        total_chunks  = len(chunks),
        total_vectors = store.total_vectors,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    End-to-end RAG query.

    Pipeline:
        raw query → normalize → embed → retrieve → build prompt → LLM → response

    Supports Telugu script, Romanized Telugu, and English queries.
    Source citations are returned alongside the answer.
    """
    store     = _get_store()
    generator = _get_generator()

    if store.total_vectors == 0:
        raise HTTPException(
            status_code = 503,
            detail      = "Index is empty. Ingest documents via POST /ingest first.",
        )

    normalized_query, script = _normalize_query(request.query)

    filter_by = {"source": request.filter_source} if request.filter_source else None

    try:
        chunks = store.query(
            query_text      = normalized_query,
            top_k           = request.top_k,
            score_threshold = request.score_threshold,
            filter_by       = filter_by,
        )
    except IndexNotFoundError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=exc.message)

    if not chunks:
        prompt   = build_no_result_prompt(normalized_query, script)
        response = generator.generate(prompt)
        return QueryResponse(
            answer                 = response.answer,
            query                  = request.query,
            query_script           = script,
            model                  = response.model,
            backend                = response.backend,
            sources_used           = [],
            total_chunks_retrieved = 0,
        )

    try:
        prompt = build_prompt(normalized_query, chunks, script)
    except (NoContextRetrievedError, PromptConstructionError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        response = generator.generate(prompt)
    except LLMInferenceError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    sources = [
        SourceReference(
            chunk_id    = s["chunk_id"],
            source      = s["source"],
            page_number = s["page_number"],
            score       = s["score"],
            script      = s["script"],
        )
        for s in response.sources_used
    ]

    return QueryResponse(
        answer                 = response.answer,
        query                  = request.query,
        query_script           = script,
        model                  = response.model,
        backend                = response.backend,
        sources_used           = sources,
        total_chunks_retrieved = len(chunks),
    )


@app.get("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(
    q:               str   = Query(..., min_length=1, max_length=1000),
    top_k:           int   = Query(default=TOP_K, ge=1, le=20),
    score_threshold: float = Query(default=SCORE_THRESHOLD, ge=0.0, le=1.0),
    filter_source:   str | None = Query(default=None),
) -> SearchResponse:
    """
    Pure semantic retrieval — returns matching chunks without LLM generation.
    Useful for debugging retrieval quality before adding the reasoning layer.
    """
    store = _get_store()

    if store.total_vectors == 0:
        raise HTTPException(
            status_code = 503,
            detail      = "Index is empty. Ingest documents first.",
        )

    normalized_query, script = _normalize_query(q)
    filter_by = {"source": filter_source} if filter_source else None

    try:
        chunks = store.query(
            query_text      = normalized_query,
            top_k           = top_k,
            score_threshold = score_threshold,
            filter_by       = filter_by,
        )
    except (IndexNotFoundError, VectorStoreError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        SearchResult(
            chunk_id    = chunk.chunk_id,
            text        = chunk.text,
            score       = chunk.score,
            source      = chunk.metadata.get("source", "unknown"),
            page_number = chunk.metadata.get("page_number", "?"),
            script      = chunk.metadata.get("script", "unknown"),
        )
        for chunk in chunks
    ]

    return SearchResponse(
        query         = q,
        query_script  = script,
        results       = results,
        total_results = len(results),
    )


@app.get("/index/stats", tags=["System"])
async def index_stats() -> dict[str, Any]:
    """
    Returns current index statistics.
    """
    store = _get_store()
    return {
        "total_vectors":  store.total_vectors,
        "vector_backend": store.backend,
        "index_ready":    store.total_vectors > 0,
    }


@app.delete("/index/reset", tags=["System"])
async def reset_index() -> dict[str, str]:
    """
    Resets the in-memory index.
    FAISS: reinitializes the store (disk files remain until next ingest overwrites).
    Qdrant: drops and recreates the collection.
    Use with caution — requires re-ingestion of all documents.
    """
    global _vector_store
    _vector_store = VectorStore()
    logger.warning("Vector index reset by API call.")
    return {"status": "reset", "message": "Index cleared. Re-ingest documents to rebuild."}


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host   = API_HOST,
        port   = API_PORT,
        reload = True,
    )