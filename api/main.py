import hashlib
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import (
    API_HOST,
    API_PORT,
    API_RELOAD,
    CITATION_VERIFY_ENABLED,
    DATA_DIR,
    GROQ_CONFIGURED,
    GROQ_MODEL,
    HYBRID_ENABLED,
    LLM_PROVIDER,
    MAX_PDF_PAGES,
    MAX_UPLOAD_MB,
    RERANK_ENABLED,
    RERANK_THRESHOLD,
    SCORE_THRESHOLD,
    TOP_K,
    threshold_for_script,
)
from core.exceptions import (
    IndexNotFoundError,
    InvalidRequestError,
    LLMInferenceError,
    NoContextRetrievedError,
    PromptConstructionError,
    RerankError,
    VectorStoreError,
)
from core.security import (
    auth_configured,
    global_auth_middleware,
    ingest_rate_limit_guard,
    rate_limit_guard,
    recent_requests,
    require_api_key,
    telemetry_middleware,
)
from ingestion.chunker import chunk_document
from ingestion.normalizer import detect_script, normalize
from ingestion.parser import parse_document
from reasoning.generator import (
    SUPPORTED_PROVIDERS,
    Generator,
    list_groq_models,
    ollama_reachable,
)
from reasoning.prompter import build_no_result_prompt, build_prompt
from reasoning.verify import verify_answer
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
    expose_headers    = ["X-Request-ID"],
)

# Order matters: Starlette makes the *last* added middleware the outermost one,
# so telemetry is registered first and auth second in order that blanket auth
# runs before telemetry — an unauthorised request is rejected before it is
# counted as usage.
app.middleware("http")(telemetry_middleware)
app.middleware("http")(global_auth_middleware)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

# `score_threshold` applies to whichever score ranks the results, so its default
# has to move with that scale: cosine similarity when reranking is off, the
# reranker's sigmoid score when it is on. Both are 0..1, but the same number
# means different things on each scale.
_DEFAULT_SCORE_THRESHOLD = RERANK_THRESHOLD if RERANK_ENABLED else SCORE_THRESHOLD


class HistoryTurn(BaseModel):
    role:    str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    query:           str   = Field(..., min_length=1, max_length=1000,
                                   description="Query in Telugu, Romanized Telugu, or English.")
    top_k:           int   = Field(default=TOP_K, ge=1, le=20)
    # None means "use the calibrated default for this query's script". An
    # explicit value always wins, so clients that pin a threshold are unaffected.
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filter_source:   str | None = Field(default=None,
                                        description="Filter results to a specific source PDF filename.")
    provider:        str | None = Field(default=None,
                                        description="LLM provider override: 'ollama' or 'groq'. "
                                                    "Omit to use the server default (LLM_PROVIDER).")
    history:         list[HistoryTurn] = Field(default_factory=list,
                                        max_length=20,
                                        description="Prior turns for follow-up questions. "
                                                    "Used as chat context only — never citable.")


class SourceReference(BaseModel):
    chunk_id:        str
    source:          str
    page_number:     int | str
    score:           float
    retrieval_score: float = 0.0
    script:          str
    # The cited passage, so a client can show the evidence rather than only a
    # filename and page number.
    text:            str = ""


class CitationVerification(BaseModel):
    passed:             bool
    total_claims:       int
    uncited_claims:     int
    invalid_citations:  list[int]
    comparable_claims:  int
    unsupported_claims: int
    min_support:        float
    flagged:            list[dict] = []


class QueryResponse(BaseModel):
    answer:        str
    query:         str
    query_script:  str
    model:         str
    provider:      str
    backend:       str
    sources_used:  list[SourceReference]
    total_chunks_retrieved: int
    reranked:      bool = False
    hybrid:        bool = False
    score_threshold_used: float
    verification:  CitationVerification | None = None


class IngestResponse(BaseModel):
    status:        str
    source:        str
    total_pages:   int
    total_chunks:  int
    total_vectors: int
    file_hash:     str = ""
    replaced:      bool = False
    replaced_chunks: int = 0


class SearchResult(BaseModel):
    chunk_id:        str
    text:            str
    score:           float
    retrieval_score: float = 0.0
    source:          str
    page_number:     int | str
    script:          str


class SearchResponse(BaseModel):
    query:        str
    query_script: str
    results:      list[SearchResult]
    total_results: int
    reranked:     bool = False


class FeedbackRequest(BaseModel):
    verdict:  str  = Field(..., pattern="^(up|down)$",
                           description="'up' or 'down'.")
    query:    str  = Field(..., max_length=2000)
    answer:   str  = Field(..., max_length=8000)
    comment:  str | None = Field(default=None, max_length=1000)
    provider: str | None = Field(default=None, max_length=50)
    model:    str | None = Field(default=None, max_length=100)
    sources:  list[SourceReference] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status:        str
    total_vectors: int
    vector_backend: str
    llm_backend:   str
    llm_default:   str
    groq_configured: bool
    # Ollama is only a usable provider where an Ollama server exists. Clients use
    # this to avoid offering (or defaulting to) a provider that cannot answer.
    ollama_reachable: bool = True
    # Whether destructive routes require an X-API-Key.
    auth_configured: bool = False
    hybrid_enabled:  bool = False
    rerank_enabled:  bool = False
    groq_model:    str = ""
    # True/False/None — None means "could not check" (no key / call failed), which
    # is different from "the id is retired".
    groq_model_available: bool | None = None
    sources:       list[str] = []


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


# The frontend polls /health; probing Ollama on every poll would be wasteful, and
# a provider does not appear or vanish mid-session. Cache the answer briefly.
_OLLAMA_PROBE_TTL_S = 30.0
_ollama_probe_cache: tuple[float, bool] | None = None


def _ollama_reachable_cached() -> bool:
    """Whether an Ollama server answers, re-probed at most once per TTL."""
    global _ollama_probe_cache

    now = time.monotonic()
    if _ollama_probe_cache and (now - _ollama_probe_cache[0]) < _OLLAMA_PROBE_TTL_S:
        return _ollama_probe_cache[1]

    reachable = ollama_reachable()
    _ollama_probe_cache = (now, reachable)
    return reachable


def _normalize_query(query: str) -> tuple[str, str]:
    """
    Normalize a raw query and detect its script.

    Script is detected on the RAW query before normalization, then used
    to decide whether transliteration should run at all. This prevents
    genuine English queries ("who am i", "what is dharma") from being
    routed through the Romanized-Telugu transliterator, which would
    otherwise mangle them into garbage before embedding.

    Returns:
        (normalized_query, script) where script ∈
        {"telugu", "roman", "mixed", "unknown"}.
    """
    script = detect_script(query)
    normalized = normalize(query, transliterate=True)
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
    default_provider = _generator.default_provider if _generator else LLM_PROVIDER
    try:
        sources = store.sources()
    except Exception:  # a store that cannot enumerate must not break liveness
        sources = []
    return HealthResponse(
        status          = "ok",
        total_vectors   = store.total_vectors,
        vector_backend  = store.backend,
        llm_backend     = default_provider,
        llm_default     = default_provider,
        groq_configured = GROQ_CONFIGURED,
        ollama_reachable = _ollama_reachable_cached(),
        auth_configured = auth_configured(),
        hybrid_enabled  = HYBRID_ENABLED,
        rerank_enabled  = RERANK_ENABLED,
        groq_model      = GROQ_MODEL,
        groq_model_available = _groq_model_available_cached(),
        sources         = sources,
    )


# The live Groq catalogue changes slowly and the call is a network round trip,
# so cache the verdict for the same reason the Ollama probe is cached.
_GROQ_MODEL_TTL_S = 300.0
_groq_model_cache: tuple[float, bool | None] | None = None


def _groq_model_available_cached() -> bool | None:
    global _groq_model_cache
    if not GROQ_CONFIGURED:
        return None
    now = time.monotonic()
    if _groq_model_cache and (now - _groq_model_cache[0]) < _GROQ_MODEL_TTL_S:
        return _groq_model_cache[1]
    try:
        from reasoning.generator import groq_model_available
        verdict = groq_model_available(GROQ_MODEL)
    except Exception:
        verdict = None
    _groq_model_cache = (now, verdict)
    return verdict


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"],
          dependencies=[Depends(require_api_key), Depends(ingest_rate_limit_guard)])
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Ingests a PDF document into the vector index, replacing any earlier copy.

    Pipeline:
        upload → hash → drop old vectors for this source → parse → chunk → embed
               → index → persist

    Accepts: PDF files only, up to MAX_UPLOAD_MB and MAX_PDF_PAGES. Re-ingesting
    the same filename is idempotent: its previous vectors are removed first, so
    the index does not silently accumulate duplicates.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code = 422,
            detail      = "Only PDF files are supported. Upload a .pdf file.",
        )

    # Reject an oversized upload on its declared size before reading it whole.
    declared = getattr(file, "size", None)
    if declared is not None and declared > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code = 413,
            detail      = f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    store       = _get_store()
    destination = DATA_DIR / file.filename

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(
                status_code = 413,
                detail      = f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
            )
        with open(destination, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file → {destination}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to save uploaded file: {exc}",
        )

    file_hash = hashlib.sha256(content).hexdigest()

    try:
        parsed_doc = parse_document(destination, extract_images=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Document parsing failed: {exc}")

    if parsed_doc.total_pages > MAX_PDF_PAGES:
        raise HTTPException(
            status_code = 413,
            detail      = f"Document has {parsed_doc.total_pages} pages, over the "
                          f"{MAX_PDF_PAGES}-page limit.",
        )

    try:
        chunks = chunk_document(parsed_doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}")

    if not chunks:
        # Distinguish "no engine installed" from "genuinely empty" — the first is
        # an operator problem with a fix, the second is a bad upload.
        ocr_unavailable = parsed_doc.doc_metadata.get("ocr_unavailable")
        if ocr_unavailable:
            raise HTTPException(
                status_code = 422,
                detail      = "Document produced zero chunks and no OCR engine is "
                              f"available ({ocr_unavailable}). Install Tesseract "
                              "(or EasyOCR) with Telugu data, or upload a text PDF.",
            )
        raise HTTPException(
            status_code = 422,
            detail      = "Document produced zero chunks. It may be empty or image-only.",
        )

    # Idempotent ingest: drop this source's old vectors before adding new ones.
    replaced_chunks = store.remove_source(file.filename, persist=False)

    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["source"]    = file.filename

    try:
        store.index_chunks(chunks, show_progress=False, persist=True)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc.message}")

    return IngestResponse(
        status          = "success",
        source          = file.filename,
        total_pages     = parsed_doc.total_pages,
        total_chunks    = len(chunks),
        total_vectors   = store.total_vectors,
        file_hash       = file_hash,
        replaced        = replaced_chunks > 0,
        replaced_chunks = replaced_chunks,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"],
          dependencies=[Depends(rate_limit_guard)])
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

    # Resolve the provider for this request: explicit body value, else env default.
    provider_name = (request.provider or LLM_PROVIDER).lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid provider '{request.provider}'. "
                          f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}.",
        )
    if provider_name == "groq" and not GROQ_CONFIGURED:
        raise HTTPException(
            status_code = 503,
            detail      = "Provider 'groq' selected but GROQ_API_KEY is not "
                          "configured on the server.",
        )

    if store.total_vectors == 0:
        raise HTTPException(
            status_code = 503,
            detail      = "Index is empty. Ingest documents via POST /ingest first.",
        )

    normalized_query, script = _normalize_query(request.query)

    # An explicit client threshold wins; otherwise apply the calibrated default
    # for this query's script (which falls back to SCORE_THRESHOLD when no
    # per-script value has been calibrated yet).
    threshold = request.score_threshold
    if threshold is None:
        threshold = threshold_for_script(script)

    filter_by = {"source": request.filter_source} if request.filter_source else None
    history   = [turn.model_dump() for turn in request.history]

    try:
        chunks = store.query(
            query_text      = normalized_query,
            top_k           = request.top_k,
            score_threshold = threshold,
            filter_by       = filter_by,
        )
    except IndexNotFoundError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=exc.message)
    except RerankError as exc:
        raise HTTPException(status_code=500, detail=exc.message)

    if not chunks:
        prompt   = build_no_result_prompt(normalized_query, script, history=history)
        response = generator.generate(prompt, provider=provider_name)
        return QueryResponse(
            answer                 = response.answer,
            query                  = request.query,
            query_script           = script,
            model                  = response.model,
            provider               = response.provider,
            backend                = response.backend,
            sources_used           = [],
            total_chunks_retrieved = 0,
            reranked               = store.reranking_active,
            hybrid                 = HYBRID_ENABLED and not store.reranking_active,
            score_threshold_used   = threshold,
        )

    try:
        prompt = build_prompt(normalized_query, chunks, script, history=history)
    except (NoContextRetrievedError, PromptConstructionError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        response = generator.generate(prompt, provider=provider_name)
    except LLMInferenceError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    sources = [
        SourceReference(
            chunk_id        = s["chunk_id"],
            source          = s["source"],
            page_number     = s["page_number"],
            score           = s["score"],
            retrieval_score = s.get("retrieval_score", s["score"]),
            script          = s["script"],
            text            = s.get("text", ""),
        )
        for s in response.sources_used
    ]

    verification = None
    if CITATION_VERIFY_ENABLED:
        result = verify_answer(response.answer, [c.text for c in chunks])
        verification = CitationVerification(**result.to_dict())

    return QueryResponse(
        answer                 = response.answer,
        query                  = request.query,
        query_script           = script,
        model                  = response.model,
        provider               = response.provider,
        backend                = response.backend,
        sources_used           = sources,
        total_chunks_retrieved = len(chunks),
        reranked               = store.reranking_active,
        hybrid                 = HYBRID_ENABLED and not store.reranking_active,
        score_threshold_used   = threshold,
        verification           = verification,
    )


@app.get("/search", response_model=SearchResponse, tags=["Retrieval"],
         dependencies=[Depends(rate_limit_guard)])
async def search(
    q:               str   = Query(..., min_length=1, max_length=1000),
    top_k:           int   = Query(default=TOP_K, ge=1, le=20),
    score_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    filter_source:   str | None = Query(default=None),
) -> SearchResponse:
    """
    Pure semantic retrieval — returns matching chunks without LLM generation.
    Useful for debugging retrieval quality before adding the reasoning layer.

    `score` is the score that ranked the result; `retrieval_score` is always the
    dense-retrieval cosine similarity. They differ when `reranked` is true, in
    which case `score` is the cross-encoder's and `score_threshold` applies to it.
    """
    store = _get_store()

    if store.total_vectors == 0:
        raise HTTPException(
            status_code = 503,
            detail      = "Index is empty. Ingest documents first.",
        )

    normalized_query, script = _normalize_query(q)
    if score_threshold is None:
        score_threshold = threshold_for_script(script)
    filter_by = {"source": filter_source} if filter_source else None

    try:
        chunks = store.query(
            query_text      = normalized_query,
            top_k           = top_k,
            score_threshold = score_threshold,
            filter_by       = filter_by,
        )
    except (IndexNotFoundError, VectorStoreError, RerankError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        SearchResult(
            chunk_id        = chunk.chunk_id,
            text            = chunk.text,
            score           = chunk.score,
            retrieval_score = chunk.retrieval_score,
            source          = chunk.metadata.get("source", "unknown"),
            page_number     = chunk.metadata.get("page_number", "?"),
            script          = chunk.metadata.get("script", "unknown"),
        )
        for chunk in chunks
    ]

    return SearchResponse(
        query         = q,
        query_script  = script,
        results       = results,
        total_results = len(results),
        reranked      = store.reranking_active,
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
        "sources":        store.sources(),
    }


@app.get("/index/sources", tags=["System"])
async def index_sources() -> dict[str, Any]:
    """Distinct source filenames in the index — used by the UI's source filter."""
    store = _get_store()
    return {"sources": store.sources(), "total_vectors": store.total_vectors}


@app.delete("/index/reset", tags=["System"], dependencies=[Depends(require_api_key)])
async def reset_index() -> dict[str, str]:
    """
    Resets the index in memory *and* on disk.

    FAISS: deletes vaaniverse.faiss / vaaniverse_meta.pkl and clears memory.
    Qdrant: drops and recreates the collection.

    Previously this cleared memory only, so a restart resurrected the index the
    caller believed they had deleted. Requires X-API-Key when one is configured.
    """
    global _vector_store
    _vector_store = VectorStore()
    _vector_store.reset()
    logger.warning("Vector index reset by API call (memory + disk).")
    return {"status": "reset", "message": "Index cleared on disk and in memory. Re-ingest to rebuild."}


@app.get("/admin/requests", tags=["System"], dependencies=[Depends(require_api_key)])
async def admin_requests(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    """
    Recent request telemetry (in-process ring buffer).

    Enough to answer "what did that failing request look like?" on a small
    deployment; not a replacement for real observability.
    """
    return {"requests": recent_requests(limit), "auth_configured": auth_configured()}


# Feedback is appended, not scored: a thumbs-down with no stored answer is not
# actionable, so the client sends the question and answer it is judging.
_FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"


@app.post("/feedback", tags=["Query"], dependencies=[Depends(rate_limit_guard)])
async def submit_feedback(payload: FeedbackRequest) -> dict[str, str]:
    """
    Record a thumbs-up/down on an answer.

    Deliberately minimal: one JSON line per verdict, so a bad answer can be found
    later instead of living only in someone's memory. Nothing here trains or
    rewrites anything.
    """
    record = {
        "ts":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict":  payload.verdict,
        "query":    payload.query[:2000],
        "answer":   payload.answer[:4000],
        "comment":  (payload.comment or "")[:1000],
        "provider": payload.provider or "",
        "model":    payload.model or "",
        "sources":  [s.model_dump() for s in payload.sources][:20],
    }
    try:
        with open(_FEEDBACK_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # a full or read-only disk must not 500 the UI
        logger.warning(f"Could not persist feedback: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not record feedback: {exc}")
    return {"status": "recorded"}


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host   = API_HOST,
        port   = API_PORT,
        reload = API_RELOAD,
    )