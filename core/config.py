import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "vectors"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")          # "cuda" if GPU available
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── Vector Store ───────────────────────────────────────────────────────────────
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "faiss")            # "faiss" | "qdrant"
FAISS_INDEX_PATH: str = str(INDEX_DIR / "vaaniverse.faiss")
FAISS_META_PATH: str = str(INDEX_DIR / "vaaniverse_meta.pkl")

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "vaaniverse")

# ── LLM provider switch ────────────────────────────────────────────────────────
# LLM_PROVIDER is the source of truth ("ollama" | "groq"); Ollama is the default.
# LLM_BACKEND is retained as a backward-compatible alias for older deployments.
LLM_PROVIDER: str = (
    os.getenv("LLM_PROVIDER")
    or os.getenv("LLM_BACKEND")
    or "ollama"
).lower()
LLM_BACKEND: str = LLM_PROVIDER                                       # legacy alias

# ── Groq (opt-in; key comes from the environment only) ─────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Groq retires model ids on its own schedule: `llama-3.3-70b-versatile` was the
# default here until it began returning 404 model_not_found. This replacement was
# picked on measurement — fastest of the models the key can see (~190 ms vs
# ~1000 ms), and unlike the gpt-oss models it honours the literal [SOURCE n]
# citation markers the prompt asks for instead of translating them. If Groq
# retires this one too, list what is currently available:
#   https://console.groq.com/docs/models
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_CONFIGURED: bool = bool(GROQ_API_KEY)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))                  # tokens
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")         # fallback; Telugu uses rule-based

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.45"))

# Per-script score thresholds.
#
# A single 0.45 cutoff was calibrated against a corpus that is ~85% native Telugu
# text, so it is really a Telugu cutoff applied to every script. Native-script
# queries embed closer to the (native-script) passages than Roman or English
# queries do, which is measurable as a lower top-1 score at equal relevance —
# and that is how the 0.45 gate ends up rejecting English and Roman queries that
# retrieval actually ranked correctly.
#
# Values below are measured, not guessed: `python eval/run_eval.py --offline` on
# the 33-chunk sample book, choosing the cutoff that maximizes in-corpus
# passage subject to near-zero out-of-domain passage (see docs/EVALUATION.md).
# Measured on the 53-question gold set:
#   telugu 0.41 → 23/23 Telugu gold queries pass, 1/3 OOD probes do
#   roman  0.46 → 12/14 Roman+English gold queries pass, 0/13 OOD probes do
# A single 0.45 passed 81.1% in-corpus; per-script raises that to 94.6% for one
# extra near-miss (the Telugu "who is the author" probe, which cannot be answered
# from the corpus). Re-run scripts/calibrate_threshold.py after any change to the
# corpus or embedder.
SCORE_THRESHOLD_TELUGU: float = float(os.getenv("SCORE_THRESHOLD_TELUGU", "0.41"))
SCORE_THRESHOLD_ROMAN: float = float(os.getenv("SCORE_THRESHOLD_ROMAN", "0.46"))
SCORE_THRESHOLD_MIXED: float = float(os.getenv("SCORE_THRESHOLD_MIXED", "0.45"))


def threshold_for_script(script: str | None) -> float:
    """The score floor that applies to a query of the given script."""
    return {
        "telugu": SCORE_THRESHOLD_TELUGU,
        "roman":  SCORE_THRESHOLD_ROMAN,
        "mixed":  SCORE_THRESHOLD_MIXED,
    }.get(script or "", SCORE_THRESHOLD)


# ── Hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion) ─────────
# Dense-only retrieval is weak exactly where heritage corpora are hard: proper
# names, transliterated titles and rare terms ("emiti", "ఏమిటి", "kosha") that
# BGE-M3 maps to a fuzzy neighbourhood. BM25 nails those literal matches and is
# blind to paraphrase, so the two are complementary. RRF fuses ranks rather than
# scores, which avoids having to calibrate two incomparable scales against each
# other.
HYBRID_ENABLED: bool = os.getenv("HYBRID_ENABLED", "false").lower() == "true"
# Weight of the dense ranking in the fusion. 1.0 = dense-only, 0.0 = BM25-only.
# Applied as weighted RRF: score = w / (k + rank_dense) + (1-w) / (k + rank_bm25).
HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
HYBRID_CANDIDATES: int = int(os.getenv("HYBRID_CANDIDATES", "30"))
RRF_K: int = int(os.getenv("RRF_K", "60"))

# ── Reranking (second-stage cross-encoder) ─────────────────────────────────────
# Off by default, for two measured reasons rather than caution:
#   * the reranker is a second ~2.3 GB model, and Streamlit Community Cloud's
#     default RAM limit is too small to hold it alongside BGE-M3;
#   * on this project's 4 GB GPU it would contend with BGE-M3 + the LLM.
# RERANK_DEVICE therefore defaults to CPU even when the embedder is on GPU:
# a cross-encoder over ~20 candidates is cheap, and OOM would break retrieval.
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE: str = os.getenv("RERANK_DEVICE", "cpu")
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "20"))  # retrieved before reranking
RERANK_THRESHOLD: float = float(os.getenv("RERANK_THRESHOLD", "0.5"))  # swept, see docs/EVALUATION.md

# ── OCR (scanned / image-only pages) ───────────────────────────────────────────
# Whole-pipeline OCR is off the *embedding path*: it only ever runs for a page
# whose text layer is empty or vestigial, so the existing text PDF is unaffected.
OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto")        # auto | tesseract | easyocr
OCR_LANG: str = os.getenv("OCR_LANG", "te,en")           # ISO 639-1; engine codes mapped internally
OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))          # page rasterization resolution
OCR_MIN_CHARS: int = int(os.getenv("OCR_MIN_CHARS", "16"))  # under this, a page counts as image-only
# Pre-processing the raster before recognition. Old print is noisy and skewed;
# grayscale + autocontrast + Otsu binarization is cheap and measurably helps
# Tesseract on degraded scans. OCR_PSM 6 ("assume a uniform block of text") fits
# a body page better than the 3 default, which looks for automatic layout.
OCR_PREPROCESS: bool = os.getenv("OCR_PREPROCESS", "true").lower() == "true"
OCR_PSM: int = int(os.getenv("OCR_PSM", "6"))

# ── Conversation memory ────────────────────────────────────────────────────────
# Each question used to be answered in isolation, so "and what did that mean?"
# had nothing to resolve against. History is supplied by the client (the server
# is stateless), truncated to the last N turns, and fed to the model as real
# chat turns — never as sources, so it cannot be cited.
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "4"))
MAX_HISTORY_CHARS: int = int(os.getenv("MAX_HISTORY_CHARS", "2000"))

# ── Answer verification (claim ↔ citation) ─────────────────────────────────────
# Every claim is supposed to carry a [SOURCE N] whose passage supports it. This
# checks that: each cited claim is compared against its cited passage, and a
# claim whose overlap falls under the floor is flagged. Lexical overlap keeps it
# offline and deterministic; it is a smoke alarm, not a judge.
CITATION_VERIFY_ENABLED: bool = os.getenv("CITATION_VERIFY_ENABLED", "true").lower() == "true"
CITATION_SUPPORT_MIN: float = float(os.getenv("CITATION_SUPPORT_MIN", "0.12"))
# Citations that point at a source number never supplied are always errors.
CITATION_MAX_INVALID: int = int(os.getenv("CITATION_MAX_INVALID", "0"))

# ── API security & limits ──────────────────────────────────────────────────────
# /ingest and DELETE /index/reset are destructive and were open to anyone who
# could reach the port. When VANIVERSE_API_KEY is set, those two endpoints (and
# the admin views) require it via the `X-API-Key` header. Read-only endpoints
# (/health, /search, /query) stay open unless REQUIRE_AUTH_FOR_ALL is set.
VANIVERSE_API_KEY: str = os.getenv("VANIVERSE_API_KEY", "")
REQUIRE_AUTH_FOR_ALL: bool = os.getenv("REQUIRE_AUTH_FOR_ALL", "false").lower() == "true"

# Rate limiting (in-process token bucket). Cheap insurance for a public demo;
# not a substitute for a real gateway in front of a multi-process deployment.
RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
INGEST_LIMIT_PER_HOUR: int = int(os.getenv("INGEST_LIMIT_PER_HOUR", "10"))

# Uploads: reject oversized files before reading them into memory, and cap the
# number of pages so a huge PDF cannot exhaust a small container.
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "2000"))

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
# Off by default: uvicorn's reloader spawns a supervisor + a child, doubles model
# memory and is wrong inside a container. Opt in locally with API_RELOAD=true.
API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"