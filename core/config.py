import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
# Replace these two lines in core/config.py
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
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"