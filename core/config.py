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

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "groq")                   # "groq" | "ollama"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

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

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"