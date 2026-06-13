class VaaniverseBaseError(Exception):
    """Root exception for all VāṇiVerse errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


# ── Ingestion ──────────────────────────────────────────────────────────────────
class DocumentParseError(VaaniverseBaseError):
    """Raised when PyMuPDF fails to open or extract content from a document."""

class UnsupportedFileTypeError(VaaniverseBaseError):
    """Raised when an uploaded file type is not supported by the parser."""

class ChunkingError(VaaniverseBaseError):
    """Raised when the chunker fails to segment a document."""

class NormalizationError(VaaniverseBaseError):
    """Raised when Unicode normalization or transliteration fails."""


# ── Retrieval ──────────────────────────────────────────────────────────────────
class EmbeddingError(VaaniverseBaseError):
    """Raised when BGE-M3 fails to generate embeddings."""

class VectorStoreError(VaaniverseBaseError):
    """Raised on FAISS or Qdrant read/write failures."""

class IndexNotFoundError(VaaniverseBaseError):
    """Raised when a query is issued before any index has been built."""


# ── Reasoning ─────────────────────────────────────────────────────────────────
class LLMInferenceError(VaaniverseBaseError):
    """Raised when the Groq or Ollama backend returns an error or times out."""

class PromptConstructionError(VaaniverseBaseError):
    """Raised when retrieved context cannot be assembled into a valid prompt."""

class NoContextRetrievedError(VaaniverseBaseError):
    """Raised when retrieval returns zero results above the score threshold."""


# ── API ────────────────────────────────────────────────────────────────────────
class InvalidRequestError(VaaniverseBaseError):
    """Raised when an API request fails input validation."""