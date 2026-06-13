import re
from dataclasses import dataclass, field

import spacy
from spacy.language import Language

from core.config import CHUNK_SIZE, CHUNK_OVERLAP, SPACY_MODEL
from core.exceptions import ChunkingError
from ingestion.normalizer import detect_script
from ingestion.parser import ParsedDocument, ParsedPage

# Telugu sentence boundary markers (punctuation + Devanagari/Telugu dandas)
_TELUGU_SENTENCE_END = re.compile(r"[।॥\u0964\u0965.!?]+")

# Cached spaCy model — loaded once per process
_NLP_CACHE: dict[str, Language] = {}


def _load_spacy(model_name: str) -> Language:
    if model_name not in _NLP_CACHE:
        try:
            _NLP_CACHE[model_name] = spacy.load(model_name)
        except OSError:
            # Fallback: blank English model with sentencizer
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _NLP_CACHE[model_name] = nlp
    return _NLP_CACHE[model_name]


@dataclass
class TextChunk:
    chunk_id:   str             # "{source}_p{page}_{chunk_index}"
    text:       str             # normalized chunk text
    token_len:  int             # approximate token count
    metadata:   dict = field(default_factory=dict)
    # metadata keys: source, page_number, total_pages,
    #                chunk_index, has_images, script


def _estimate_tokens(text: str) -> int:
    """
    Fast whitespace-based token estimator.
    BGE-M3 uses a SentencePiece tokenizer; true count ≈ word_count * 1.3 for Telugu.
    We use 1.4x multiplier to stay conservatively under context limits.
    """
    word_count = len(text.split())
    script     = detect_script(text)
    multiplier = 1.4 if script == "telugu" else 1.1
    return int(word_count * multiplier)


def _split_telugu_sentences(text: str) -> list[str]:
    """
    Rule-based sentence splitter for Telugu text.
    Splits on Telugu/Sanskrit punctuation dandas (। ॥) and standard sentence terminators.
    Preserves the delimiter by appending it to the preceding sentence.
    """
    parts = _TELUGU_SENTENCE_END.split(text)
    delimiters = _TELUGU_SENTENCE_END.findall(text)

    sentences = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        suffix = delimiters[i] if i < len(delimiters) else ""
        sentences.append(part + suffix)

    return sentences


def _split_into_sentences(text: str, nlp: Language) -> list[str]:
    """
    Dispatches to Telugu rule-based splitter or spaCy based on detected script.
    Mixed-script text uses spaCy (handles English prose mixed into Telugu documents).
    """
    script = detect_script(text)

    if script == "telugu":
        return _split_telugu_sentences(text)

    doc       = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def _build_chunks(
    sentences: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Greedy sentence-packing algorithm with overlap window.

    Strategy:
        - Pack sentences into a buffer until adding the next sentence
          would exceed chunk_size tokens.
        - Flush the buffer as a chunk.
        - Re-seed the next buffer with the last N sentences whose
          combined token count ≤ chunk_overlap (sliding window).

    This preserves semantic continuity across chunk boundaries —
    critical for multi-sentence Telugu shlokas or prose paragraphs.
    """
    if not sentences:
        return []

    chunks:       list[str]  = []
    buffer:       list[str]  = []
    buffer_tokens: int       = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # Single sentence exceeds chunk_size: emit it alone
        if sentence_tokens >= chunk_size:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0
            chunks.append(sentence)
            continue

        if buffer_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(buffer))

            # Build overlap window from tail of current buffer
            overlap_buffer: list[str] = []
            overlap_tokens: int       = 0
            for sent in reversed(buffer):
                t = _estimate_tokens(sent)
                if overlap_tokens + t <= chunk_overlap:
                    overlap_buffer.insert(0, sent)
                    overlap_tokens += t
                else:
                    break

            buffer        = overlap_buffer
            buffer_tokens = overlap_tokens

        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Converts a ParsedDocument into a flat list of TextChunks.

    Per-page pipeline:
        normalized_text → sentence split → greedy pack → TextChunk objects

    Args:
        parsed_doc:    Output of ingestion.parser.parse_document()
        chunk_size:    Maximum tokens per chunk (default from config).
        chunk_overlap: Overlap tokens between consecutive chunks (default from config).

    Returns:
        List of TextChunk objects ready for embedding.

    Raises:
        ChunkingError: If sentence splitting or packing fails on any page.
    """
    nlp    = _load_spacy(SPACY_MODEL)
    chunks: list[TextChunk] = []

    source_name = parsed_doc.doc_metadata.get("source", "unknown")

    for page in parsed_doc.pages:
        if not page.normalized_text.strip():
            continue

        try:
            sentences  = _split_into_sentences(page.normalized_text, nlp)
            raw_chunks = _build_chunks(sentences, chunk_size, chunk_overlap)
        except Exception as exc:
            raise ChunkingError(
                f"Chunking failed on page {page.page_number} of '{source_name}'.",
                details={
                    "page":   page.page_number,
                    "source": source_name,
                    "error":  str(exc),
                },
            )

        for chunk_index, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{source_name}_p{page.page_number}_{chunk_index}"
            chunks.append(
                TextChunk(
                    chunk_id  = chunk_id,
                    text      = chunk_text,
                    token_len = _estimate_tokens(chunk_text),
                    metadata  = {
                        **page.metadata,
                        "chunk_index": chunk_index,
                        "chunk_id":    chunk_id,
                        "script":      detect_script(chunk_text),
                    },
                )
            )

    return chunks


def chunk_page(
    page: ParsedPage,
    source_name:   str = "unknown",
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Single-page chunking utility — useful for streaming ingestion
    where you process pages as they arrive rather than full documents.
    """
    nlp = _load_spacy(SPACY_MODEL)

    if not page.normalized_text.strip():
        return []

    try:
        sentences  = _split_into_sentences(page.normalized_text, nlp)
        raw_chunks = _build_chunks(sentences, chunk_size, chunk_overlap)
    except Exception as exc:
        raise ChunkingError(
            f"Chunking failed on page {page.page_number}.",
            details={"page": page.page_number, "error": str(exc)},
        )

    return [
        TextChunk(
            chunk_id  = f"{source_name}_p{page.page_number}_{i}",
            text      = chunk_text,
            token_len = _estimate_tokens(chunk_text),
            metadata  = {
                **page.metadata,
                "chunk_index": i,
                "chunk_id":    f"{source_name}_p{page.page_number}_{i}",
                "script":      detect_script(chunk_text),
            },
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]