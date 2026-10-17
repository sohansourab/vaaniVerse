from dataclasses import dataclass
from core.config import TOP_K
from core.exceptions import NoContextRetrievedError, PromptConstructionError
from ingestion.normalizer import is_english_query
from retrieval.vector_store import RetrievedChunk

# ── Maximum characters per chunk shown in prompt ───────────────────────────────
# Prevents context window overflow for long PDFs with large chunks.
# BGE-M3 chunks at 512 tokens ≈ ~1800 chars; we cap at 1500 for safety.
_MAX_CHUNK_CHARS: int = 1500

# ── Source citation format ─────────────────────────────────────────────────────
# Each context block is labelled so the LLM can cite sources explicitly.
# Format: [SOURCE 1] filename.pdf | Page 3 | Score: 0.87
_SOURCE_LABEL_TEMPLATE: str = "[SOURCE {i}] {source} | Page {page} | Score: {score:.2f}"


@dataclass
class PromptPackage:
    """
    Complete prompt payload passed to generator.py.
    Keeps system prompt and user turn separated so both
    Groq (ChatCompletion) and Ollama (generate) APIs can consume it.
    """
    system_prompt:  str
    user_prompt:    str
    context_chunks: list[RetrievedChunk]   # retained for API response metadata
    query:          str


# ══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_TELUGU = """You are VāṇiVerse, a culturally aware AI assistant specializing in Telugu language, literature, and heritage.

STRICT RULES — follow without exception:
1. Answer ONLY using the SOURCE BLOCKS provided below. Do not use your training knowledge.
2. If the answer is not found in the sources, respond exactly: "ఈ సమాచారం అందుబాటులో లేదు." (This information is not available in the provided sources.)
3. Always cite your source using the label [SOURCE N] at the end of each claim.
4. Preserve Telugu script exactly as it appears in the sources. Do not transliterate unless the user asked in Roman script.
5. If the user query is in Romanized Telugu, respond in Telugu script.
6. Never fabricate names, dates, verses, or historical facts.
7. Cite only the [SOURCE N] labels that appear in the SOURCE BLOCKS. Never invent a source number."""

_SYSTEM_PROMPT_ENGLISH = """You are VāṇiVerse, a culturally aware AI assistant specializing in Indian linguistic heritage, with deep knowledge of Telugu literature and history.

STRICT RULES — follow without exception:
1. Answer ONLY using the SOURCE BLOCKS provided below. Do not use your training knowledge.
2. ALWAYS respond in English. Never respond in Telugu even if the sources are in Telugu.
3. If the answer is not found in the sources, respond exactly: "The requested information is not available in the provided sources."
4. Always cite your source using the label [SOURCE N] at the end of each claim.
5. Translate relevant Telugu passages from sources into English in your answer.
6. Never fabricate names, dates, verses, or historical facts.
7. Cite only the [SOURCE N] labels that appear in the SOURCE BLOCKS. Never invent a source number."""

_SYSTEM_PROMPT_MIXED = """You are VāṇiVerse, a bilingual AI assistant for Telugu and English queries about Indian linguistic and cultural heritage.

STRICT RULES — follow without exception:
1. Answer ONLY using the SOURCE BLOCKS provided below. Do not use your training knowledge.
2. Match the response language to the query language. If the query is mixed Telugu-English, respond in the same mix.
3. If the answer is not found in the sources, say so explicitly in the same language as the query.
4. Always cite your source using the label [SOURCE N] at the end of each claim.
5. Preserve Telugu script exactly when quoting from sources.
6. Never fabricate names, dates, verses, or historical facts.
7. Cite only the [SOURCE N] labels that appear in the SOURCE BLOCKS. Never invent a source number."""


# ══════════════════════════════════════════════════════════════════════════════
# Internal Builders
# ══════════════════════════════════════════════════════════════════════════════

def _answer_language(script: str, query: str) -> str:
    """
    Decide which language the answer should be written in.

    `script` alone is not enough to choose a system prompt, because
    detect_script() reports "roman" for two different situations: a genuine
    English query, and Romanized Telugu. _SYSTEM_PROMPT_TELUGU already carries
    the rule for the latter ("If the user query is in Romanized Telugu, respond
    in Telugu script"), but that rule was unreachable — every Roman query fell
    through to the English prompt, whose rule 2 forbids Telugu. The visible
    effect was that "nenu evaru?" was answered in English.

    The disambiguator is the same stopword heuristic the normalizer uses to
    decide whether to transliterate at all, applied to the *normalized* query:
    by this point Romanized Telugu has already been transliterated into Telugu
    script and so no longer looks like English.

    Args:
        script: Output of detect_script() on the raw query.
        query:  The query as it will be shown to the model (post-normalize).

    Returns:
        "telugu", "english" or "mixed".
    """
    if script == "telugu":
        return "telugu"
    if script == "mixed":
        return "mixed"
    if script == "roman":
        return "english" if is_english_query(query) else "telugu"
    # "unknown" — no alphabetic characters to reason from.
    return "english"


def _select_system_prompt(script: str, query: str = "") -> str:
    """
    Selects the appropriate system prompt based on detected query script.

    Args:
        script: Output of ingestion.normalizer.detect_script().
        query:  The query text, needed to tell English apart from Romanized
                Telugu — both are reported as "roman". See _answer_language().
    """
    language = _answer_language(script, query)
    if language == "telugu":
        return _SYSTEM_PROMPT_TELUGU
    if language == "mixed":
        return _SYSTEM_PROMPT_MIXED
    return _SYSTEM_PROMPT_ENGLISH


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Renders retrieved chunks into a numbered SOURCE BLOCK string.

    Format per chunk:
        [SOURCE 1] ramayanam.pdf | Page 4 | Score: 0.91
        <chunk text, truncated to _MAX_CHUNK_CHARS>
        ─────────────────────────────────────────────

    This explicit labelling is what allows the LLM to cite [SOURCE N]
    in its answer and enables downstream answer verification.
    """
    blocks: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        source   = chunk.metadata.get("source", "unknown")
        page     = chunk.metadata.get("page_number", "?")
        score    = chunk.score
        text     = chunk.text[:_MAX_CHUNK_CHARS]

        if len(chunk.text) > _MAX_CHUNK_CHARS:
            text += "…"

        label = _SOURCE_LABEL_TEMPLATE.format(
            i=i, source=source, page=page, score=score
        )
        blocks.append(f"{label}\n{text}\n{'─' * 50}")

    return "\n\n".join(blocks)


def _build_user_prompt(query: str, context_block: str, source_count: int) -> str:
    """
    Assembles the user-turn prompt with query + context block.
    The explicit instruction to cite sources is repeated here (in addition
    to the system prompt) because frontier LLMs sometimes ignore system
    prompt constraints under long context pressure.

    `source_count` is stated numerically rather than left implicit: the blocks
    are labelled [SOURCE 1]..[SOURCE N], but nothing told the model that N is a
    ceiling, and a citation to [SOURCE 6] against five provided blocks points a
    reader at a source that does not exist. Measured at roughly 1 run in 6 before
    this bound was stated.
    """
    return (
        f"SOURCE BLOCKS:\n"
        f"{'═' * 50}\n"
        f"{context_block}\n"
        f"{'═' * 50}\n\n"
        f"QUERY: {query}\n\n"
        f"INSTRUCTION: Answer the query strictly using the SOURCE BLOCKS above. "
        f"Cite each claim with [SOURCE N], where N is between 1 and {source_count} — "
        f"there are exactly {source_count} sources and no others. "
        f"If the answer is absent from the sources, say so explicitly."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    query:   str,
    chunks:  list[RetrievedChunk],
    script:  str = "roman",
) -> PromptPackage:
    """
    Constructs a complete PromptPackage from a query and retrieved chunks.

    Args:
        query:   Normalized query string (post normalizer.normalize()).
        chunks:  Retrieved chunks from VectorStore.query() — already
                 filtered by score_threshold and sorted by score desc.
        script:  Output of detect_script() on the *raw* query. Combined with
                 `query` (see _answer_language) this selects the answer language.

    Returns:
        PromptPackage with system_prompt, user_prompt, context_chunks, query.

    Raises:
        NoContextRetrievedError:  If chunks list is empty.
        PromptConstructionError:  If prompt assembly fails.
    """
    if not chunks:
        raise NoContextRetrievedError(
            "No chunks retrieved above score threshold. Cannot build prompt.",
            details={"query": query, "chunks_count": 0},
        )

    if not query.strip():
        raise PromptConstructionError(
            "Query string is empty. Cannot build prompt.",
            details={"query": query},
        )

    try:
        context_block  = _format_context_block(chunks)
        system_prompt  = _select_system_prompt(script, query)
        user_prompt    = _build_user_prompt(query, context_block, len(chunks))
    except Exception as exc:
        raise PromptConstructionError(
            "Failed to assemble prompt from retrieved chunks.",
            details={"query": query, "chunks_count": len(chunks), "error": str(exc)},
        )

    return PromptPackage(
        system_prompt  = system_prompt,
        user_prompt    = user_prompt,
        context_chunks = chunks,
        query          = query,
    )


def build_no_result_prompt(query: str, script: str = "roman") -> PromptPackage:
    language = _answer_language(script, query)
    # A mixed query must not be answered with an English-only sentence, since
    # _SYSTEM_PROMPT_MIXED tells the model to match the query's language mix.
    fallback_message = {
        "telugu": "ఈ సమాచారం అందుబాటులో లేదు.",
        "mixed":  "ఈ సమాచారం అందుబాటులో లేదు. / "
                  "The requested information is not available in the provided sources.",
    }.get(language, "The requested information is not available in the provided sources.")

    system_prompt = _select_system_prompt(script, query)
    user_prompt   = (
        f"QUERY: {query}\n\n"
        f"SOURCE BLOCKS: [NONE]\n\n"
        f"INSTRUCTION: No relevant sources were found. "
        f"Respond only with: \"{fallback_message}\" — nothing else. "
        f"{'Respond in English only.' if language == 'english' else ''}"
    )

    return PromptPackage(
        system_prompt  = system_prompt,
        user_prompt    = user_prompt,
        context_chunks = [],
        query          = query,
    )