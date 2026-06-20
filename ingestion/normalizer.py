import re
import unicodedata

from core.exceptions import NormalizationError

# Telugu Unicode block: U+0C00–U+0C7F
_TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")

# ITRANS-adjacent Romanized Telugu → Telugu script syllable map.
# Ordered longest-match first to prevent partial substitutions.
# Only ever applied to MIXED-script text — never to pure Roman/English.
_ROMAN_TO_TELUGU: list[tuple[str, str]] = [
    ("aa", "ా"), ("ii", "ీ"), ("uu", "ూ"), ("ee", "ే"), ("oo", "ో"),
    ("ai", "ై"), ("au", "ౌ"), ("ch", "చ"), ("sh", "శ"), ("th", "థ"),
    ("dh", "ధ"), ("ph", "ఫ"), ("bh", "భ"), ("gh", "ఘ"), ("kh", "ఖ"),
    ("jh", "ఝ"), ("ng", "ఙ"), ("ny", "ఞ"),
    ("a", "అ"), ("i", "ఇ"), ("u", "ఉ"), ("e", "ఎ"), ("o", "ఒ"),
    ("k", "క"), ("g", "గ"), ("j", "జ"), ("t", "ట"), ("d", "డ"),
    ("n", "న"), ("p", "ప"), ("b", "బ"), ("m", "మ"), ("y", "య"),
    ("r", "ర"), ("l", "ల"), ("v", "వ"), ("s", "స"), ("h", "హ"),
    ("f", "ఫ"), ("z", "జ"),
]

# Romanized Telugu phrases → Telugu script.
# Safe for PURE Roman queries — exact substring match, no syllable walk.
# List, not dict, so longer/more specific phrases can be added without
# worrying about key collisions; order matters for overlapping substrings.
_PHRASE_MAP: list[tuple[str, str]] = [
    ("nenu evarini", "నేను ఎవరిని"),
    ("nenu evaru", "నేను ఎవరు"),
    ("rama evaru", "రామ ఎవరు"),
    ("krishna evaru", "కృష్ణ ఎవరు"),
    ("ramayanam", "రామాయణం"),
    ("mahabharatam", "మహాభారతం"),
    ("bhagavad gita", "భగవద్గీత"),
    ("telugu sahityam", "తెలుగు సాహిత్యం"),
    ("andhra", "ఆంధ్ర"),
    ("atma vicharana", "ఆత్మ విచారణ"),
    ("atma", "ఆత్మ"),
    ("sat chit ananda", "సత్ చిత్ ఆనందం"),
    ("dharma", "ధర్మం"),
    ("karma", "కర్మ"),
    ("moksha", "మోక్షం"),
    ("brahman", "బ్రహ్మం"),
    ("advaita", "అద్వైత"),
]

# ── Telugu Unicode constants ────────────────────────────────────────────────
_VIRAMA = "\u0C4D"  # ్  (halant / killer)

# Dependent vowel signs (matras) — U+0C3E to U+0C4C + U+0C4E-U+0C56
_DUPLICATE_MATRA = re.compile(r"([\u0C3E-\u0C4C\u0C4E-\u0C56])\1+")

# Duplicate anusvara / visarga / chandrabindu
_DUPLICATE_DIACRITIC = re.compile(r"([\u0C01-\u0C03])\1+")

# Virama followed by a space then a consonant — spurious space breaks
# conjuncts, e.g. "స్ వభావం" → "స్వభావం"
_BROKEN_CONJUNCT = re.compile(r"(\u0C4D)\s+(?=[\u0C15-\u0C39\u0C58-\u0C5A])")

# Lone virama at end of word (orphaned killer — PDF artifact)
_ORPHAN_VIRAMA = re.compile(r"\u0C4D(\s|$)")

# Zero-width / invisible Unicode artifacts from bad PDF copy
_ZWCHAR = re.compile(r"[\u200B-\u200F\u00AD\uFEFF]")

# Control characters commonly left behind by PDF extractors
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# English stopwords used to distinguish genuine English queries from
# Romanized Telugu when the script detector reports "roman".
# Two or more hits ⇒ treat the query as English and skip transliteration.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "is", "it", "its", "may", "me", "might", "my", "not", "of", "on",
    "or", "our", "shall", "should", "so", "than", "that", "the", "their",
    "them", "then", "there", "they", "this", "to", "was", "we", "what",
    "when", "where", "who", "why", "will", "with", "would",
})


def detect_script(text: str) -> str:
    """
    Classify the dominant script of `text`.

    Returns:
        "telugu"  — Telugu codepoints make up > 60% of alphabetic chars
        "roman"   — ASCII letters make up > 90% of alphabetic chars
        "mixed"   — neither threshold is met (Telugu + Roman interleaved)
        "unknown" — no alphabetic characters at all
    """
    telugu_chars = len(_TELUGU_RANGE.findall(text))
    ascii_chars = len(re.findall(r"[a-zA-Z]", text))
    total = telugu_chars + ascii_chars

    if total == 0:
        return "unknown"
    if telugu_chars / total > 0.6:
        return "telugu"
    if ascii_chars / total > 0.9:
        return "roman"
    return "mixed"


def is_english_query(text: str) -> bool:
    """
    Heuristically decide whether a Roman-script string is genuine English
    rather than Romanized Telugu.

    Uses stopword overlap: two or more English stopwords is treated as a
    strong signal of English prose ("who am i", "what is dharma"), since
    Romanized Telugu queries rarely contain English function words.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for tok in tokens if tok in _ENGLISH_STOPWORDS)
    return hits >= 2


def unicode_normalize(text: str) -> str:
    """NFC-normalize text, collapsing decomposed Telugu conjuncts."""
    try:
        return unicodedata.normalize("NFC", text)
    except Exception as exc:
        raise NormalizationError(
            "Unicode NFC normalization failed.",
            details={"input_snippet": text[:80], "error": str(exc)},
        ) from exc


def repair_telugu_unicode(text: str) -> str:
    """
    Repair common Telugu Unicode artifacts introduced by PDF extractors
    (PyMuPDF, pdfminer, etc.) and OCR engines.

    Fixes are applied in order:
        1. Strip zero-width / invisible chars (ZWSP, ZWNJ, SHY, BOM).
        2. Collapse duplicate matras       — ాా → ా  (common artifact).
        3. Collapse duplicate diacritics   — ంం → ం.
        4. Heal broken conjuncts           — స్ వభావం → స్వభావం.
        5. Drop orphaned virama at word boundaries.

    Must run after `unicode_normalize` so canonical forms are already
    composed. Only meaningful for text detected as Telugu or mixed.
    """
    text = _ZWCHAR.sub("", text)
    text = _DUPLICATE_MATRA.sub(r"\1", text)
    text = _DUPLICATE_DIACRITIC.sub(r"\1", text)
    text = _BROKEN_CONJUNCT.sub(_VIRAMA, text)
    text = _ORPHAN_VIRAMA.sub(r"\1", text)
    return text


def _apply_phrase_map(text: str) -> str:
    """Apply exact-substring Romanized-Telugu phrase substitutions."""
    lowered = text.lower().strip()
    for roman_phrase, telugu_phrase in _PHRASE_MAP:
        lowered = lowered.replace(roman_phrase, telugu_phrase)
    return lowered


def transliterate_roman_to_telugu(text: str) -> str:
    """
    Convert Romanized Telugu text to Telugu script.

    Strategy:
        1. Apply the phrase-level map first (preserves multi-word idioms).
        2. Fall back to syllable-level longest-match substitution for
           any tokens the phrase map did not cover.

    This is the "aggressive" path and should only be called for text
    already confirmed to be Romanized Telugu (mixed script, or roman
    script that failed the English-query check) — never on plain English.
    """
    lowered = _apply_phrase_map(text)

    tokens = lowered.split()
    result_tokens: list[str] = []

    for token in tokens:
        if _TELUGU_RANGE.search(token):
            result_tokens.append(token)
            continue

        output = ""
        i = 0
        while i < len(token):
            matched = False
            for roman, telugu in _ROMAN_TO_TELUGU:
                if token[i:].startswith(roman):
                    output += telugu
                    i += len(roman)
                    matched = True
                    break
            if not matched:
                output += token[i]  # passthrough punctuation / digits
                i += 1
        result_tokens.append(output)

    return " ".join(result_tokens)


def normalize(text: str, transliterate: bool = True) -> str:
    """
    Master normalization entry point.

    Pipeline:
        raw text
          → strip control characters
          → Unicode NFC
          → Telugu Unicode artifact repair        (telugu / mixed only)
          → script-aware transliteration:
                telugu  → no-op, already native script
                roman   → English query?  skip entirely
                          else apply phrase map only (safe, no syllable walk)
                mixed   → full syllable-level transliteration
          → collapse whitespace

    Args:
        text:          Raw input string (query or document chunk).
        transliterate: If True, apply script-appropriate transliteration.
                        Has no effect on text already detected as Telugu.

    Returns:
        Normalized string ready for embedding.

    Raises:
        NormalizationError: If `text` is not a string, or NFC normalization
            fails on malformed input.
    """
    if not isinstance(text, str):
        raise NormalizationError(
            "normalize() expects a str input.",
            details={"received_type": type(text).__name__},
        )

    text = _CONTROL_CHARS.sub("", text)
    text = unicode_normalize(text)

    script = detect_script(text)

    if script in ("telugu", "mixed"):
        text = repair_telugu_unicode(text)

    if transliterate:
        if script == "roman" and not is_english_query(text):
            # Likely Romanized Telugu, not English — safe phrase substitution
            text = _apply_phrase_map(text)
        elif script == "mixed":
            # Telugu + Roman interleaved — full syllable transliteration
            text = transliterate_roman_to_telugu(text)
        # script == "telugu": no-op
        # script == "roman" AND is_english_query(): no-op, leave English alone

    text = re.sub(r"\s+", " ", text).strip()
    return text