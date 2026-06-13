import re
import unicodedata
from core.exceptions import NormalizationError

# Telugu Unicode block: U+0C00–U+0C7F
_TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")

# ITRANS-adjacent Romanized Telugu → Telugu script syllable map
# Ordered longest-match first to prevent partial substitutions
_ROMAN_TO_TELUGU: list[tuple[str, str]] = [
    ("aa", "ా"), ("ii", "ీ"), ("uu", "ూ"), ("ee", "ే"), ("oo", "ో"),
    ("ai", "ై"), ("au", "ౌ"), ("ch", "చ"), ("sh", "శ"), ("th", "థ"),
    ("dh", "ధ"), ("ph", "ఫ"), ("bh", "భ"), ("gh", "ఘ"), ("kh", "ఖ"),
    ("jh", "ఝ"), ("ng", "ఙ"), ("ny", "ఞ"),
    ("a",  "అ"), ("i",  "ఇ"), ("u",  "ఉ"), ("e",  "ఎ"), ("o",  "ఒ"),
    ("k",  "క"), ("g",  "గ"), ("j",  "జ"), ("t",  "ట"), ("d",  "డ"),
    ("n",  "న"), ("p",  "ప"), ("b",  "బ"), ("m",  "మ"), ("y",  "య"),
    ("r",  "ర"), ("l",  "ల"), ("v",  "వ"), ("s",  "స"), ("h",  "హ"),
    ("f",  "ఫ"), ("z",  "జ"),
]

# Common query-level Romanized Telugu phrases → normalized Telugu
_PHRASE_MAP: dict[str, str] = {
    "rama evaru":       "రామ ఎవరు",
    "krishna evaru":    "కృష్ణ ఎవరు",
    "ramayanam":        "రామాయణం",
    "mahabharatam":     "మహాభారతం",
    "bhagavad gita":    "భగవద్గీత",
    "telugu sahityam":  "తెలుగు సాహిత్యం",
    "andhra":           "ఆంధ్ర",
}

# ── Telugu Unicode constants ───────────────────────────────────────────────────
_VIRAMA        = "\u0C4D"   # ్  (halant / killer)
_ANUSVARA      = "\u0C02"   # ం
_VISARGA       = "\u0C03"   # ః
_CHANDRABINDU  = "\u0C01"   # ఁ

# Dependent vowel signs (matras) — U+0C3E to U+0C4C + U+0C4E-U+0C56
_MATRA_RANGE   = re.compile(r"[\u0C3E-\u0C4C\u0C4E-\u0C56]")

# Duplicate matra pattern: same matra repeated (e.g., ాా, ీీ)
# This is the most common PDF extraction artifact in Telugu
_DUPLICATE_MATRA = re.compile(r"([\u0C3E-\u0C4C\u0C4E-\u0C56])\1+")

# Duplicate anusvara/visarga
_DUPLICATE_DIACRITIC = re.compile(r"([\u0C01-\u0C03])\1+")

# Virama followed by a space then a consonant — spurious space breaks conjuncts
# e.g., "స్ వభావం" → "స్వభావం"
_BROKEN_CONJUNCT = re.compile(r"(\u0C4D)\s+(?=[\u0C15-\u0C39\u0C58-\u0C5A])")

# Lone virama at end of word (orphaned killer — PDF artifact)
_ORPHAN_VIRAMA = re.compile(r"\u0C4D(\s|$)")

# Zero-width non-joiner / zero-width joiner artifacts from bad PDF copy
_ZWCHAR = re.compile(r"[\u200B-\u200F\u00AD\uFEFF]")


def detect_script(text: str) -> str:
    """
    Returns 'telugu' if Telugu codepoints dominate,
    'roman' if ASCII-only, 'mixed' otherwise.
    """
    telugu_chars = len(_TELUGU_RANGE.findall(text))
    ascii_chars  = len(re.findall(r"[a-zA-Z]", text))
    total        = telugu_chars + ascii_chars

    if total == 0:
        return "unknown"
    if telugu_chars / total > 0.6:
        return "telugu"
    if ascii_chars / total > 0.9:
        return "roman"
    return "mixed"


def unicode_normalize(text: str) -> str:
    """
    NFC normalization — collapses decomposed Telugu conjuncts
    into canonical composed forms.
    """
    try:
        return unicodedata.normalize("NFC", text)
    except Exception as exc:
        raise NormalizationError(
            "Unicode NFC normalization failed.",
            details={"input_snippet": text[:80], "error": str(exc)},
        )


def repair_telugu_unicode(text: str) -> str:
    """
    Repairs common Telugu Unicode artifacts introduced by PDF extractors
    (PyMuPDF, pdfminer, etc.) and OCR engines.

    Fixes applied in order:
      1. Strip zero-width / invisible chars (ZWSP, ZWNJ, SHY, BOM)
      2. Remove duplicate matras       — ాా → ా   (most common artifact)
      3. Remove duplicate diacritics   — ంం → ం
      4. Heal broken conjuncts         — స్ వభావం → స్వభావం
      5. Drop orphaned virama at word boundaries

    This runs AFTER NFC so canonical forms are already composed.
    Only applied to text detected as Telugu or mixed script.
    """
    # 1. Zero-width artifacts
    text = _ZWCHAR.sub("", text)

    # 2. Duplicate matras (e.g., స్వభావానిి → స్వభావాని is closer, not perfect
    #    but prevents embedding corruption from doubled codepoints)
    text = _DUPLICATE_MATRA.sub(r"\1", text)

    # 3. Duplicate anusvara / visarga / chandrabindu
    text = _DUPLICATE_DIACRITIC.sub(r"\1", text)

    # 4. Broken conjuncts — virama + spurious space + consonant
    text = _BROKEN_CONJUNCT.sub(_VIRAMA, text)

    # 5. Orphaned virama at word boundary (replace with space)
    text = _ORPHAN_VIRAMA.sub(r"\1", text)

    return text


def transliterate_roman_to_telugu(text: str) -> str:
    """
    Converts Romanized Telugu query text to Telugu script.
    Strategy:
      1. Check phrase-level map first (preserves multi-word idioms).
      2. Fall back to syllable-level longest-match substitution.
    """
    lowered = text.lower().strip()

    # Phase 1: phrase-level substitution
    for roman_phrase, telugu_phrase in _PHRASE_MAP.items():
        lowered = lowered.replace(roman_phrase, telugu_phrase)

    # Phase 2: syllable-level substitution on remaining ASCII tokens
    tokens = lowered.split()
    result_tokens: list[str] = []

    for token in tokens:
        # Skip tokens already in Telugu script
        if _TELUGU_RANGE.search(token):
            result_tokens.append(token)
            continue

        # Longest-match greedy syllable walk
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
                output += token[i]   # passthrough punctuation / digits
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
          → Telugu Unicode repair  (NEW — fixes PDF extraction artifacts)
          → transliterate Romanized → Telugu (if enabled & script is roman/mixed)
          → collapse whitespace

    Args:
        text:           Raw input string (query or document chunk).
        transliterate:  If True, run Roman→Telugu transliteration on non-Telugu text.

    Returns:
        Normalized string ready for embedding.
    """
    if not isinstance(text, str):
        raise NormalizationError(
            "normalize() expects a str input.",
            details={"received_type": type(text).__name__},
        )

    # Strip null bytes and control characters (common in PDF extractions)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Unicode NFC — canonical composition first
    text = unicode_normalize(text)

    # Telugu artifact repair — runs after NFC so composed forms are stable
    script = detect_script(text)
    if script in ("telugu", "mixed"):
        text = repair_telugu_unicode(text)

    # Transliteration pass (Roman/mixed queries only)
    if transliterate and script in ("roman", "mixed"):
        text = transliterate_roman_to_telugu(text)

    # Collapse excess whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text