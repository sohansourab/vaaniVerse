import re
import unicodedata

from core.exceptions import NormalizationError

# Telugu Unicode block: U+0C00–U+0C7F
_TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")

# ── Romanized-Telugu transliteration tables ─────────────────────────────────
# Telugu is an abugida: a consonant carries an inherent "a", and every other
# vowel is written as a dependent sign (matra) attached to that consonant.
# A flat 1:1 substitution therefore cannot work — it emits the *independent*
# vowel letter in vowel position too, turning "eruka" into "ఎరఉకఅ" instead of
# "ఎరుక". `_CONSONANTS` and `_VOWELS` are the two halves that the state
# machine in `_transliterate_token` recombines.
#
# Applied to MIXED-script text, and to pure-Roman text that failed the
# English-query check. The tables are scheme-agnostic rules, not a word list:
# no corpus- or benchmark-specific vocabulary is special-cased here.

# Consonants, longest roman form first (so "kh" matches before "k").
# Plain "t"/"d"/"n" map to the DENTAL series, per ISO 15919. The accepted
# input alphabet is lower-case ASCII only, so retroflex (ట/డ/ణ) cannot be
# expressed — inputs that need it will transliterate to the dental series.
_CONSONANTS: list[tuple[str, str]] = [
    ("kh", "ఖ"), ("gh", "ఘ"), ("ch", "చ"), ("jh", "ఝ"), ("ny", "ఞ"),
    ("ng", "ఙ"), ("th", "థ"), ("dh", "ధ"), ("ph", "ఫ"), ("bh", "భ"),
    ("sh", "శ"),
    ("k", "క"), ("g", "గ"), ("j", "జ"), ("t", "త"), ("d", "ద"),
    ("n", "న"), ("p", "ప"), ("b", "బ"), ("m", "మ"), ("y", "య"),
    ("r", "ర"), ("l", "ల"), ("v", "వ"), ("s", "స"), ("h", "హ"),
    ("f", "ఫ"), ("z", "జ"),
]

# Vowels: (roman, independent form, dependent sign / matra).
# An empty matra is the inherent vowel — after a consonant it is written by
# adding nothing at all.
_VOWELS: list[tuple[str, str, str]] = [
    ("aa", "ఆ", "ా"), ("ai", "ఐ", "ై"), ("au", "ఔ", "ౌ"),
    ("ii", "ఈ", "ీ"), ("ee", "ఏ", "ే"), ("uu", "ఊ", "ూ"), ("oo", "ఓ", "ో"),
    ("a", "అ", ""), ("i", "ఇ", "ి"), ("u", "ఉ", "ు"),
    ("e", "ఎ", "ె"), ("o", "ఒ", "ొ"),
]

# Every roman unit, longest first, tagged so the state machine can tell
# consonants from vowels. `sorted` is stable, so at equal length consonants
# (listed first) win — "sh" matches before "s" + "h".
_ROMAN_UNITS: list[tuple[str, str, object]] = sorted(
    [(r, "c", t) for r, t in _CONSONANTS]
    + [(r, "v", (i, m)) for r, i, m in _VOWELS],
    key=lambda unit: -len(unit[0]),
)

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
_ANUSVARA = "\u0C02"  # ం  (word-final or pre-consonantal "m")

# Dependent vowel signs (matras) — U+0C3E to U+0C4C + U+0C4E-U+0C56
_DUPLICATE_MATRA = re.compile(r"([\u0C3E-\u0C4C\u0C4E-\u0C56])\1+")

# Duplicate anusvara / visarga / chandrabindu
_DUPLICATE_DIACRITIC = re.compile(r"([\u0C01-\u0C03])\1+")

# Virama followed by a space then a consonant — spurious space breaks
# conjuncts, e.g. "స్ వభావం" → "స్వభావం"
_BROKEN_CONJUNCT = re.compile(r"(\u0C4D)\s+(?=[\u0C15-\u0C39\u0C58-\u0C5A])")

# A space injected between a base letter and a *combining* mark it belongs to.
# This is the single most common Telugu PDF-extraction artifact and it directly
# weakens embeddings: "మ ైన" renders as మ + space + ై, which is not the word
# "మైన" to the tokenizer. A combining mark (matra / virama / anusvara / visarga)
# can never legitimately begin a word, so removing the space is always a repair,
# never a corruption.
_SPACED_DIACRITIC = re.compile(
    r"([\u0C00-\u0C7F])\s+(?=[\u0C01-\u0C04\u0C3E-\u0C4C\u0C4E-\u0C56])"
)

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
        5. Rejoin a spaced combining mark  — మ ైన → మైన.
        6. Drop orphaned virama at word boundaries.

    Must run after `unicode_normalize` so canonical forms are already
    composed. Only meaningful for text detected as Telugu or mixed.
    """
    text = _ZWCHAR.sub("", text)
    text = _DUPLICATE_MATRA.sub(r"\1", text)
    text = _DUPLICATE_DIACRITIC.sub(r"\1", text)
    text = _BROKEN_CONJUNCT.sub(_VIRAMA, text)
    # Run repeatedly: "మ  ైన" (double space) needs more than one pass.
    previous = None
    while previous != text:
        previous = text
        text = _SPACED_DIACRITIC.sub(r"\1", text)
    text = _ORPHAN_VIRAMA.sub(r"\1", text)
    return text


def _apply_phrase_map(text: str) -> str:
    """Apply exact-substring Romanized-Telugu phrase substitutions."""
    lowered = text.lower().strip()
    for roman_phrase, telugu_phrase in _PHRASE_MAP:
        lowered = lowered.replace(roman_phrase, telugu_phrase)
    return lowered


def _match_unit(token: str, index: int) -> tuple[str, str, object] | None:
    """Return the longest roman unit at `index` as (roman, kind, value)."""
    for roman, kind, value in _ROMAN_UNITS:
        if token.startswith(roman, index):
            return roman, kind, value
    return None


def _transliterate_token(token: str) -> str:
    """
    Transliterate one Romanized-Telugu word using abugida rules.

    A consonant carries an inherent "a"; a vowel that follows a consonant is
    written as a dependent sign rather than an independent letter; a consonant
    that follows another consonant takes a virama to form a cluster; and a
    word-final (or pre-consonantal) "m" becomes anusvara.

    Deliberately scheme-agnostic — it assumes *consistent* Romanization and
    special-cases no vocabulary. Ad-hoc phonetic spellings are not recoverable
    by any fixed rule set and will come out wrong: "emiti" for ఏమిటి yields
    ఎమిటి, and "ghadha" for గాఢ yields ఘధ.
    """
    out: list[str] = []
    prev_consonant = False
    i = 0

    while i < len(token):
        unit = _match_unit(token, i)
        if unit is None:
            # Punctuation, digits, or an unmapped character — pass through and
            # close any consonant left hanging.
            if prev_consonant:
                out.append(_VIRAMA)
                prev_consonant = False
            out.append(token[i])
            i += 1
            continue

        roman, kind, value = unit

        if kind == "c":
            if roman == "m":
                following = _match_unit(token, i + len(roman))
                if following is None or following[1] == "c":
                    # Word-final or before another consonant → anusvara.
                    out.append(_ANUSVARA)
                    prev_consonant = False
                    i += len(roman)
                    continue
            if prev_consonant:
                out.append(_VIRAMA)
            out.append(value)  # type: ignore[arg-type]
            prev_consonant = True
        else:
            independent, matra = value  # type: ignore[misc]
            if matra == "":
                # Inherent "a": written only when it is not carried by a
                # preceding consonant.
                if not prev_consonant:
                    out.append(independent)
            else:
                out.append(matra if prev_consonant else independent)
            prev_consonant = False

        i += len(roman)

    if prev_consonant:
        out.append(_VIRAMA)
    return "".join(out)


def transliterate_roman_to_telugu(text: str) -> str:
    """
    Convert Romanized Telugu text to Telugu script.

    Strategy:
        1. Apply the phrase-level map first (preserves multi-word idioms that
           Romanization rules would otherwise break apart).
        2. Transliterate each remaining Roman token with the abugida-aware
           state machine in `_transliterate_token`.

    Called for MIXED-script text and for pure-Roman text that failed the
    English-query check — never on plain English.
    """
    lowered = _apply_phrase_map(text)

    result_tokens: list[str] = []
    for token in lowered.split():
        if _TELUGU_RANGE.search(token):
            result_tokens.append(token)
        else:
            result_tokens.append(_transliterate_token(token))

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
                          else phrase map + abugida state machine
                mixed   → full transliteration
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
        if script == "mixed":
            # Telugu + Roman interleaved — full transliteration
            text = transliterate_roman_to_telugu(text)
        elif script == "roman" and not is_english_query(text):
            # Likely Romanized Telugu, not English — phrase map, then the
            # abugida state machine. Previously this path applied the phrase
            # map only, which left ~all Romanized queries untransliterated.
            text = transliterate_roman_to_telugu(text)
        # script == "telugu": no-op
        # script == "roman" AND is_english_query(): no-op, leave English alone

    text = re.sub(r"\s+", " ", text).strip()
    return text