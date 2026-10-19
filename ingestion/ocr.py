"""
Pluggable OCR for scanned or image-only pages.

Only engines with *verified* Telugu support are registered here:

    engine       code   evidence
    tesseract    tel    tel.traineddata ships in the official tessdata-fast
    easyocr      te     easyocr/character/te_char.txt ships in the EasyOCR repo

PaddleOCR is deliberately absent — it has no Telugu dictionary in
``ppocr/utils/dict``, so advertising Telugu support for it would be wrong.
Surya is absent because its repository contains no Telugu language entry.

Engines are imported lazily and cached, so a CPU-only deployment that never
meets a scanned page never pays for any of this, and never needs the binary.
The ``auto`` engine picks the first available option in ``_PREFERENCE`` order.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from core import config
from core.exceptions import OCREngineError

# Canonical language codes are ISO 639-1 ("te", "en"). Each engine has its own
# spelling, so translate rather than making the caller remember both.
_TESSERACT_CODES = {
    "te": "tel", "en": "eng", "hi": "hin", "ta": "tam", "kn": "kan",
    "ml": "mal", "bn": "ben", "gu": "guj", "mr": "mar", "pa": "pan",
    "or": "ori", "as": "asm", "sa": "san", "ur": "urd", "ne": "nep",
}

_PREFERENCE = ("tesseract", "easyocr")

_TESSERACT_EXE_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
)

# The Windows installer ships only eng/osd and writes into Program Files, which
# is not user-writable, so `scripts/fetch_ocr_tessdata.py` stages a complete set
# under data/tessdata (gitignored). When it exists it is used *instead of* the
# system directory, so it must be self-contained.
_LOCAL_TESSDATA = config.BASE_DIR / "data" / "tessdata"


def _tessdata_dir() -> Path | None:
    """The repo-local tessdata directory, if it has been staged."""
    if (_LOCAL_TESSDATA / "tel.traineddata").exists():
        return _LOCAL_TESSDATA
    return None


def _ensure_tessdata_env() -> None:
    """
    Point Tesseract at the staged data directory, when there is one.

    Uses TESSDATA_PREFIX rather than --tessdata-dir on purpose: the option has
    to survive pytesseract's shell splitting, and a quoted path arrives at
    Tesseract with the quote characters still attached.
    """
    local = _tessdata_dir()
    if local is not None:
        os.environ["TESSDATA_PREFIX"] = str(local)


@dataclass
class OCRResult:
    """Text recovered from one image, with provenance."""
    text: str
    engine: str
    mean_confidence: float | None = None


# ── language helpers ───────────────────────────────────────────────────────────

def _split_langs(spec: str) -> list[str]:
    """Parse "te,en" / "te+en" / "te eng" into ["te", "en"]."""
    return [p for p in spec.replace("+", ",").replace(" ", ",").split(",") if p.strip()]


def _tesseract_langs(spec: str) -> str:
    codes = [_TESSERACT_CODES.get(c.lower(), c.lower()) for c in _split_langs(spec)]
    return "+".join(codes)


def _easyocr_langs(spec: str) -> list[str]:
    # EasyOCR already uses ISO 639-1.
    return [c.lower() for c in _split_langs(spec)]


# ── engine availability ────────────────────────────────────────────────────────

def find_tesseract_binary() -> str | None:
    override = os.getenv("TESSERACT_CMD")
    if override and Path(override).exists():
        return override
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_EXE_CANDIDATES:
        path = candidate.format(user=Path.home().name)
        if Path(path).exists():
            return path
    return None


@lru_cache(maxsize=1)
def _probe_tesseract() -> str:
    """Return "" when usable, otherwise the reason it is not."""
    binary = find_tesseract_binary()
    if binary is None:
        return "tesseract binary not found (install Tesseract, or set TESSERACT_CMD)"

    local = _tessdata_dir()
    if local is not None:
        return ""  # staged locally, already verified to contain tel.traineddata

    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = binary
        langs = set(pytesseract.get_languages(config=""))
        if "tel" not in langs:
            return (
                "the 'tel' (Telugu) traineddata is missing — "
                "run scripts/fetch_ocr_tessdata.py"
            )
        return ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"pytesseract unusable: {exc}"


@lru_cache(maxsize=1)
def _probe_easyocr() -> str:
    """Return "" when usable, otherwise the reason it is not."""
    try:
        import easyocr  # noqa: F401

        return ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"easyocr not importable: {exc}"


_PROBES = {"tesseract": _probe_tesseract, "easyocr": _probe_easyocr}


def probe_engines() -> dict[str, str]:
    """``{engine: ""}`` when usable, ``{engine: reason}`` otherwise. Cached."""
    return {engine: probe() for engine, probe in _PROBES.items()}


def available_engines() -> list[str]:
    """Installed engines that also have a Telugu model, best first."""
    return [engine for engine in _PREFERENCE if _PROBES[engine]() == ""]


def resolve_engine(requested: str | None = None) -> str:
    """
    Turn a config value into a concrete engine name, or explain why it is not.

    Deliberately short-circuits: resolving ``auto`` must not import easyocr
    (and therefore torch) just to discover that tesseract already works.

    Raises:
        OCREngineError: if nothing usable is available, with per-engine reasons.
    """
    requested = (requested or "auto").lower()

    if requested == "auto":
        for engine in _PREFERENCE:
            if _PROBES[engine]() == "":
                return engine
        reasons = "; ".join(f"{e}: {_PROBES[e]()}" for e in _PREFERENCE)
        raise OCREngineError("No OCR engine is available.", details={"attempted": reasons})

    if requested not in _PREFERENCE:
        raise OCREngineError(
            f"Unknown OCR engine {requested!r}.", details={"supported": list(_PREFERENCE)}
        )

    reason = _PROBES[requested]()
    if reason:
        raise OCREngineError(
            f"OCR engine {requested!r} is not usable.", details={"reason": reason}
        )
    return requested


# ── pre-processing ─────────────────────────────────────────────────────────────

def _otsu_threshold(gray: Image.Image) -> int:
    """Otsu's threshold from a grayscale image's histogram (pure PIL)."""
    histogram = gray.histogram()[:256]
    total = sum(histogram)
    if total == 0:
        return 128
    sum_total = sum(i * h for i, h in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    best_threshold, best_variance = 128, -1.0
    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = t
    return best_threshold


def _preprocess(image: Image.Image) -> Image.Image:
    """
    Grayscale → autocontrast → Otsu binarize.

    Old Telugu print is low-contrast and foxed; binarizing against an adaptive
    threshold measurably helps Tesseract more than it hurts a clean scan, and the
    step is cheap relative to recognition. Kept pure-Pillow so no OpenCV or
    scikit-image dependency is pulled in for it.
    """
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    threshold = _otsu_threshold(gray)
    return gray.point(lambda pixel: 255 if pixel > threshold else 0, mode="1")


# ── engines ────────────────────────────────────────────────────────────────────

def _run_tesseract(image: Image.Image, lang_spec: str, psm: int) -> OCRResult:
    import pytesseract

    binary = find_tesseract_binary()
    if binary:
        pytesseract.pytesseract.tesseract_cmd = binary

    _ensure_tessdata_env()

    lang = _tesseract_langs(lang_spec)
    config_str = f"--psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config_str)

    confidence: float | None = None
    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config=config_str, output_type=pytesseract.Output.DICT
        )
        scores = [float(c) for c in data.get("conf", []) if str(c) not in ("-1", "")]
        if scores:
            confidence = sum(scores) / len(scores) / 100.0
    except Exception:
        confidence = None  # confidence is a nicety, never fatal

    return OCRResult(text=text, engine="tesseract", mean_confidence=confidence)


@lru_cache(maxsize=4)
def _easyocr_reader(lang_tuple: tuple[str, ...], gpu: bool):
    import easyocr

    return easyocr.Reader(list(lang_tuple), gpu=gpu, verbose=False)


def _run_easyocr(image: Image.Image, lang_spec: str, psm: int = 3) -> OCRResult:
    import numpy as np

    langs = tuple(_easyocr_langs(lang_spec))
    gpu = os.getenv("OCR_DEVICE", "cpu").lower() == "cuda"
    reader = _easyocr_reader(langs, gpu)

    detections = reader.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
    pieces, scores = [], []
    for _bbox, text, conf in detections:
        if text and text.strip():
            pieces.append(text.strip())
            scores.append(float(conf))

    return OCRResult(
        text="\n".join(pieces),
        engine="easyocr",
        mean_confidence=(sum(scores) / len(scores)) if scores else None,
    )


_ENGINES = {"tesseract": _run_tesseract, "easyocr": _run_easyocr}


# ── public entry point ─────────────────────────────────────────────────────────

def ocr_image(
    image: Image.Image,
    engine: str | None = None,
    lang: str = "te,en",
    psm: int | None = None,
    preprocess: bool | None = None,
) -> OCRResult:
    """
    Transcribe one image.

    Args:
        image:      Anything Pillow can open — a rasterized PDF page or a crop.
        engine:     "tesseract" | "easyocr" | "auto" | None (auto).
        lang:       ISO 639-1 codes, comma/plus separated.
        psm:        Tesseract page-segmentation mode. Defaults to config.OCR_PSM
                    (6 = a uniform block of text, which fits a body page better
                    than 3's automatic layout analysis on marked-up scans).
        preprocess: Grayscale + autocontrast + Otsu binarize before recognition.
                    Defaults to config.OCR_PREPROCESS.

    Raises:
        OCREngineError: if the requested engine is missing or fails a hard error.
    """
    if image is None:
        raise OCREngineError("No image supplied to OCR.")

    chosen = resolve_engine(engine)
    runner = _ENGINES[chosen]

    if preprocess is None:
        preprocess = config.OCR_PREPROCESS
    if preprocess:
        image = _preprocess(image)

    if psm is None:
        psm = config.OCR_PSM

    try:
        if chosen == "tesseract":
            return runner(image, lang, psm)
        return runner(image, lang)
    except OCREngineError:
        raise
    except Exception as exc:
        raise OCREngineError(
            f"{chosen} failed on this image.", details={"error": str(exc), "engine": chosen}
        ) from exc
