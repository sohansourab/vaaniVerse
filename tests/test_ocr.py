"""Unit tests for the OCR layer and the parser's OCR fallback.

No real engine is required: the probes and the engine runners are monkeypatched,
so these tests pass on a machine with no Tesseract and no easyocr installed.
"""
from __future__ import annotations

import pymupdf as fitz  # PyMuPDF (the `fitz` name is a deprecated alias)
import pytest
from PIL import Image

from core.exceptions import OCREngineError
from ingestion import ocr
from ingestion import parser
from ingestion.ocr import OCRResult, _easyocr_langs, _tesseract_langs


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_text_pdf(path, text: str):
    """A normal, selectable-text PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    doc.save(str(path))
    doc.close()
    return path


def _make_scanned_pdf(path):
    """An image-only page: ink, but no text layer at all."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(60, 60, 320, 260), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()
    return path


def _all_engines_available(monkeypatch):
    monkeypatch.setattr(ocr, "_PROBES", {"tesseract": lambda: "", "easyocr": lambda: ""})


def _no_engines_available(monkeypatch):
    monkeypatch.setattr(
        ocr, "_PROBES", {"tesseract": lambda: "no binary", "easyocr": lambda: "not installed"}
    )


# ── language mapping ─────────────────────────────────────────────────────────

def test_language_codes_are_translated_per_engine():
    # Tesseract spells Telugu "tel"; EasyOCR spells it "te".
    assert _tesseract_langs("te,en") == "tel+eng"
    assert _tesseract_langs("te+en") == "tel+eng"
    assert _easyocr_langs("te,en") == ["te", "en"]
    assert _easyocr_langs("te") == ["te"]


def test_unknown_language_passes_through_unchanged():
    assert _tesseract_langs("xx") == "xx"
    assert _easyocr_langs("xx") == ["xx"]


# ── engine resolution ────────────────────────────────────────────────────────

def test_resolve_rejects_unknown_engine_name():
    with pytest.raises(OCREngineError) as exc:
        ocr.resolve_engine("paddleocr")
    assert "paddleocr" in str(exc.value)


def test_resolve_auto_short_circuits_to_first_available(monkeypatch):
    _all_engines_available(monkeypatch)
    assert ocr.resolve_engine("auto") == "tesseract"  # preference order
    assert ocr.available_engines() == ["tesseract", "easyocr"]


def test_resolve_auto_reports_every_reason_when_nothing_works(monkeypatch):
    _no_engines_available(monkeypatch)
    with pytest.raises(OCREngineError) as exc:
        ocr.resolve_engine("auto")
    assert "no binary" in str(exc.value.details)
    assert "not installed" in str(exc.value.details)


def test_resolve_explicit_engine_reports_its_own_reason(monkeypatch):
    monkeypatch.setattr(ocr, "_PROBES", {"tesseract": lambda: "", "easyocr": lambda: "no torch"})
    assert ocr.resolve_engine("tesseract") == "tesseract"
    with pytest.raises(OCREngineError) as exc:
        ocr.resolve_engine("easyocr")
    assert exc.value.details["reason"] == "no torch"


# ── ocr_image dispatch ───────────────────────────────────────────────────────

def test_ocr_image_dispatches_to_the_selected_engine(monkeypatch):
    _all_engines_available(monkeypatch)
    captured = {}

    def fake(image, lang, psm):
        captured.update(lang=lang, psm=psm)
        return OCRResult(text="ఎరుక", engine="tesseract", mean_confidence=0.9)

    monkeypatch.setattr(ocr, "_ENGINES", {"tesseract": fake})
    result = ocr.ocr_image(Image.new("RGB", (32, 32)), engine="tesseract", lang="te", psm=7)

    assert result.text == "ఎరుక"
    assert result.engine == "tesseract"
    assert captured == {"lang": "te", "psm": 7}


def test_ocr_image_rejects_a_missing_image():
    with pytest.raises(OCREngineError):
        ocr.ocr_image(None)


# ── parser fallback ──────────────────────────────────────────────────────────

def test_text_pdf_never_touches_the_ocr_engine(tmp_path, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("OCR must not run for a page that already has text")

    monkeypatch.setattr(parser, "resolve_engine", lambda *_a, **_k: "tesseract")
    monkeypatch.setattr(parser, "ocr_image", explode)

    path = _make_text_pdf(
        tmp_path / "text.pdf", "This page already has a selectable text layer."
    )
    doc = parser.parse_document(str(path))

    assert doc.doc_metadata["ocr_pages"] == 0
    assert "selectable text layer" in doc.pages[0].raw_text


def test_scanned_pdf_is_routed_through_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "resolve_engine", lambda *_a, **_k: "tesseract")
    monkeypatch.setattr(
        parser,
        "ocr_image",
        lambda *_a, **_k: OCRResult(text="నిజ స్వరూపం", engine="tesseract", mean_confidence=0.75),
    )

    doc = parser.parse_document(str(_make_scanned_pdf(tmp_path / "scan.pdf")))

    assert doc.doc_metadata["ocr_pages"] == 1
    assert doc.doc_metadata["ocr_engine"] == "tesseract"
    page = doc.pages[0]
    assert page.raw_text == "నిజ స్వరూపం"
    assert page.normalized_text == "నిజ స్వరూపం"
    assert page.metadata["ocr_engine"] == "tesseract"
    assert page.metadata["ocr_confidence"] == 0.75


def test_ocr_false_disables_the_fallback_entirely(tmp_path, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("ocr=False must not call the engine")

    monkeypatch.setattr(parser, "ocr_image", explode)
    doc = parser.parse_document(str(_make_scanned_pdf(tmp_path / "scan2.pdf")), ocr=False)

    assert doc.doc_metadata["ocr_pages"] == 0
    assert doc.pages[0].raw_text == ""


def test_missing_engine_degrades_instead_of_breaking_ingestion(tmp_path, monkeypatch):
    def unavailable(*_a, **_k):
        raise OCREngineError("No OCR engine is available.", details={"attempted": "none"})

    monkeypatch.setattr(parser, "resolve_engine", unavailable)

    doc = parser.parse_document(str(_make_scanned_pdf(tmp_path / "scan3.pdf")))

    assert doc.doc_metadata["ocr_pages"] == 0
    assert "No OCR engine is available." in doc.doc_metadata["ocr_unavailable"]
    assert doc.pages[0].raw_text == ""  # no crash, just no text


def test_missing_engine_raises_when_ocr_is_explicitly_required(tmp_path, monkeypatch):
    def unavailable(*_a, **_k):
        raise OCREngineError("No OCR engine is available.", details={"attempted": "none"})

    monkeypatch.setattr(parser, "resolve_engine", unavailable)
    path = _make_scanned_pdf(tmp_path / "scan4.pdf")

    with pytest.raises(OCREngineError):
        parser.parse_document(str(path), ocr=True)


def test_page_rasterization_scales_with_dpi(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    low = parser._page_to_image(page, 72)
    high = parser._page_to_image(page, 144)
    doc.close()

    assert isinstance(low, Image.Image)
    assert high.width > low.width
