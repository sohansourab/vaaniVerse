import io
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from core import config
from core.exceptions import DocumentParseError, OCREngineError, UnsupportedFileTypeError
from ingestion.normalizer import normalize
from ingestion.ocr import OCRResult, ocr_image, resolve_engine

SUPPORTED_EXTENSIONS = {".pdf"}


@dataclass
class ParsedPage:
    page_number: int          # 1-indexed
    raw_text: str             # text as extracted by PyMuPDF
    normalized_text: str      # Unicode-normalized, transliteration-applied
    images: list[dict]        # list of {"bbox": Rect, "image": PIL.Image}
    metadata: dict            # source file, page count, etc.


@dataclass
class ParsedDocument:
    source_path: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)
    doc_metadata: dict       = field(default_factory=dict)


def _extract_images_from_page(page: fitz.Page, doc: fitz.Document) -> list[dict]:
    """
    Extracts embedded images from a PyMuPDF page.
    Returns a list of dicts containing the bounding box and PIL Image object.
    Skips images smaller than 50x50px (decorative rules / borders).
    """
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            width, height = pil_image.size
            if width < 50 or height < 50:
                continue

            # Locate bounding box of this image on the page
            bbox = None
            for item in page.get_image_info(full=True):
                if item.get("xref") == xref:
                    bbox = item.get("bbox")
                    break

            images.append({
                "xref":   xref,
                "bbox":   bbox,
                "width":  width,
                "height": height,
                "image":  pil_image,
            })
        except Exception:
            # Non-fatal: skip corrupt or unreadable image xrefs
            continue

    return images


def _extract_text_from_page(page: fitz.Page) -> str:
    """
    Extracts text using PyMuPDF's 'text' mode.
    Falls back to 'blocks' mode if standard extraction yields empty string
    (common in scanned PDFs with embedded OCR layers).
    """
    text = page.get_text("text").strip()

    if not text:
        blocks = page.get_text("blocks")
        text = "\n".join(
            b[4].strip() for b in blocks if isinstance(b[4], str) and b[4].strip()
        )

    return text


def _page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    """
    Rasterize a page to a PIL image for OCR.

    PyMuPDF hands back raw pixel samples, so the array is wrapped rather than
    encoded — no temporary file is ever written.
    """
    zoom   = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def parse_document(
    file_path: str | Path,
    extract_images: bool = True,
    ocr: bool | None = None,
) -> ParsedDocument:
    """
    Entry point for the ingestion parser.

    Args:
        file_path:       Absolute or relative path to a PDF file.
        extract_images:  Whether to extract embedded images per page.
        ocr:             OCR policy for pages with no usable text layer.
                           None (default) — follow ``config.OCR_ENABLED``; if no
                             engine is installed, degrade instead of failing.
                           True  — require OCR; raise when no engine is usable.
                           False — never OCR.

    Returns:
        ParsedDocument containing all pages with text and image data.

    Raises:
        UnsupportedFileTypeError: If the file is not a supported type.
        DocumentParseError:       If PyMuPDF cannot open or read the file.
        OCREngineError:           If ``ocr=True`` and no engine is usable.
    """
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{path.suffix}' is not supported.",
            details={"supported": list(SUPPORTED_EXTENSIONS), "received": path.suffix},
        )

    if not path.exists():
        raise DocumentParseError(
            f"File not found: {path}",
            details={"path": str(path)},
        )

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise DocumentParseError(
            f"PyMuPDF failed to open document: {path.name}",
            details={"error": str(exc), "path": str(path)},
        )

    doc_metadata = {
        "source":      path.name,
        "total_pages": doc.page_count,
        "pdf_metadata": doc.metadata,   # author, title, subject, etc.
    }

    parsed_doc = ParsedDocument(
        source_path  = str(path),
        total_pages  = doc.page_count,
        doc_metadata = doc_metadata,
    )

    # Resolve OCR once per document — probing an engine imports torch, so this
    # must not happen inside the page loop.
    ocr_requested   = config.OCR_ENABLED if ocr is None else ocr
    ocr_engine: str | None    = None
    ocr_unavailable: str | None = None
    if ocr_requested:
        try:
            ocr_engine = resolve_engine(config.OCR_ENGINE)
        except OCREngineError as exc:
            if ocr is True:
                doc.close()
                raise
            # Default policy: a missing engine must not break ingestion of
            # text PDFs. Record why, then carry on with the text layer alone.
            ocr_requested   = False
            ocr_unavailable = f"{exc.message} {exc.details}"

    ocr_pages = 0

    for page_index in range(doc.page_count):
        try:
            page = doc[page_index]
        except Exception as exc:
            raise DocumentParseError(
                f"Failed to load page {page_index + 1} from {path.name}.",
                details={"page": page_index + 1, "error": str(exc)},
            )

        raw_text        = _extract_text_from_page(page)
        ocr_result: OCRResult | None = None

        # A page with no text layer is a scan. A page with a *vestigial* one — a
        # stray page number — is also a scan, which is why this is a character
        # threshold and not a truthiness check.
        if ocr_requested and len(raw_text) < config.OCR_MIN_CHARS:
            ocr_result = ocr_image(
                _page_to_image(page, config.OCR_DPI),
                engine=ocr_engine,
                lang=config.OCR_LANG,
            )
            if ocr_result.text.strip():
                raw_text = ocr_result.text.strip()
                ocr_pages += 1

        normalized_text = normalize(raw_text, transliterate=True)
        images          = _extract_images_from_page(page, doc) if extract_images else []

        parsed_doc.pages.append(
            ParsedPage(
                page_number     = page_index + 1,
                raw_text        = raw_text,
                normalized_text = normalized_text,
                images          = images,
                metadata        = {
                    "source":         path.name,
                    "page_number":    page_index + 1,
                    "total_pages":    doc.page_count,
                    "has_images":     len(images) > 0,
                    "ocr_engine":     ocr_result.engine if ocr_result else None,
                    "ocr_confidence": ocr_result.mean_confidence if ocr_result else None,
                },
            )
        )

    doc_metadata["ocr_engine"]      = ocr_engine
    doc_metadata["ocr_pages"]       = ocr_pages
    doc_metadata["ocr_unavailable"] = ocr_unavailable

    doc.close()
    return parsed_doc