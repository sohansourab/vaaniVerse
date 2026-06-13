#!/usr/bin/env python3
"""
CLI entry point for batch document ingestion.

Usage:
    # Single file
    python -m scripts.ingest_documents --input data/ramayanam.pdf

    # Entire directory
    python -m scripts.ingest_documents --input data/

    # Custom chunk settings + Qdrant backend
    python -m scripts.ingest_documents --input data/ --chunk-size 256 --overlap 32 --backend qdrant

    # Dry run (parse + chunk only, no embedding or indexing)
    python -m scripts.ingest_documents --input data/ --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    VECTOR_BACKEND,
)
from core.exceptions import (
    ChunkingError,
    DocumentParseError,
    EmbeddingError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from ingestion.chunker import chunk_document
from ingestion.parser import parse_document
from retrieval.vector_store import VectorStore

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _collect_pdfs(input_path: Path) -> list[Path]:
    """
    Returns a sorted list of PDF paths from a file or directory.
    Raises SystemExit if no PDFs are found.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"Input file is not a PDF: {input_path}")
            sys.exit(1)
        return [input_path]

    if input_path.is_dir():
        pdfs = sorted(input_path.glob("**/*.pdf"))
        if not pdfs:
            logger.error(f"No PDF files found under: {input_path}")
            sys.exit(1)
        return pdfs

    logger.error(f"Input path does not exist: {input_path}")
    sys.exit(1)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ══════════════════════════════════════════════════════════════════════════════
# Per-document pipeline
# ══════════════════════════════════════════════════════════════════════════════

def ingest_file(
    pdf_path:      Path,
    store:         VectorStore | None,
    chunk_size:    int,
    chunk_overlap: int,
    dry_run:       bool,
) -> dict:
    """
    Runs the full ingestion pipeline for a single PDF.

    Returns a result dict:
        {
            "file":   str,
            "status": "success" | "skipped" | "failed",
            "pages":  int,
            "chunks": int,
            "error":  str | None,
        }
    """
    result = {
        "file":   pdf_path.name,
        "status": "failed",
        "pages":  0,
        "chunks": 0,
        "error":  None,
    }

    # ── Parse ──────────────────────────────────────────────────────────────────
    try:
        logger.info(f"  Parsing   → {pdf_path.name}")
        parsed_doc = parse_document(pdf_path, extract_images=False)
        result["pages"] = parsed_doc.total_pages
    except UnsupportedFileTypeError as exc:
        result["error"] = f"Unsupported type: {exc.message}"
        result["status"] = "skipped"
        return result
    except DocumentParseError as exc:
        result["error"] = f"Parse error: {exc.message}"
        return result

    # ── Chunk ──────────────────────────────────────────────────────────────────
    try:
        logger.info(f"  Chunking  → {pdf_path.name} ({parsed_doc.total_pages} pages)")
        chunks = chunk_document(
            parsed_doc,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        result["chunks"] = len(chunks)
    except ChunkingError as exc:
        result["error"] = f"Chunking error: {exc.message}"
        return result

    if not chunks:
        result["error"]  = "Zero chunks produced — document may be image-only."
        result["status"] = "skipped"
        return result

    logger.info(f"  Chunks    → {len(chunks)} produced")

    # ── Dry run exit ───────────────────────────────────────────────────────────
    if dry_run:
        result["status"] = "success"
        logger.info(f"  Dry run   → skipping embed + index for {pdf_path.name}")
        return result

    # ── Embed + Index ──────────────────────────────────────────────────────────
    try:
        logger.info(f"  Embedding → {pdf_path.name}")
        store.index_chunks(chunks, show_progress=True, persist=False)
    except (EmbeddingError, VectorStoreError) as exc:
        result["error"] = f"Index error: {exc.message}"
        return result

    result["status"] = "success"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "VāṇiVerse — Batch document ingestion CLI",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required = True,
        type     = Path,
        help     = "Path to a PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--chunk-size",
        type    = int,
        default = CHUNK_SIZE,
        help    = f"Max tokens per chunk (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type    = int,
        default = CHUNK_OVERLAP,
        help    = f"Overlap tokens between chunks (default: {CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--backend",
        type    = str,
        default = VECTOR_BACKEND,
        choices = ["faiss", "qdrant"],
        help    = f"Vector backend (default: {VECTOR_BACKEND}).",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Parse and chunk only — skip embedding and indexing.",
    )
    parser.add_argument(
        "--no-persist",
        action  = "store_true",
        help    = "Do not save index to disk after ingestion (FAISS only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdfs = _collect_pdfs(args.input)
    logger.info(f"Found {len(pdfs)} PDF(s) to ingest.")

    # ── Initialize vector store ────────────────────────────────────────────────
    store: VectorStore | None = None
    if not args.dry_run:
        try:
            store = VectorStore(backend=args.backend)
            # Load existing index if present so we append rather than overwrite
            try:
                store.load()
                logger.info(
                    f"Existing index loaded "
                    f"({store.total_vectors} vectors). "
                    f"New documents will be appended."
                )
            except Exception:
                logger.info("No existing index found. Starting fresh.")
        except VectorStoreError as exc:
            logger.error(f"Failed to initialize vector store: {exc.message}")
            sys.exit(1)

    # ── Process each PDF ───────────────────────────────────────────────────────
    results  : list[dict] = []
    total_start = time.time()

    for i, pdf_path in enumerate(pdfs, start=1):
        logger.info(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        file_start = time.time()

        result = ingest_file(
            pdf_path      = pdf_path,
            store         = store,
            chunk_size    = args.chunk_size,
            chunk_overlap = args.overlap,
            dry_run       = args.dry_run,
        )
        result["duration"] = _format_duration(time.time() - file_start)
        results.append(result)

        if result["status"] == "success":
            logger.info(
                f"  ✓ {pdf_path.name} — "
                f"{result['pages']} pages, "
                f"{result['chunks']} chunks "
                f"[{result['duration']}]"
            )
        elif result["status"] == "skipped":
            logger.warning(f"  ⚠ Skipped: {result['error']}")
        else:
            logger.error(f"  ✗ Failed:  {result['error']}")

    # ── Persist index ──────────────────────────────────────────────────────────
    if not args.dry_run and not args.no_persist and store:
        if args.backend == "faiss":
            try:
                store._store.save()
                logger.info(
                    f"FAISS index saved. "
                    f"Total vectors: {store.total_vectors}"
                )
            except VectorStoreError as exc:
                logger.error(f"Failed to save index: {exc.message}")
        else:
            logger.info("Qdrant backend — index persisted automatically.")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_duration = _format_duration(time.time() - total_start)
    succeeded = [r for r in results if r["status"] == "success"]
    skipped   = [r for r in results if r["status"] == "skipped"]
    failed    = [r for r in results if r["status"] == "failed"]

    print("\n" + "═" * 55)
    print(f"  VāṇiVerse Ingestion Summary")
    print("═" * 55)
    print(f"  Total files  : {len(pdfs)}")
    print(f"  Succeeded    : {len(succeeded)}")
    print(f"  Skipped      : {len(skipped)}")
    print(f"  Failed       : {len(failed)}")
    print(f"  Total pages  : {sum(r['pages']  for r in succeeded)}")
    print(f"  Total chunks : {sum(r['chunks'] for r in succeeded)}")
    if not args.dry_run and store:
        print(f"  Index size   : {store.total_vectors} vectors")
    print(f"  Duration     : {total_duration}")
    print("═" * 55)

    if failed:
        print("\n  Failed files:")
        for r in failed:
            print(f"    ✗ {r['file']}: {r['error']}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()