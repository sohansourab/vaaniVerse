#!/usr/bin/env python3
"""
Bootstrap a demo index from a public sample document.

A fresh clone has no index, so `/query` answers "the corpus does not contain
this" to everything and the app reads as broken. This downloads one small,
freely-licensed Telugu PDF, ingests it through the normal pipeline, and leaves a
working index behind.

The PDF is deliberately NOT committed (it is large and third-party), which is
why this script exists instead of a checked-in file.

Usage:
    python -m scripts.load_sample                 # download + ingest
    python -m scripts.load_sample --no-ingest     # download only
    python -m scripts.load_sample --force         # re-download even if present
    python -m scripts.load_sample --dry-run       # download, parse+chunk, no embed
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import httpx

from core.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, VECTOR_BACKEND

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("load_sample")

# "Who Am I?" (నేను ఎవరు?) — a short Telugu self-inquiry text on archive.org.
SAMPLE_URL  = "https://archive.org/download/who-am-i-telugu/Who_Am_I_Telugu.pdf"
SAMPLE_NAME = "Who_Am_I_Telugu.pdf"
CHUNK_BYTES = 1 << 16


class SampleDownloadError(RuntimeError):
    """The sample could not be downloaded or was not a usable PDF."""


def _is_pdf(path: Path) -> bool:
    """Cheap magic-byte check — an HTML error page must not pass as a PDF."""
    try:
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sample(
    url:      str       = SAMPLE_URL,
    dest_dir: Path      = DATA_DIR,
    filename: str       = SAMPLE_NAME,
    force:    bool      = False,
    timeout:  float     = 120.0,
) -> Path:
    """
    Download the sample PDF to ``dest_dir`` and return its path.

    Idempotent: an existing valid PDF is reused unless ``force`` is set. Writes
    to a ``.part`` file and renames on success, so an interrupted download can
    never be mistaken for a complete document.

    Raises:
        SampleDownloadError: on a network/HTTP failure, a short read, or a body
            that is not actually a PDF.
    """
    dest = Path(dest_dir) / filename
    if dest.exists() and not force and _is_pdf(dest):
        logger.info(f"Sample already present: {dest} (use --force to re-download)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    logger.info(f"Downloading {url}")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total   = int(resp.headers.get("content-length") or 0)
            written = 0
            with part.open("wb") as fh:
                for block in resp.iter_bytes(CHUNK_BYTES):
                    fh.write(block)
                    written += len(block)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise SampleDownloadError(f"Download failed: {str(exc)[:200]}") from exc

    if written == 0 or (total and written != total):
        part.unlink(missing_ok=True)
        raise SampleDownloadError(
            f"Incomplete download ({written} of {total or '?'} bytes)."
        )
    if not _is_pdf(part):
        part.unlink(missing_ok=True)
        raise SampleDownloadError("Downloaded file is not a PDF (bad URL or error page).")

    part.replace(dest)
    logger.info(f"Saved {dest} ({written / 1024:.0f} KB, sha256 {_sha256(dest)[:12]}…)")
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Download and ingest a public sample document so a fresh "
                      "clone has a working index.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    parser.add_argument("--url", default=SAMPLE_URL, help="Override the sample URL.")
    parser.add_argument("--filename", default=SAMPLE_NAME, help="Local file name to save as.")
    parser.add_argument("--dest-dir", type=Path, default=DATA_DIR,
                        help=f"Where to save the PDF (default: {DATA_DIR}).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if a valid copy already exists.")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Only download; do not touch the index.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--backend", default=VECTOR_BACKEND, choices=["faiss", "qdrant"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and chunk only — no embedding or indexing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pdf_path = download_sample(
            url      = args.url,
            dest_dir = args.dest_dir,
            filename = args.filename,
            force    = args.force,
        )
    except SampleDownloadError as exc:
        logger.error(str(exc))
        return 1

    if args.no_ingest:
        print(f"\nDownloaded: {pdf_path}")
        return 0

    # Imported here so `--no-ingest` works without the embedding stack installed.
    from retrieval.vector_store import VectorStore
    from scripts.ingest_documents import ingest_file

    store = VectorStore(backend=args.backend)
    try:
        store.load()
        before = store.total_vectors
        logger.info(f"Existing index loaded ({before} vectors).")
    except Exception:
        before = 0
        logger.info("No existing index found. Starting fresh.")

    result = ingest_file(
        pdf_path      = pdf_path,
        store         = store,
        chunk_size    = args.chunk_size,
        chunk_overlap = args.overlap,
        dry_run       = args.dry_run,
        replace       = True,
    )

    if result["status"] != "success":
        logger.error(f"Ingestion failed: {result['error']}")
        return 1

    if args.dry_run:
        print(f"\nDry run OK: {result['pages']} pages → {result['chunks']} chunks "
              f"(nothing indexed).")
        return 0

    if args.backend == "faiss":
        try:
            store._store.save()
        except Exception as exc:
            logger.error(f"Failed to save the FAISS index: {exc}")
            return 1

    print("\n" + "=" * 55)
    print("  Sample index ready")
    print("=" * 55)
    print(f"  Document    : {pdf_path.name}")
    print(f"  Pages       : {result['pages']}")
    print(f"  Chunks      : {result['chunks']}")
    print(f"  Replaced    : {result.get('removed', 0)} old chunks")
    print(f"  Index now   : {store.total_vectors} vectors (was {before})")
    print(f"  Backend     : {args.backend}")
    print("=" * 55)
    print("\nStart the API (uvicorn api.main:app) and ask, e.g.:")
    print("  నేను ఎవరు?  ·  What is my true nature?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
