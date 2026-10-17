#!/usr/bin/env python3
"""
Stage Tesseract language data into data/tessdata/.

Why this exists: the Windows Tesseract installer ships only `eng` and `osd`,
writes into Program Files (not user-writable), and its silent install offers no
Telugu model. So the system directory cannot simply be extended.

This copies whatever the system install has into a repo-local, gitignored
directory, then downloads the missing models from the official tessdata_fast
repository. `ingestion/ocr.py` prefers that directory when it holds
`tel.traineddata`, passing it to Tesseract with --tessdata-dir.

Usage:
    python -m scripts.fetch_ocr_tessdata                 # tel, plus eng/osd
    python -m scripts.fetch_ocr_tessdata --langs tel,eng
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from core.config import BASE_DIR
from ingestion.ocr import find_tesseract_binary

TESSDATA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{lang}.traineddata"
TARGET       = BASE_DIR / "data" / "tessdata"


def _system_tessdata() -> Path | None:
    binary = find_tesseract_binary()
    if binary is None:
        return None
    directory = Path(binary).parent / "tessdata"
    return directory if directory.is_dir() else None


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langs",
        default="tel",
        help="comma-separated tessdata codes to guarantee (default: tel)",
    )
    args = parser.parse_args()

    wanted = [code.strip() for code in args.langs.split(",") if code.strip()]
    TARGET.mkdir(parents=True, exist_ok=True)

    # 1. Seed from the system install so the local directory is self-contained:
    #    Tesseract looks in exactly one data dir, so eng/osd must come along.
    source = _system_tessdata()
    copied = 0
    if source is not None:
        for model in source.glob("*.traineddata"):
            destination = TARGET / model.name
            if not destination.exists():
                shutil.copy2(model, destination)
                copied += 1
    print(f"[seed] copied {copied} model(s) from {source or 'nothing'}")

    # 2. Fetch whatever is still missing.
    failures: list[str] = []
    for lang in wanted:
        destination = TARGET / f"{lang}.traineddata"
        if destination.exists():
            print(f"[have] {lang}  ({_mb(destination)})")
            continue

        url = TESSDATA_URL.format(lang=lang)
        print(f"[get ] {lang}  <- {url}")
        try:
            urllib.request.urlretrieve(url, destination)
        except Exception as exc:
            failures.append(lang)
            print(f"[fail] {lang}: {exc}")
            destination.unlink(missing_ok=True)
            continue
        print(f"[ok  ] {lang}  ({_mb(destination)})")

    if failures:
        print(f"\nCould not fetch: {', '.join(failures)}")
        return 1

    print(f"\nTessdata staged at {TARGET}")
    print("OCR will pick it up automatically — no environment variable needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
