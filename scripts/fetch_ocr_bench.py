#!/usr/bin/env python3
"""
Download OCR evaluation fixtures into data/ocr_bench/ (gitignored).

Two sources, chosen because their ground truth is actually trustworthy:

  sarvam      sarvamai/indic-ocr-bench — Telugu text blocks whose transcriptions
              were reviewed twice by human language experts. Apache-2.0.
  wikisource  AlbertoChestnut/telugu-ocr — full page scans of Telugu books with
              human transcriptions, from Telugu Wikisource. CC-BY-SA-4.0.

Both are downloaded on demand and never committed. The Wikisource corpus is
share-alike and ~11 GB in full, so only a small page slice is pulled.

Deliberately NOT used: meharuhanzz/OCR-Bench1000-Telugu. It is MIT and tidy, but
its own card states the annotations were reviewed by someone who cannot read
Telugu, so it cannot support a benchmark claim.

Usage:
    python -m scripts.fetch_ocr_bench
    python -m scripts.fetch_ocr_bench --sarvam-limit 25 --wikisource-pages 6
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from core.config import BASE_DIR

ROOT = BASE_DIR / "data" / "ocr_bench"

SARVAM_DATASET = "sarvamai/indic-ocr-bench"
SARVAM_CONFIG  = "default"
SARVAM_SPLIT   = "small_representative"
SARVAM_LANG    = "Telugu"

WIKI_DATASET = "AlbertoChestnut/telugu-ocr"
WIKI_BOOK    = ".Kalidasa-Charitra_by_chilakamarthi_lakshminarasimham"

_UA = {"User-Agent": "vaaniverse-ocr-fixture-fetch/1.0"}


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    destination.write_bytes(data)
    return len(data)


# ── sarvam: Telugu text blocks ─────────────────────────────────────────────────

def fetch_sarvam(limit: int) -> int:
    """Page through the rows API, keep the Telugu samples, save image + text."""
    destination = ROOT / "sarvam"
    destination.mkdir(parents=True, exist_ok=True)

    base = (
        "https://datasets-server.huggingface.co/rows?"
        + urllib.parse.urlencode(
            {
                "dataset": SARVAM_DATASET,
                "config": SARVAM_CONFIG,
                "split": SARVAM_SPLIT,
            }
        )
    )

    saved = 0
    offset = 0
    total = None

    while saved < limit and (total is None or offset < total):
        page = _get_json(f"{base}&offset={offset}&length=100")
        total = page.get("num_rows_total", 0)
        rows = page.get("rows", [])
        if not rows:
            break

        for entry in rows:
            if saved >= limit:
                break
            row = entry["row"]
            if row.get("language") != SARVAM_LANG:
                continue

            name = row["image_name"]
            text = row.get("gt") or ""
            image_url = (row.get("image") or {}).get("src")
            if not image_url or not text.strip():
                continue

            # The URL is signed and short-lived, so fetch it immediately.
            try:
                size = _download(image_url, destination / f"{name}.jpg")
            except Exception as exc:
                print(f"  [fail] {name}: {exc}")
                continue

            (destination / f"{name}.txt").write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [ok  ] {name}  ({size / 1024:.0f} KB)")

        offset += len(rows)

    print(f"[sarvam] {saved} Telugu sample(s) in {destination}")
    return saved


# ── wikisource: real page scans ────────────────────────────────────────────────

def fetch_wikisource(pages: int) -> int:
    """Pull a few page scans (with transcriptions) from one Telugu book."""
    destination = ROOT / "wikisource"
    destination.mkdir(parents=True, exist_ok=True)

    tree = _get_json(
        f"https://huggingface.co/api/datasets/{WIKI_DATASET}/tree/main/"
        + urllib.parse.quote(WIKI_BOOK)
    )
    names = {entry["path"].split("/")[-1] for entry in tree}

    # A page is usable only when both the scan and its transcription exist.
    stems = sorted(
        name[:-4] for name in names
        if name.endswith(".jpg") and f"{name[:-4]}.txt" in names
    )

    saved = 0
    for stem in stems[:pages]:
        for suffix in (".jpg", ".txt"):
            url = (
                f"https://huggingface.co/datasets/{WIKI_DATASET}/resolve/main/"
                + urllib.parse.quote(f"{WIKI_BOOK}/{stem}{suffix}")
            )
            try:
                size = _download(url, destination / f"{stem}{suffix}")
            except Exception as exc:
                print(f"  [fail] {stem}{suffix}: {exc}")
                break
        else:
            saved += 1
            print(f"  [ok  ] {stem}  ({size / 1024:.0f} KB)")

    print(f"[wikisource] {saved} page(s) from {WIKI_BOOK} in {destination}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sarvam-limit", type=int, default=51,
                        help="Telugu text blocks to fetch (default: all 51)")
    parser.add_argument("--wikisource-pages", type=int, default=6,
                        help="book pages to fetch (default: 6)")
    args = parser.parse_args()

    sarvam = fetch_sarvam(args.sarvam_limit)
    wiki   = fetch_wikisource(args.wikisource_pages)

    if not sarvam and not wiki:
        print("\nNothing was downloaded — check connectivity.")
        return 1

    print(f"\nFixtures under {ROOT} (gitignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
