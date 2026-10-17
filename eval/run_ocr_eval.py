#!/usr/bin/env python3
"""
Measure Telugu OCR quality per engine against the fixtures in data/ocr_bench/.

Reports character error rate (CER), word error rate (WER), exact-match rate and
latency, per corpus and per engine, so an engine choice is a measurement rather
than a preference.

The two fixture corpora are deliberately different jobs:

  sarvam      multi-line text blocks, expert-verified transcriptions. Tests
              recognition on already-segmented text.
  wikisource  full page scans of real Telugu books. Tests the whole path the
              app actually needs — page in, text out.

CER/WER are computed on NFC-normalized text with runs of whitespace collapsed,
so layout differences do not masquerade as recognition errors.

Usage:
    python eval/run_ocr_eval.py
    python eval/run_ocr_eval.py --engines tesseract --limit 10
    python eval/run_ocr_eval.py --device cuda
"""

import argparse
import json
import os
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from core.config import BASE_DIR  # noqa: E402
from ingestion.ocr import available_engines, ocr_image, probe_engines  # noqa: E402

ROOT = BASE_DIR / "data" / "ocr_bench"

# Page-segmentation modes: a text block is not a page.
DEFAULT_PSM = {"sarvam": 6, "wikisource": 3}


# ── metrics ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """NFC + collapse whitespace, so layout is not scored as recognition error."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a, b) -> int:
    """Edit distance over any two indexable sequences (characters, or words)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        item_a = a[i - 1]
        current = [i]
        for j in range(1, len(b) + 1):
            current.append(
                min(
                    previous[j] + 1,                        # deletion
                    current[j - 1] + 1,                     # insertion
                    previous[j - 1] + (item_a != b[j - 1]),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = _normalize(reference)
    return edit_distance(reference, _normalize(hypothesis)) / max(1, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Token-level error rate, on the same normalized text as CER."""
    reference_words = _normalize(reference).split()
    hypothesis_words = _normalize(hypothesis).split()
    return edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))


# ── fixtures ───────────────────────────────────────────────────────────────────

def load_samples(group: str | None = None) -> list[dict]:
    """Pair every .jpg with its .txt transcript, skipping empty transcripts."""
    samples = []
    for group_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if group is not None and group_dir.name != group:
            continue
        for image_path in sorted(group_dir.glob("*.jpg")):
            transcript = image_path.with_suffix(".txt")
            if not transcript.exists():
                continue
            reference = transcript.read_text(encoding="utf-8").strip()
            if not reference:
                continue
            samples.append(
                {"group": group_dir.name, "image": image_path, "reference": reference}
            )
    return samples


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(engine: str, samples: list[dict], psm: dict[str, int], warmup: bool = True) -> list[dict]:
    rows = []

    for index, sample in enumerate(samples):
        with Image.open(sample["image"]) as opened:
            image = opened.convert("RGB")

            # The first call pays for model loading; keep it out of the timings.
            if warmup and index == 0:
                ocr_image(image, engine=engine, psm=psm[sample["group"]])
                continue

            started = time.perf_counter()
            result = ocr_image(image, engine=engine, psm=psm[sample["group"]])
            elapsed_ms = (time.perf_counter() - started) * 1000

        rows.append(
            {
                "group": sample["group"],
                "image": sample["image"].name,
                "cer": character_error_rate(sample["reference"], result.text),
                "wer": word_error_rate(sample["reference"], result.text),
                "exact": _normalize(sample["reference"]) == _normalize(result.text),
                "confidence": result.mean_confidence,
                "latency_ms": elapsed_ms,
                "hypothesis": result.text,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    cers = [r["cer"] for r in rows]
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    latencies = [r["latency_ms"] for r in rows]
    return {
        "n": len(rows),
        "cer_mean": statistics.mean(cers),
        "cer_median": statistics.median(cers),
        "wer_mean": statistics.mean(r["wer"] for r in rows),
        "exact_match": sum(r["exact"] for r in rows) / len(rows),
        "confidence_mean": statistics.mean(confidences) if confidences else None,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", default="",
                        help="comma-separated engines; default = all available")
    parser.add_argument("--group", default=None, help="sarvam | wikisource")
    parser.add_argument("--limit", type=int, default=0, help="cap samples per group")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--psm-sarvam", type=int, default=DEFAULT_PSM["sarvam"],
                        help="Tesseract page-segmentation mode for text blocks")
    parser.add_argument("--psm-wikisource", type=int, default=DEFAULT_PSM["wikisource"],
                        help="Tesseract page-segmentation mode for full pages")
    parser.add_argument("--out", default=str(BASE_DIR / "eval" / "ocr_results.json"))
    args = parser.parse_args()

    os.environ["OCR_DEVICE"] = args.device

    psm = {"sarvam": args.psm_sarvam, "wikisource": args.psm_wikisource}

    engines = (
        [e.strip() for e in args.engines.split(",") if e.strip()]
        if args.engines
        else available_engines()
    )
    if not engines:
        print("No OCR engine available. Reasons:")
        for engine, reason in probe_engines().items():
            print(f"  {engine}: {reason}")
        return 1

    samples = load_samples(args.group)
    if args.limit:
        limited: list[dict] = []
        seen: dict[str, int] = {}
        for sample in samples:
            count = seen.get(sample["group"], 0)
            if count < args.limit:
                limited.append(sample)
                seen[sample["group"]] = count + 1
        samples = limited

    if not samples:
        print(f"No fixtures under {ROOT}. Run: python -m scripts.fetch_ocr_bench")
        return 1

    groups = sorted({s["group"] for s in samples})
    print(f"Fixtures: {len(samples)} image(s) across {', '.join(groups)}")
    print(f"Engines : {', '.join(engines)} (device={args.device})\n")

    all_results: dict[str, dict] = {}

    for engine in engines:
        print(f"── {engine} " + "─" * max(0, 60 - len(engine)))
        rows = evaluate(engine, samples, psm)
        per_group = {group: summarize([r for r in rows if r["group"] == group]) for group in groups}

        for group in groups:
            summary = per_group.get(group)
            if not summary:
                continue
            print(
                f"  {group:<11} n={summary['n']:<3} "
                f"CER={summary['cer_mean']:.3f}  WER={summary['wer_mean']:.3f}  "
                f"exact={summary['exact_match']:.3f}  "
                f"conf={summary['confidence_mean'] if summary['confidence_mean'] is not None else float('nan'):.2f}  "
                f"p50={summary['latency_p50_ms']:.0f}ms"
            )

        overall = summarize(rows)
        print(
            f"  {'ALL':<11} n={overall['n']:<3} CER={overall['cer_mean']:.3f}  "
            f"WER={overall['wer_mean']:.3f}  exact={overall['exact_match']:.3f}\n"
        )
        all_results[engine] = {"overall": overall, "by_group": per_group, "rows": rows}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {"engines": engines, "device": args.device,
                           "samples": len(samples), "groups": groups},
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
