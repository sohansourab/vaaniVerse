#!/usr/bin/env python3
"""
Calibrate the retrieval score threshold, per query script.

The shipped 0.45 cutoff was measured on a corpus that is mostly native Telugu,
and applied to every script. But native-script queries embed closer to the
(native-script) passages than Roman or English queries do at equal relevance, so
one number cannot be right for all three: it either lets out-of-domain Telugu
through, or (as measured) stops genuine English and Romanized questions at the
gate.

This script measures the score distribution of gold in-corpus vs out-of-domain
questions, per script, and picks the threshold that maximizes in-corpus passage
subject to zero out-of-domain passage — the most conservative useful cutoff.

Usage:
    # against a running API with the sample book ingested
    python -m scripts.calibrate_threshold --gold eval/gold_qa.jsonl

    # print values to paste into the environment / .env
    python -m scripts.calibrate_threshold --emit-env
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "eval" / "gold_qa.jsonl"

# gold `language` → the script detect_script() reports for that query.
_LANGUAGE_TO_SCRIPT = {"telugu": "telugu", "roman": "roman", "english": "roman"}

_CANDIDATES = [round(t / 100, 2) for t in range(0, 91)]


def load_gold(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def top1_scores(client: httpx.Client, api: str, gold: list[dict], top_k: int) -> list[dict]:
    """Fetch the top-1 score for every query with the threshold disabled."""
    rows = []
    for item in gold:
        resp = client.get(
            f"{api}/search",
            params={"q": item["query"], "top_k": top_k, "score_threshold": 0.0},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        rows.append({
            "id": item["id"],
            "in_corpus": item["in_corpus"],
            "script": _LANGUAGE_TO_SCRIPT.get(item["language"], "roman"),
            "top1": results[0]["score"] if results else 0.0,
        })
    return rows


def best_threshold(rows: list[dict]) -> dict:
    """
    Pick the cutoff maximizing in-corpus passage with zero out-of-domain passage.

    Returns the chosen threshold plus the rates at that cutoff, so the trade-off
    is visible rather than hidden behind a single number.
    """
    in_rows = [r for r in rows if r["in_corpus"]]
    ood_rows = [r for r in rows if not r["in_corpus"]]
    if not in_rows:
        return {"threshold": 0.45, "in_pass": 0.0, "ood_pass": 0.0, "n_in": 0, "n_ood": len(ood_rows)}

    best = None
    for t in _CANDIDATES:
        in_pass = sum(1 for r in in_rows if r["top1"] >= t) / len(in_rows)
        ood_pass = (sum(1 for r in ood_rows if r["top1"] >= t) / len(ood_rows)) if ood_rows else 0.0
        # Prefer zero OOD passage; among those, the highest in-corpus passage.
        # Ties broken by the *highest* threshold (more conservative).
        key = (ood_pass == 0.0, in_pass, t)
        if best is None or key > best[0]:
            best = (key, t, in_pass, ood_pass)

    _, t, in_pass, ood_pass = best
    return {
        "threshold": t,
        "in_pass": round(in_pass, 4),
        "ood_pass": round(ood_pass, 4),
        "n_in": len(in_rows),
        "n_ood": len(ood_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the retrieval score threshold.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--emit-env", action="store_true", help="Print shell-ready env assignments.")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    api = args.api_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{api}/health").json()
        if not health.get("total_vectors"):
            print("ERROR: index is empty — ingest a document first.", file=sys.stderr)
            return 2
        rows = top1_scores(client, api, gold, args.top_k)

    overall = best_threshold(rows)
    by_script = {
        script: best_threshold([r for r in rows if r["script"] == script])
        for script in ("telugu", "roman")
    }

    report = {
        "n_queries": len(rows),
        "overall": overall,
        "by_script": by_script,
        "note": ("threshold maximizes in-corpus passage subject to zero "
                 "out-of-domain passage; re-run after any corpus or embedder change"),
    }
    print(json.dumps(report, indent=2))

    if args.emit_env:
        print("\n# paste into your environment / .env")
        print(f"SCORE_THRESHOLD_TELUGU={by_script['telugu']['threshold']}")
        print(f"SCORE_THRESHOLD_ROMAN={by_script['roman']['threshold']}")
        print(f"SCORE_THRESHOLD_MIXED={overall['threshold']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
