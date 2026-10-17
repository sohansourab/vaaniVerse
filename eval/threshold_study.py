#!/usr/bin/env python3
"""
Threshold study for the abstention gate.

docs/EVALUATION.md documents the core retrieval weakness: with dense cosine
similarity, in-corpus and out-of-domain score distributions **overlap**, so no
fixed cutoff can both answer genuine questions and refuse nonsense. At 0.45 the
gate rejects 100% of out-of-domain queries but also abstains on ~23% of real
ones. A single number cannot separate two overlapping distributions.

This measures whether reranking actually fixes that. For each gold question it
records the **best score** retrieval produced (which is what decides abstention:
a query is answered if any result clears the threshold), then sweeps the
threshold to find the trade-off between:

    in-corpus pass rate   — real questions actually answered
    out-of-domain reject  — nonsense questions actually refused

and reports the area under the ROC curve for using that score as a
genuine-vs-nonsense discriminator. AUC 0.5 means the score carries no signal
(overlapping distributions); 1.0 means perfect separation.

Run it once per configuration against a live API and compare the two files:

    # dense-only
    python eval/threshold_study.py --out eval/gate_dense.json
    # reranked
    RERANK_ENABLED=true python eval/threshold_study.py --out eval/gate_reranked.json

Usage:
    python eval/threshold_study.py --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
GOLD = Path(__file__).resolve().parent / "gold_qa.jsonl"

# A query can only be answered if retrieval offers something inside top-k, so
# sweeping the threshold over candidates beyond top-k would be meaningless.
TOP_K = 5


def load_gold(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def probe_query(client: httpx.Client, api: str, query: str) -> tuple[float, bool]:
    """
    Highest score retrieval produced (with the threshold disabled), plus whether
    the server reranked. One request per query — reranking is the expensive part.
    """
    response = client.get(
        f"{api}/search",
        params={"q": query, "top_k": TOP_K, "score_threshold": 0.0},
    )
    response.raise_for_status()
    payload = response.json()
    scores = [result["score"] for result in payload.get("results", [])]
    return (max(scores) if scores else 0.0), bool(payload.get("reranked", False))


def auc(positive: list[float], negative: list[float]) -> float:
    """
    P(score of a genuine query > score of an out-of-domain query), rank-based.

    Ties count a half. 0.5 means the score cannot tell the two apart at all,
    which is exactly the overlapping-distributions problem this study exists to
    quantify.
    """
    if not positive or not negative:
        return 0.0
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def sweep(genuine: list[float], out_of_domain: list[float],
          grid: list[float]) -> list[dict]:
    rows = []
    for threshold in grid:
        passed = sum(1 for score in genuine if score >= threshold) / len(genuine)
        rejected = sum(1 for score in out_of_domain if score < threshold) / len(out_of_domain)
        rows.append({
            "threshold": round(threshold, 3),
            "in_corpus_pass_rate": round(passed, 4),
            "out_of_domain_reject_rate": round(rejected, 4),
            "balanced_accuracy": round((passed + rejected) / 2, 4),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=str(REPO / "eval" / "gate_study.json"))
    parser.add_argument("--grid-step", type=float, default=0.02)
    args = parser.parse_args()

    gold = load_gold(GOLD)
    client = httpx.Client(timeout=120.0)

    reranked: bool | None = None
    per_query = []

    for item in gold:
        score, reranked_now = probe_query(client, args.api, item["query"])
        if reranked is None:
            reranked = reranked_now

        per_query.append({
            "id": item["id"],
            "language": item["language"],
            "in_corpus": item["in_corpus"],
            "best_score": round(score, 4),
        })
        print(f"  {item['id']:>4}  in_corpus={str(item['in_corpus']):5}  best={score:.4f}", flush=True)

    genuine = [row["best_score"] for row in per_query if row["in_corpus"]]
    out_of_domain = [row["best_score"] for row in per_query if not row["in_corpus"]]

    steps = int(round(1.0 / args.grid_step))
    grid = [round(i * args.grid_step, 4) for i in range(steps + 1)]
    rows = sweep(genuine, out_of_domain, grid)

    best = max(rows, key=lambda row: row["balanced_accuracy"])
    discrimination = auc(genuine, out_of_domain)

    def spread(values: list[float]) -> dict:
        return {
            "n": len(values),
            "min": round(min(values), 4) if values else 0.0,
            "median": round(sorted(values)[len(values) // 2], 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
        }

    genuine_stats = spread(genuine)
    ood_stats = spread(out_of_domain)
    overlap = (
        genuine_stats["min"] <= ood_stats["max"]
        and ood_stats["min"] <= genuine_stats["max"]
    )

    print(f"\nreranked           : {reranked}")
    print(f"in-corpus scores   : {genuine_stats}")
    print(f"out-of-domain      : {ood_stats}")
    print(f"ranges overlap     : {overlap}")
    print(f"AUC (separability) : {discrimination:.4f}")
    print(f"best threshold     : {best}")

    output = {
        "reranked": reranked,
        "top_k": TOP_K,
        "n_queries": len(per_query),
        "in_corpus_scores": genuine_stats,
        "out_of_domain_scores": ood_stats,
        "ranges_overlap": overlap,
        "auc": round(discrimination, 4),
        "best_threshold": best,
        "sweep": rows,
        "per_query": per_query,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
