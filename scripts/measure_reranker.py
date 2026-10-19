#!/usr/bin/env python3
"""
Measure the cross-encoder reranker end to end — the run this project kept losing.

The README admits the before/after reranker evaluation "was lost twice to a
Windows uvicorn accept-loop fault and cut for time". That is a reason to move the
measurement off the HTTP path, which is what this does: it loads the persisted
index locally, ranks each gold query with dense retrieval, then re-ranks the same
candidates with the cross-encoder, and reports recall@k and latency for both.

Because both stages run in-process, there is no server to drop and no connection
to lose. Nothing here touches the API.

Usage:
    python -m scripts.measure_reranker                 # full run (loads 2 models)
    python -m scripts.measure_reranker --limit 10      # smoke test
    python -m scripts.measure_reranker --out eval/reranker.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "eval" / "gold_qa.jsonl"

# Must match the retrieval-time candidate pool, or "after" is measured against a
# different proposition than the one that ships.
from core.config import RERANK_CANDIDATES  # noqa: E402
from ingestion.normalizer import normalize  # noqa: E402
from retrieval.embeddings import embed_query  # noqa: E402
from retrieval.reranker import score_pairs  # noqa: E402
from retrieval.vector_store import FAISSVectorStore  # noqa: E402


def load_gold(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recall_at(pages: list, expected: set, k: int) -> float:
    return 1.0 if any(p in expected for p in pages[:k]) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure dense vs reranked retrieval.")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "reranker.json")
    args = parser.parse_args()

    gold = [g for g in load_gold(args.gold) if g["in_corpus"]]
    if args.limit:
        gold = gold[: args.limit]
    if not gold:
        print("ERROR: no in-corpus gold questions.", file=sys.stderr)
        return 2

    store = FAISSVectorStore()
    try:
        store.load()
    except Exception as exc:
        print(f"ERROR: could not load the index — ingest first ({exc}).", file=sys.stderr)
        return 2

    dense_recall: list[float] = []
    rerank_recall: list[float] = []
    dense_ms: list[float] = []
    rerank_ms: list[float] = []

    for item in gold:
        query = normalize(item["query"])
        expected = set(item["expected_pages"])

        t0 = time.perf_counter()
        vector = embed_query(query)
        candidates = store.search(vector, top_k=args.candidates, score_threshold=0.0)
        dense_ms.append((time.perf_counter() - t0) * 1000)

        dense_pages = [c.metadata.get("page_number") for c in candidates]
        dense_recall.append(recall_at(dense_pages, expected, args.top_k))

        t1 = time.perf_counter()
        scores = score_pairs(query, [c.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        rerank_ms.append((time.perf_counter() - t1) * 1000)

        reranked_pages = [candidates[i].metadata.get("page_number") for i in order]
        rerank_recall.append(recall_at(reranked_pages, expected, args.top_k))

        print(
            f"  {item['id']:>4}  dense={dense_recall[-1]:.0f} rerank={rerank_recall[-1]:.0f}  "
            f"({dense_ms[-1]:.0f}ms + {rerank_ms[-1]:.0f}ms)",
            flush=True,
        )

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    report = {
        "n_queries": len(gold),
        "top_k": args.top_k,
        "candidates": args.candidates,
        "dense": {
            "recall_at_k": mean(dense_recall),
            "latency_ms_mean": mean(dense_ms),
        },
        "reranked": {
            "recall_at_k": mean(rerank_recall),
            "latency_ms_mean": mean(rerank_ms),
        },
        "delta": {
            "recall_at_k": round(mean(rerank_recall) - mean(dense_recall), 4),
            "latency_ms_added": round(mean(rerank_ms), 2),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
