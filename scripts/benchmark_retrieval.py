#!/usr/bin/env python3
"""Benchmark VāṇiVerse retrieval latency and result quality against a local API.

Usage examples:
    python scripts/benchmark_retrieval.py --api-url http://localhost:8000 --dataset data/eval/queries.jsonl --top-k 5

The dataset expects JSONL entries like:
    {"query": "నేను ఎవరు", "language": "telugu"}

If the API is down, the script prints a clear error and exits with a non-zero status.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def benchmark(api_url: str, queries: list[dict[str, Any]], top_k: int, score_threshold: float) -> dict[str, Any]:
    endpoint = api_url.rstrip("/") + "/query"
    results: list[dict[str, Any]] = []
    total_latency = 0.0

    for item in queries:
        query_text = str(item.get("query") or "").strip()
        if not query_text:
            continue

        payload = {
            "query": query_text,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        start = time.perf_counter()
        response = httpx.post(endpoint, json=payload, timeout=120)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_latency += elapsed_ms

        response.raise_for_status()
        body = response.json()
        result_count = len(body.get("sources_used", [])) or len(body.get("results", []))

        results.append({
            "query": query_text,
            "latency_ms": round(elapsed_ms, 2),
            "result_count": result_count,
            "answer_length": len(str(body.get("answer") or "")),
            "language": item.get("language", "unknown"),
        })

    return {
        "api_url": api_url,
        "total_queries": len(results),
        "average_latency_ms": round(total_latency / len(results), 2) if results else 0.0,
        "max_latency_ms": round(max(item["latency_ms"] for item in results), 2) if results else 0.0,
        "min_latency_ms": round(min(item["latency_ms"] for item in results), 2) if results else 0.0,
        "average_result_count": round(sum(item["result_count"] for item in results) / len(results), 2) if results else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the VāṇiVerse retrieval API.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="Base URL for the FastAPI endpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL file containing queries to benchmark.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval results to request.")
    parser.add_argument("--score-threshold", type=float, default=0.45, help="Minimum retrieval threshold.")
    args = parser.parse_args()

    queries = load_queries(args.dataset)
    summary = benchmark(args.api_url, queries, args.top_k, args.score_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
