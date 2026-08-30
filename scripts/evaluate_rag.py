#!/usr/bin/env python3
"""Evaluate VāṇiVerse-style RAG outputs from a JSON/JSONL result file.

Expected records:
    {
      "query": "నేను ఎవరు",
      "answer": "...",
      "ground_truth": "...",
      "expected_sources": ["source.pdf"],
      "citations": ["source.pdf"],
      "latency_ms": 1234.5,
      "language": "telugu"
    }

The script computes basic retrieval/answer quality metrics and prints a summary
as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(pred: str, truth: str) -> float:
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(truth_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, truth: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(truth) else 0.0


def citation_precision(citations: Iterable[str], expected_sources: Iterable[str]) -> float:
    cits = {str(item).strip() for item in citations if str(item).strip()}
    expected = {str(item).strip() for item in expected_sources if str(item).strip()}
    if not cits:
        return 0.0
    return len(cits & expected) / len(cits)


def citation_recall(citations: Iterable[str], expected_sources: Iterable[str]) -> float:
    cits = {str(item).strip() for item in citations if str(item).strip()}
    expected = {str(item).strip() for item in expected_sources if str(item).strip()}
    if not expected:
        return 1.0
    if not cits:
        return 0.0
    return len(cits & expected) / len(expected)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("results", [])
        raise ValueError(f"Unsupported JSON structure in {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {
        "exact_match": [],
        "f1": [],
        "citation_precision": [],
        "citation_recall": [],
        "latency_ms": [],
    }
    by_language: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        pred_answer = str(record.get("answer") or "")
        truth = str(record.get("ground_truth") or record.get("expected_answer") or "")
        expected_sources = record.get("expected_sources") or []
        citations = record.get("citations") or []
        language = str(record.get("language") or "unknown")
        latency = float(record.get("latency_ms") or 0.0)

        scores["exact_match"].append(exact_match(pred_answer, truth))
        scores["f1"].append(token_f1(pred_answer, truth))
        scores["citation_precision"].append(citation_precision(citations, expected_sources))
        scores["citation_recall"].append(citation_recall(citations, expected_sources))
        scores["latency_ms"].append(latency)

        by_language[language]["exact_match"].append(exact_match(pred_answer, truth))
        by_language[language]["f1"].append(token_f1(pred_answer, truth))
        by_language[language]["citation_precision"].append(citation_precision(citations, expected_sources))
        by_language[language]["citation_recall"].append(citation_recall(citations, expected_sources))
        by_language[language]["latency_ms"].append(latency)

    summary: dict[str, Any] = {
        "total_examples": len(records),
        "exact_match": round(sum(scores["exact_match"]) / len(scores["exact_match"]) if scores["exact_match"] else 0.0, 4),
        "f1": round(sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0, 4),
        "citation_precision": round(sum(scores["citation_precision"]) / len(scores["citation_precision"]) if scores["citation_precision"] else 0.0, 4),
        "citation_recall": round(sum(scores["citation_recall"]) / len(scores["citation_recall"]) if scores["citation_recall"] else 0.0, 4),
        "avg_latency_ms": round(sum(scores["latency_ms"]) / len(scores["latency_ms"]) if scores["latency_ms"] else 0.0, 2),
    }

    summary["by_language"] = {}
    for language, metric_map in sorted(by_language.items()):
        summary["by_language"][language] = {
            "exact_match": round(sum(metric_map["exact_match"]) / len(metric_map["exact_match"]) if metric_map["exact_match"] else 0.0, 4),
            "f1": round(sum(metric_map["f1"]) / len(metric_map["f1"]) if metric_map["f1"] else 0.0, 4),
            "citation_precision": round(sum(metric_map["citation_precision"]) / len(metric_map["citation_precision"]) if metric_map["citation_precision"] else 0.0, 4),
            "citation_recall": round(sum(metric_map["citation_recall"]) / len(metric_map["citation_recall"]) if metric_map["citation_recall"] else 0.0, 4),
            "avg_latency_ms": round(sum(metric_map["latency_ms"]) / len(metric_map["latency_ms"]) if metric_map["latency_ms"] else 0.0, 2),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VāṇiVerse RAG outputs.")
    parser.add_argument("--results", type=Path, required=True, help="Path to a JSON or JSONL file with evaluation records.")
    parser.add_argument("--output", type=Path, help="Optional path to write metrics JSON.")
    args = parser.parse_args()

    records = load_records(args.results)
    summary = compute_metrics(records)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
