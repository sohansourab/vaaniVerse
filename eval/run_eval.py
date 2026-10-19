#!/usr/bin/env python3
"""
Evaluate VāṇiVerse retrieval and end-to-end grounding against a live API.

This measures three things that matter, and deliberately avoids one that does not:

  1. RETRIEVAL   — does the ranker put an answer-bearing page in the top-k?
                   recall@k, precision@k and MRR over a hand-built gold set, with
                   a random baseline for scale, plus the score distribution of
                   in-corpus vs out-of-domain queries (can a threshold separate them?).
  2. GROUNDING   — when it answers, is the answer grounded? Two independent signals:
                   (a) does it ABSTAIN on questions the corpus cannot answer, and
                   (b) are the cited pages the answer-bearing ones.
  3. LATENCY     — retrieval-only (normalize + embed + FAISS) vs end-to-end, which
                   is dominated by the LLM. Reported as p50/p95.

It does NOT report token-F1 / exact-match as a headline number. The source PDF's
Telugu text is corrupted by font-encoding artifacts during extraction, so a correct
Telugu answer will never string-match the stored text; token similarity there
measures PDF extraction damage, not answer quality.

CAVEAT on citation precision/recall: POST /query returns every chunk that was placed
in the prompt as a "source", not only the ones the model actually leaned on. Citation
precision therefore closely tracks retrieval precision. The genuinely independent
grounding signal in this harness is the abstention behaviour.

Usage:
    python eval/run_eval.py                          # full run (slow: makes LLM calls)
    python eval/run_eval.py --limit 4                # smoke test
    python eval/run_eval.py --skip-query             # retrieval-only, no LLM
    python eval/run_eval.py --from-results eval/results.json   # re-aggregate, no API calls
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
GOLD = Path(__file__).resolve().parent / "gold_qa.jsonl"

# Offline mode imports the repo's own packages; make that work whether this is
# run as `python eval/run_eval.py` or `python -m eval.run_eval`.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOP_K = 5
PROD_THRESHOLD = 0.45  # the *global* shipped default (core/config.py: SCORE_THRESHOLD)
RANK_THRESHOLD = 0.0   # threshold disabled, to measure ranking quality alone

# gold `language` → the script detect_script() reports, for the script-aware gate.
_LANGUAGE_TO_SCRIPT = {"telugu": "telugu", "roman": "roman", "english": "roman"}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_gold(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return round(ordered[idx], 2)


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def page_to_chunk_map() -> dict[int, str]:
    """
    Best-effort page → chunk_id map, read from the local index metadata.

    Only used to emit a results file for scripts/evaluate_rag.py. Optional:
    if the pickle is absent (fresh clone, nothing ingested) the map is empty and
    those rows simply carry no expected_sources.
    """
    pkl = REPO / "data" / "vectors" / "vaaniverse_meta.pkl"
    if not pkl.exists():
        return {}
    try:
        records = pickle.loads(pkl.read_bytes())
    except Exception:
        return {}
    mapping: dict[int, str] = {}
    for record in records:
        page = record.get("page_number")
        if isinstance(page, int) and page not in mapping:
            mapping[page] = record.get("chunk_id", "")
    return mapping


# ── API calls ──────────────────────────────────────────────────────────────────

def call_search(client: httpx.Client, api: str, query: str,
                top_k: int, threshold: float) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = client.get(
        f"{api}/search",
        params={"q": query, "top_k": top_k, "score_threshold": threshold},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


class HttpSearcher:
    """Retrieval through a running API (GET /search)."""

    def __init__(self, client: httpx.Client, api: str):
        self._client = client
        self._api = api

    def search(self, query: str, top_k: int, threshold: float) -> tuple[list[dict], float]:
        payload, ms = call_search(self._client, self._api, query, top_k, threshold)
        return payload.get("results", []), ms


class OfflineSearcher:
    """
    Retrieval straight against the persisted index — no server, no HTTP.

    This exists because the reranker/retrieval reruns kept being lost to a
    uvicorn accept-loop fault: an in-process run has no socket to drop. Scripts
    are normalized exactly as the API does, so the numbers stay comparable to a
    live run for the retrieval layer.
    """

    def __init__(self, backend: str | None = None):
        from ingestion.normalizer import detect_script, normalize
        from retrieval.vector_store import VectorStore

        self._normalize = normalize
        self._detect = detect_script
        self._store = VectorStore() if not backend else VectorStore(backend=backend)
        self._store.load()

    @property
    def corpus_chunks(self) -> int:
        return self._store.total_vectors

    def search(self, query: str, top_k: int, threshold: float) -> tuple[list[dict], float]:
        normalized = self._normalize(query, transliterate=True)
        start = time.perf_counter()
        chunks = self._store.query(
            query_text      = normalized,
            top_k           = top_k,
            score_threshold = threshold,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        results = [
            {
                "page_number":   c.metadata.get("page_number", "?"),
                "score":         c.score,
                "chunk_id":      c.chunk_id,
                "retrieval_score": c.retrieval_score,
            }
            for c in chunks
        ]
        return results, elapsed_ms


def call_query(client: httpx.Client, api: str, query: str,
               top_k: int, threshold: float) -> tuple[dict[str, Any], float, int]:
    start = time.perf_counter()
    response = client.post(
        f"{api}/query",
        json={"query": query, "top_k": top_k, "score_threshold": threshold},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return (response.json() if response.content else {}), elapsed_ms, response.status_code


# ── collection phases ──────────────────────────────────────────────────────────

def collect_retrieval(searcher, gold: list[dict[str, Any]],
                      top_k: int, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for item in gold:
        ranked_results, ms = searcher.search(item["query"], top_k, RANK_THRESHOLD)
        gated_results, _ = searcher.search(item["query"], top_k, threshold)
        ranked = {"results": ranked_results}
        gated = {"results": gated_results}

        pages = [r["page_number"] for r in ranked.get("results", [])]
        expected = set(item["expected_pages"])
        hits = [p for p in pages if p in expected]
        first_rank = next((i + 1 for i, p in enumerate(pages) if p in expected), None)

        rows.append({
            "id": item["id"],
            "language": item["language"],
            "in_corpus": item["in_corpus"],
            "expected_pages": sorted(expected),
            "retrieved_pages": pages,
            "hit": bool(hits),
            "first_hit_rank": first_rank,
            "precision": (len(hits) / len(pages)) if pages else 0.0,
            "passed_threshold": len(gated.get("results", [])),
            "latency_ms": round(ms, 2),
            "top_scores": [round(r["score"], 4) for r in ranked.get("results", [])][:3],
        })
        print(f"  {item['id']:>4}  pages={pages}  hit={bool(hits)}  {ms:6.0f}ms", flush=True)
    return rows


def collect_queries(client: httpx.Client, api: str, gold: list[dict[str, Any]],
                    top_k: int, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for item in gold:
        payload, ms, status = call_query(client, api, item["query"], top_k, threshold)
        if status != 200:
            rows.append({"id": item["id"], "status": status,
                         "error": payload.get("detail"), "latency_ms": round(ms, 2)})
            print(f"  {item['id']:>4}  HTTP {status}  {ms:6.0f}ms", flush=True)
            continue

        cited = [s["page_number"] for s in payload.get("sources_used", [])]
        expected = set(item["expected_pages"])
        hits = [p for p in cited if p in expected]
        rows.append({
            "id": item["id"],
            "status": 200,
            "language": item["language"],
            "in_corpus": item["in_corpus"],
            "query_script": payload.get("query_script"),
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "answer": payload.get("answer", ""),
            "cited_pages": cited,
            "cited_chunk_ids": [s["chunk_id"] for s in payload.get("sources_used", [])],
            "expected_pages": sorted(expected),
            "citation_precision": (len(hits) / len(cited)) if cited else None,
            "citation_recall": (len(hits) / len(expected)) if expected else None,
            "abstained": not cited,
            "latency_ms": round(ms, 2),
        })
        print(f"  {item['id']:>4}  cited={cited}  abstained={not cited}  {ms:6.0f}ms", flush=True)
    return rows


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate(retrieval_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]],
              corpus_chunks: int, top_k: int, threshold: float,
              skipped: bool) -> dict[str, Any]:
    in_rows = [r for r in retrieval_rows if r["in_corpus"]]
    ood_rows = [r for r in retrieval_rows if not r["in_corpus"]]

    recall = (sum(1 for r in in_rows if r["hit"]) / len(in_rows)) if in_rows else 0.0
    mrr = (statistics.fmean([(1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0
                             for r in in_rows]) if in_rows else 0.0)
    precision = (statistics.fmean([r["precision"] for r in in_rows]) if in_rows else 0.0)

    by_language: dict[str, Any] = {}
    for lang in sorted({r["language"] for r in in_rows}):
        rows = [r for r in in_rows if r["language"] == lang]
        by_language[lang] = {
            "n": len(rows),
            "recall_at_k": round(sum(1 for r in rows if r["hit"]) / len(rows), 4),
            "mrr": round(statistics.fmean([(1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0
                                           for r in rows]), 4),
        }

    # Script-aware gate: apply the per-script threshold from config to each row's
    # top-1 score. This is what the API now does by default, so it is the number
    # that actually reflects production behaviour.
    try:
        from core.config import threshold_for_script
    except Exception:  # pragma: no cover - config import is trivial in practice
        threshold_for_script = lambda _script: threshold

    def _script_aware_pass(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        passed = sum(
            1 for r in rows
            if r["top_scores"]
            and r["top_scores"][0] >= threshold_for_script(_LANGUAGE_TO_SCRIPT.get(r["language"], ""))
        )
        return round(passed / len(rows), 4)

    summary: dict[str, Any] = {
        "retrieval": {
            "n_queries": len(retrieval_rows),
            "n_in_corpus": len(in_rows),
            "n_out_of_domain": len(ood_rows),
            "top_k": top_k,
            "corpus_chunks": corpus_chunks,
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
            "mrr": round(mrr, 4),
            "random_baseline_recall_at_k": round(top_k / corpus_chunks, 4) if corpus_chunks else 0.0,
            "by_language": by_language,
            "threshold": threshold,
            "gate_pass_rate_in_corpus": round(
                sum(1 for r in in_rows if r["passed_threshold"]) / len(in_rows), 4) if in_rows else 0.0,
            "gate_pass_rate_out_of_domain": round(
                sum(1 for r in ood_rows if r["passed_threshold"]) / len(ood_rows), 4) if ood_rows else 0.0,
            "script_aware_gate_pass_rate_in_corpus": _script_aware_pass(in_rows),
            "script_aware_gate_pass_rate_out_of_domain": _script_aware_pass(ood_rows),
            "top1_score_in_corpus": distribution([r["top_scores"][0] for r in in_rows if r["top_scores"]]),
            "top1_score_out_of_domain": distribution([r["top_scores"][0] for r in ood_rows if r["top_scores"]]),
            "latency_ms": latency_summary([r["latency_ms"] for r in retrieval_rows]),
        },
        "end_to_end": {"skipped": skipped},
        "retrieval_detail": retrieval_rows,
    }

    if query_rows:
        ok = [r for r in query_rows if r["status"] == 200]
        answered = [r for r in ok if not r["abstained"]]
        ood = [r for r in ok if not r["in_corpus"]]
        corpus = [r for r in ok if r["in_corpus"]]
        corpus_answered = [r for r in corpus if not r["abstained"]]

        cprec = [r["citation_precision"] for r in answered if r["citation_precision"] is not None]
        crec = [r["citation_recall"] for r in answered if r["citation_recall"] is not None]

        expected_script = {"telugu": "telugu", "roman": "roman", "english": "roman"}
        script_ok = sum(1 for r in ok if r["query_script"] == expected_script.get(r["language"]))

        summary["end_to_end"] = {
            "skipped": False,
            "n_queries": len(ok),
            "http_errors": len(query_rows) - len(ok),
            "answered": len(answered),
            "abstention_on_out_of_domain": round(
                sum(1 for r in ood if r["abstained"]) / len(ood), 4) if ood else 0.0,
            "abstention_in_corpus": round(
                sum(1 for r in corpus if r["abstained"]) / len(corpus), 4) if corpus else 0.0,
            "grounded_answer_rate": round(
                sum(1 for r in corpus_answered
                    if r["citation_recall"] and r["citation_recall"] > 0) / len(corpus), 4) if corpus else 0.0,
            "fully_grounded_answer_rate": round(
                sum(1 for r in corpus_answered
                    if r["citation_recall"] == 1.0) / len(corpus), 4) if corpus else 0.0,
            "citation_precision": round(statistics.fmean(cprec), 4) if cprec else 0.0,
            "citation_recall": round(statistics.fmean(crec), 4) if crec else 0.0,
            "citation_note": ("citation precision/recall are computed over every chunk placed in the "
                              "prompt, so precision tracks retrieval precision (see module docstring)"),
            "script_accuracy": round(script_ok / len(ok), 4) if ok else 0.0,
            "latency_ms": latency_summary([r["latency_ms"] for r in ok]),
            "latency_answered_ms": latency_summary([r["latency_ms"] for r in answered]),
            "latency_abstained_ms": latency_summary([r["latency_ms"] for r in ok if r["abstained"]]),
            "provider": ok[0]["provider"] if ok else None,
            "model": ok[0]["model"] if ok else None,
        }
        summary["query_detail"] = query_rows

    return summary


def write_results_jsonl(gold: list[dict[str, Any]], query_rows: list[dict[str, Any]], path: Path) -> None:
    """Emit a file that scripts/evaluate_rag.py can consume."""
    pmap = page_to_chunk_map()
    by_id = {g["id"]: g for g in gold}
    with path.open("w", encoding="utf-8") as fh:
        for r in query_rows:
            if r.get("status") != 200:
                continue
            item = by_id.get(r["id"], {})
            fh.write(json.dumps({
                "query": item.get("query", ""),
                "answer": r["answer"],
                "ground_truth": "",           # no reference answers: see module docstring
                "expected_sources": [pmap[p] for p in r["expected_pages"] if p in pmap],
                "citations": r["cited_chunk_ids"],
                "latency_ms": r["latency_ms"],
                "language": r["language"],
            }, ensure_ascii=False) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VāṇiVerse against a live API.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=PROD_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N gold items (smoke test).")
    parser.add_argument("--skip-query", action="store_true", help="Retrieval-only mode: skip LLM calls.")
    parser.add_argument("--from-results", type=Path, default=None,
                        help="Re-aggregate an existing results.json without any API calls.")
    parser.add_argument("--offline", action="store_true",
                        help="Run retrieval against the local index in-process (no API, "
                             "no HTTP) and skip /query. Immune to the uvicorn accept-loop "
                             "fault that lost earlier retrieval runs on Windows.")
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "results.json")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    if args.limit:
        gold = gold[: args.limit]

    # ── re-aggregate an existing run ──────────────────────────────────────────
    if args.from_results:
        previous = json.loads(args.from_results.read_text(encoding="utf-8"))
        retrieval_rows = previous["retrieval_detail"]
        query_rows = previous.get("query_detail", [])
        summary = aggregate(
            retrieval_rows, query_rows,
            corpus_chunks=previous["retrieval"]["corpus_chunks"],
            top_k=previous["retrieval"]["top_k"],
            threshold=previous["retrieval"]["threshold"],
            skipped=previous.get("end_to_end", {}).get("skipped", False),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        headline = {k: v for k, v in summary.items() if not k.endswith("_detail")}
        print(json.dumps(headline, ensure_ascii=False, indent=2))
        print("re-aggregated into:", args.out)
        if query_rows:
            write_results_jsonl(gold, query_rows, args.out.with_suffix(".jsonl"))
        return 0

    # ── offline retrieval-only run ────────────────────────────────────────────
    if args.offline:
        print(f"[offline] retrieval over {len(gold)} queries (top_k={args.top_k})...", flush=True)
        try:
            searcher = OfflineSearcher()
        except Exception as exc:
            print(f"ERROR: could not open the local index ({exc}). Ingest first.", file=sys.stderr)
            return 2
        if not searcher.corpus_chunks:
            print("ERROR: index is empty — ingest a document first.", file=sys.stderr)
            return 2
        retrieval_rows = collect_retrieval(searcher, gold, args.top_k, args.threshold)
        summary = aggregate(retrieval_rows, [], searcher.corpus_chunks, args.top_k,
                            args.threshold, skipped=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        headline = {k: v for k, v in summary.items() if not k.endswith("_detail")}
        print("\n" + "=" * 70)
        print(json.dumps(headline, ensure_ascii=False, indent=2))
        print("=" * 70)
        print("wrote:", args.out)
        return 0

    # ── live run ──────────────────────────────────────────────────────────────
    api = args.api_url.rstrip("/")
    # Reuse a single connection for the whole run. httpx would otherwise drop and
    # re-open a connection between every query (its default keepalive_expiry is
    # 5s, and these queries are ~55s apart), and each re-open is a fresh accept on
    # the server. On Windows, uvicorn's asyncio accept loop intermittently dies
    # with WinError 64 on exactly that path — the run then continues on the
    # surviving keep-alive connection until it, too, is dropped. A long expiry on
    # both sides means the socket is accepted once and reused. Pair this with
    # `--timeout-keep-alive 300` on the server, which is longer than one query.
    limits = httpx.Limits(max_keepalive_connections=2, keepalive_expiry=900.0)
    with httpx.Client(timeout=600.0, limits=limits) as client:
        health = client.get(f"{api}/health").json()
        print(f"health: {json.dumps(health)}", flush=True)
        if not health.get("total_vectors"):
            print("ERROR: index is empty — ingest a document first.", file=sys.stderr)
            return 2
        corpus_chunks = int(health["total_vectors"])
        searcher      = HttpSearcher(client, api)

        print(f"\n[1/2] retrieval over {len(gold)} queries (top_k={args.top_k})...", flush=True)
        retrieval_rows = collect_retrieval(searcher, gold, args.top_k, args.threshold)

        query_rows: list[dict[str, Any]] = []
        if not args.skip_query:
            print(f"\n[2/2] end-to-end /query over {len(gold)} queries (LLM calls, this is slow)...", flush=True)
            query_rows = collect_queries(client, api, gold, args.top_k, args.threshold)

    summary = aggregate(retrieval_rows, query_rows, corpus_chunks, args.top_k, args.threshold,
                        skipped=bool(args.skip_query))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    headline = {k: v for k, v in summary.items() if not k.endswith("_detail")}
    print("\n" + "=" * 70)
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    print("=" * 70)
    print("wrote:", args.out)

    if query_rows:
        write_results_jsonl(gold, query_rows, args.out.with_suffix(".jsonl"))
        print("wrote:", args.out.with_suffix(".jsonl"),
              "(for scripts/evaluate_rag.py — citation metrics are meaningful, F1/EM are not)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
