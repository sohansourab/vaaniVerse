#!/usr/bin/env python3
"""
Judge VāṇiVerse answers with an LLM, reference-free.

Why reference-free: the source PDF's Telugu is damaged by font-encoding artifacts
on extraction, so a *correct* Telugu answer will never string-match the stored
text. Token-F1 and exact-match therefore measure extraction damage, not answer
quality (see eval/run_eval.py's docstring). What we can measure without a
reference answer is whether the answer is (a) relevant to the question, (b)
faithful to the passages it cites, (c) in the right language, and (d) abstains
exactly when the corpus cannot answer.

Input is the output of `eval/run_eval.py`:
    - eval/results.jsonl   (live run: query + answer + citations)   [default]
    - eval/results.json    (query_detail list — same fields, plus ids)
Both are produced by run_eval and are gitignored. The gold file supplies the
`query` → id / in_corpus / language mapping (results.jsonl rows carry no id).

Cited passages are recovered from the local index pickle, so the judge can read
the actual context the model was given — not just the citation ids.

Usage:
    python eval/judge_answers.py                       # judge eval/results.jsonl
    python eval/judge_answers.py --provider groq
    python eval/judge_answers.py --results eval/results.json
    python eval/judge_answers.py --dry-run --limit 2   # print prompts, no LLM call
    python eval/judge_answers.py --emulated            # heuristic stand-in (tests/CI)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RESULTS = REPO / "eval" / "results.jsonl"
DEFAULT_GOLD = REPO / "eval" / "gold_qa.jsonl"
DEFAULT_OUT = REPO / "eval" / "judge.json"
META_PICKLE = REPO / "data" / "vectors" / "vaaniverse_meta.pkl"

# The judge emits a single JSON object with these keys. Scores are 1–5; the two
# booleans are tri-state (`null` when the dimension does not apply, e.g. language
# match on an abstention).
SCORE_KEYS = ("relevance", "faithfulness", "conciseness", "overall")

_SYSTEM_PROMPT = """\
You are a strict, fair evaluator of a Telugu/English retrieval-augmented QA system.

You will be shown: the user's question, the system's answer, and the passages the
system cited (its only source of truth). You are NOT given a reference answer, and
you must not invent one. Judge only against the cited passages and the question.

Score each dimension from 1 (bad) to 5 (excellent):

  relevance     — does the answer actually address the question asked?
  faithfulness  — is every factual claim supported by the cited passages?
                  5 = fully supported; 1 = claims contradict or invent material.
                  If the system abstains, score 5 when the passages genuinely do
                  not contain the answer, else 1 if they clearly do.
  conciseness   — is it focused, without padding or repetition?

Also decide:
  language_match     — is the answer in the same language as the question
                       (Telugu question → Telugu answer; English → English)?
  abstention_correct — true if declining was the right call, false if it declined
                       a question the cited passages answer, or answered a question
                       that should have been declined. null if not applicable.

Reply with ONE JSON object and nothing else, no markdown fences:
{"relevance": 1-5, "faithfulness": 1-5, "conciseness": 1-5, "overall": 1-5,
 "language_match": true|false|null, "abstention_correct": true|false|null,
 "notes": "one short sentence"}
"""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    """Map normalized query text → gold record (results.jsonl rows carry no id)."""
    by_query: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_query
    for item in load_jsonl(path):
        query = item.get("query")
        if isinstance(query, str):
            by_query[_norm(query)] = item
    return by_query


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_meta_index(path: Path = META_PICKLE) -> dict[str, str]:
    """chunk_id → passage text, read straight from the persisted index."""
    if not path.exists():
        return {}
    try:
        records = pickle.loads(path.read_bytes())
    except Exception:
        return {}
    out: dict[str, str] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str):
            out.setdefault(chunk_id, record.get("text", "") or "")
    return out


def load_answers(path: Path) -> list[dict[str, Any]]:
    """
    Normalize an eval output file into judge rows:

        {query, answer, cited_chunk_ids, language, in_corpus, id, abstained}

    Accepts `results.jsonl` (run_eval) or `results.json` (query_detail).
    """
    if path.suffix.lower() == ".jsonl":
        raw = load_jsonl(path)
        rows = [
            {
                "id": None,
                "query": r.get("query", ""),
                "answer": r.get("answer", ""),
                "cited_chunk_ids": r.get("citations", []) or [],
                "language": r.get("language"),
                "in_corpus": None,
                "abstained": not (r.get("citations") or []),
            }
            for r in raw
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("query_detail") or []
        rows = [
            {
                "id": r.get("id"),
                "query": r.get("query", ""),
                "answer": r.get("answer", ""),
                "cited_chunk_ids": r.get("cited_chunk_ids", []) or [],
                "language": r.get("language"),
                "in_corpus": r.get("in_corpus"),
                "abstained": bool(r.get("abstained")),
            }
            for r in raw
            if r.get("status") == 200
        ]
    return rows


def build_prompt(row: dict[str, Any], passages: list[str],
                 gold_item: dict[str, Any] | None) -> str:
    cited = "\n\n".join(
        f"[{i + 1}] {(p or '(no text)').strip()[:1200]}"
        for i, p in enumerate(passages)
    ) or "(the system cited no passages)"

    where = ""
    if gold_item is not None:
        where = ("This question IS known to be answerable from the corpus."
                 if gold_item.get("in_corpus")
                 else "This question is NOT answerable from the corpus; the correct "
                      "behaviour is to decline.")

    return (
        f"{where}\n\n"
        f"QUESTION:\n{row['query'].strip()}\n\n"
        f"ANSWER GIVEN:\n{(row['answer'] or '(empty)').strip()}\n\n"
        f"CITED PASSAGES:\n{cited}\n"
    )


def _clamp_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, score))


def _as_tristate(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
    return None


def parse_judgement(text: str) -> dict[str, Any] | None:
    """
    Extract the judge's JSON object from a completion.

    Tolerant of markdown fences and surrounding prose — small local models do
    both — but returns None rather than guessing when no object is present, so a
    malformed judgement is counted as a failure instead of silently scored 0.
    """
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip())
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    obj: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return None

    judgement: dict[str, Any] = {}
    for key in SCORE_KEYS:
        score = _clamp_score(obj.get(key))
        if score is not None:
            judgement[key] = score
    judgement["language_match"] = _as_tristate(obj.get("language_match"))
    judgement["abstention_correct"] = _as_tristate(obj.get("abstention_correct"))
    notes = obj.get("notes")
    judgement["notes"] = notes.strip()[:300] if isinstance(notes, str) else ""

    # A judgement with no scores at all is not usable.
    if not any(k in judgement for k in SCORE_KEYS):
        return None
    return judgement


def emulated_judgement(row: dict[str, Any], passages: list[str]) -> dict[str, Any]:
    """
    Deterministic stand-in used by tests and offline CI.

    Heuristic, not a real judge: an answer that cites passages is treated as
    relevant and faithful; a refusal (no citations) is credited only when we do
    not expect an answer. It exists so the pipeline can be exercised without
    calling a model, and is labelled as such in the output.
    """
    answered = bool(passages)
    in_corpus = row.get("in_corpus")
    if not answered:
        should_answer = in_corpus is True
        return {
            "relevance": 1 if should_answer else 4,
            "faithfulness": 1 if should_answer else 4,
            "conciseness": 5,
            "overall": 1 if should_answer else 4,
            "language_match": None,
            "abstention_correct": (not should_answer) if in_corpus is not None else None,
            "notes": "emulated: refusal",
        }
    return {
        "relevance": 4,
        "faithfulness": 3,
        "conciseness": 3,
        "overall": 3,
        "language_match": True,
        "abstention_correct": (False if in_corpus is False else None),
        "notes": "emulated: answered from cited passages",
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _rate(values: list[bool]) -> float | None:
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("judgement")]
    unscored = [r for r in rows if not r.get("judgement")]

    summary: dict[str, Any] = {
        "n": len(rows),
        "n_scored": len(scored),
        "n_unscored": len(unscored),
        "unscored_ids": [r.get("id") or r.get("query", "")[:40] for r in unscored],
        "mean": {
            key: _mean([r["judgement"][key] for r in scored
                        if key in r["judgement"]])
            for key in SCORE_KEYS
        },
        "language_match_rate": _rate([r["judgement"]["language_match"] for r in scored
                                      if r["judgement"].get("language_match") is not None]),
        "abstention_accuracy": _rate([r["judgement"]["abstention_correct"] for r in scored
                                      if r["judgement"].get("abstention_correct") is not None]),
    }

    # Overall mean broken out by language and by in/out-of-corpus, because a
    # single number hides the difference that actually matters here.
    def _group(predicate) -> dict[str, Any]:
        subset = [r for r in scored if predicate(r)]
        return {
            "n": len(subset),
            "overall": _mean([r["judgement"]["overall"] for r in subset
                              if "overall" in r["judgement"]]),
            "faithfulness": _mean([r["judgement"]["faithfulness"] for r in subset
                                   if "faithfulness" in r["judgement"]]),
        }

    summary["by_language"] = {
        lang: _group(lambda r, lang=lang: r.get("language") == lang)
        for lang in sorted({r.get("language") for r in scored if r.get("language")})
    }
    summary["by_corpus"] = {
        "in_corpus": _group(lambda r: r.get("in_corpus") is True),
        "out_of_domain": _group(lambda r: r.get("in_corpus") is False),
    }
    return summary


def judge_one(call_llm, row: dict[str, Any], passages: list[str],
              gold_item: dict[str, Any] | None) -> dict[str, Any] | None:
    prompt = build_prompt(row, passages, gold_item)
    try:
        completion = call_llm(_SYSTEM_PROMPT, prompt)
    except Exception as exc:  # a single bad call must not abort the run
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return parse_judgement(completion)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-judge VāṇiVerse answers.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="eval/results.jsonl or eval/results.json.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provider", default=None, help="ollama | groq (default: env).")
    parser.add_argument("--limit", type=int, default=0, help="Judge only the first N rows.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompts and exit without calling the model.")
    parser.add_argument("--emulated", action="store_true",
                        help="Use the deterministic heuristic judge (no model calls).")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found. Run eval/run_eval.py first "
              "(live, without --skip-query).", file=sys.stderr)
        return 2

    rows = load_answers(args.results)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("ERROR: no answered rows found (did the eval run with --skip-query?).",
              file=sys.stderr)
        return 2

    gold = load_gold(args.gold)
    meta = load_meta_index()
    if not meta:
        print("WARNING: index pickle not found; judging without cited passage text.",
              file=sys.stderr)

    for row in rows:
        gold_item = gold.get(_norm(row.get("query", "")))
        if gold_item:
            row["id"] = row.get("id") or gold_item.get("id")
            row["in_corpus"] = gold_item.get("in_corpus")
            row["language"] = row.get("language") or gold_item.get("language")
        row["_gold"] = gold_item
        row["passages"] = [meta.get(cid, "") for cid in row.get("cited_chunk_ids", [])]

    if args.dry_run:
        for i, row in enumerate(rows, 1):
            print("=" * 70)
            print(f"# {i}  id={row.get('id')}  language={row.get('language')}  "
                  f"in_corpus={row.get('in_corpus')}")
            print(build_prompt(row, row["passages"], row["_gold"]))
        return 0

    call_llm = None
    provider_name = "emulated"
    model_name = "heuristic"
    if not args.emulated:
        from reasoning.generator import Generator

        generator = Generator() if args.provider is None else Generator(args.provider)
        provider = generator.get_provider(args.provider)
        provider_name = provider.name
        model_name = provider.model_name

        def call_llm(system_prompt: str, user_prompt: str) -> str:
            return provider.generate(user_prompt, system_prompt=system_prompt)

    print(f"[judge] {len(rows)} answers via {provider_name}/{model_name}...", flush=True)
    for i, row in enumerate(rows, 1):
        if args.emulated:
            row["judgement"] = emulated_judgement(row, row["passages"])
        else:
            row["judgement"] = judge_one(call_llm, row, row["passages"], row["_gold"])
        tag = row["id"] or row.get("query", "")[:24]
        verdict = row["judgement"]
        state = ("FAIL" if not verdict else
                 ("error" if "error" in verdict else
                  f"overall={verdict.get('overall', '?')}"))
        print(f"  {i:>3}  {tag:<6} {state}", flush=True)

    summary = aggregate(rows)
    summary["judge"] = {"provider": provider_name, "model": model_name,
                        "emulated": args.emulated, "source": str(args.results)}
    summary["detail"] = [
        {k: v for k, v in row.items() if k != "_gold"} for row in rows
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    headline = {k: v for k, v in summary.items() if k != "detail"}
    print("\n" + "=" * 70)
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    print("=" * 70)
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
