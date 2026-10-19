"""Unit tests for eval/judge_answers.py.

Fully offline: no model is called. The tests cover the parts that must be
trustworthy when the judge *is* called — prompt construction, tolerant parsing of
a model's JSON, and the aggregation — plus the emulated stand-in that keeps CI
model-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.judge_answers import (
    aggregate,
    build_prompt,
    emulated_judgement,
    load_answers,
    load_gold,
    parse_judgement,
)


# ── parse_judgement ────────────────────────────────────────────────────────────

def test_parses_plain_json():
    out = parse_judgement('{"relevance": 4, "faithfulness": 5, "conciseness": 3, '
                          '"overall": 4, "language_match": true, '
                          '"abstention_correct": null, "notes": "ok"}')
    assert out["relevance"] == 4 and out["faithfulness"] == 5
    assert out["language_match"] is True
    assert out["abstention_correct"] is None
    assert out["notes"] == "ok"


def test_parses_fenced_and_surrounded_json():
    out = parse_judgement(
        "Here is my evaluation:\n```json\n"
        '{"relevance": 2, "faithfulness": 1, "overall": 2}\n'
        "```\nHope that helps!"
    )
    assert out["relevance"] == 2 and out["overall"] == 2


def test_clamps_out_of_range_scores():
    out = parse_judgement('{"relevance": 9, "faithfulness": 0, "overall": 3}')
    assert out["relevance"] == 5
    assert out["faithfulness"] == 1


def test_rejects_completion_without_scores():
    assert parse_judgement("I cannot evaluate this.") is None
    assert parse_judgement("") is None
    assert parse_judgement('{"notes": "no scores here"}') is None


def test_string_booleans_are_coerced():
    out = parse_judgement('{"overall": 3, "language_match": "yes", '
                          '"abstention_correct": "no"}')
    assert out["language_match"] is True
    assert out["abstention_correct"] is False


# ── build_prompt ───────────────────────────────────────────────────────────────

def test_prompt_includes_question_answer_and_passages():
    row = {"query": "నేను ఎవరు?", "answer": "స్త్-చిత్-ఆనందం", "language": "telugu"}
    prompt = build_prompt(row, ["passage one", "passage two"], {"in_corpus": True})
    assert "నేను ఎవరు?" in prompt
    assert "స్త్-చిత్-ఆనందం" in prompt
    assert "[1] passage one" in prompt and "[2] passage two" in prompt
    assert "IS known to be answerable" in prompt


def test_prompt_marks_out_of_corpus_questions():
    row = {"query": "capital of France?", "answer": "", "language": "english"}
    prompt = build_prompt(row, [], {"in_corpus": False})
    assert "NOT answerable" in prompt
    assert "cited no passages" in prompt


# ── load_answers ───────────────────────────────────────────────────────────────

def test_load_answers_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"query": "q1", "answer": "a1", "citations": ["c1"],
                    "language": "telugu"}) + "\n"
        + json.dumps({"query": "q2", "answer": "", "citations": [], "language": "roman"})
        + "\n",
        encoding="utf-8",
    )
    rows = load_answers(path)
    assert len(rows) == 2
    assert rows[0]["cited_chunk_ids"] == ["c1"]
    assert rows[0]["abstained"] is False
    assert rows[1]["abstained"] is True


def test_load_answers_results_json_uses_query_detail(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "query_detail": [
            {"id": "t01", "status": 200, "query": "q", "answer": "a",
             "in_corpus": True, "language": "telugu",
             "cited_chunk_ids": ["c"], "abstained": False},
            {"id": "t02", "status": 500},
        ]
    }), encoding="utf-8")
    rows = load_answers(path)
    assert len(rows) == 1
    assert rows[0]["id"] == "t01" and rows[0]["in_corpus"] is True


def test_load_gold_maps_queries_to_records(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"id": "t01", "query": "  నేను   ఎవరు?  ",
                                "in_corpus": True}) + "\n", encoding="utf-8")
    gold = load_gold(path)
    assert any(item["id"] == "t01" for item in gold.values())


# ── aggregate ──────────────────────────────────────────────────────────────────

def _scored(overall, language="telugu", in_corpus=True, faith=4):
    return {"id": "x", "language": language, "in_corpus": in_corpus,
            "judgement": {"overall": overall, "faithfulness": faith,
                          "relevance": 4, "conciseness": 4,
                          "language_match": True, "abstention_correct": True}}


def test_aggregate_means_and_rates():
    rows = [_scored(5), _scored(3), _scored(4, in_corpus=False, language="roman")]
    summary = aggregate(rows)
    assert summary["n"] == 3 and summary["n_scored"] == 3
    assert summary["mean"]["overall"] == 4.0
    assert summary["language_match_rate"] == 1.0
    assert summary["abstention_accuracy"] == 1.0
    assert summary["by_corpus"]["in_corpus"]["n"] == 2
    assert summary["by_corpus"]["out_of_domain"]["n"] == 1


def test_aggregate_counts_unscored_rows():
    rows = [_scored(4), {"id": "bad", "language": "telugu", "in_corpus": True,
                         "judgement": None}]
    summary = aggregate(rows)
    assert summary["n_unscored"] == 1
    assert summary["unscored_ids"] == ["bad"]
    assert summary["mean"]["overall"] == 4.0


def test_aggregate_ignores_null_dimensions():
    row = _scored(4)
    row["judgement"]["language_match"] = None
    summary = aggregate([row])
    # The only row's language_match is null → the rate is undefined, not 0.
    assert summary["language_match_rate"] is None


# ── emulated judge ─────────────────────────────────────────────────────────────

def test_emulated_credits_refusal_only_when_unanswerable():
    answered = emulated_judgement({"in_corpus": True}, ["a passage"])
    assert answered["overall"] >= 3
    refused_on_answerable = emulated_judgement({"in_corpus": True}, [])
    assert refused_on_answerable["overall"] == 1
    refused_on_unanswerable = emulated_judgement({"in_corpus": False}, [])
    assert refused_on_unanswerable["abstention_correct"] is True


def test_emulated_flow_is_deterministic():
    row = {"query": "q", "answer": "a", "language": "telugu", "in_corpus": True}
    assert emulated_judgement(row, ["p"]) == emulated_judgement(row, ["p"])


@pytest.mark.parametrize("text", ['{"overall": 3}', '{"overall": "3"}'])
def test_overall_accepts_int_and_numeric_string(text):
    assert parse_judgement(text)["overall"] == 3
