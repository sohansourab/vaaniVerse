# VāṇiVerse Product Roadmap

## Overview

This roadmap translates the architecture into a sequence of milestones for building a production-grade
Indic-language RAG system. It balances research quality, cultural preservation requirements, and
engineering reliability.

## Status at a glance

This file is the **single source of truth** for project phase. The roadmap table in the
[README](../README.md#roadmap) summarises these same six milestones; if the two ever disagree, this
file wins.

| Milestone | Focus | Status |
| --- | --- | --- |
| **1 — Foundation & Stability** | Core pipeline, provider switch, config, API/ingestion docs | ✅ Complete, except transliteration quality |
| **2 — Retrieval Quality & Evaluation** | Gold set, retrieval + grounding metrics, per-language analysis | ✅ Complete, except nDCG/MAP and data-driven tuning |
| **3 — Robustness & Document Handling** | Transliteration, reranking, layout, token estimation, OCR, provenance | 🔄 In progress |
| **4 — Cultural Intelligence** | Inscriptions, historical documents, heritage knowledge graph, community workflows | Planned |
| **5 — Indic Expansion** | Hindi, Tamil, Kannada, Sanskrit, cross-language retrieval | Planned |
| **6 — Production Readiness** | Telemetry, health checks, rate limits, admin tooling, deployment | Planned |

Every claim of "complete" above should be checkable against a measured number in
[EVALUATION.md](EVALUATION.md). Anything not backed by a number there is a plan, not a claim.

---

## Milestone 1 — Foundation & Stability

### Goal

Stabilize the core retrieval and generation pipeline for Telugu, Romanized Telugu, and English queries.

### Status

✅ **Complete, with one exception.** The pipeline, provider switch, configuration and documentation are
done and covered by 15 passing tests. The transliteration heuristic exists but is measurably weak:
romanized-Telugu retrieval is **60%** against **85%** for native Telugu. That gap is carried into
milestone 3.

### Tasks

- finalize the multilingual normalization pipeline
- validate script detection on real examples
- improve transliteration heuristics for Romanized Telugu
- ensure source-grounded prompting is enforced
- add explicit failure handling for empty retrieval results
- standardize configuration and environment variables
- document API contracts and ingestion flow

### Exit criteria

- queries in Telugu, Romanized Telugu, and English return consistent behavior
- no silent hallucinations when context is missing
- retrieval and answer generation are tracked through logs and metadata

---

## Milestone 2 — Retrieval Quality & Evaluation

### Goal

Measure and improve retrieval accuracy and answer quality using a repeatable evaluation framework.

### Status

✅ **Complete, except two gaps.** `eval/gold_qa.jsonl` (40 questions), `eval/run_eval.py` and
[EVALUATION.md](EVALUATION.md) are shipped, covering Recall@k, Precision@k, MRR, per-language
breakdowns and JSON output. Still outstanding: **nDCG and MAP** are not implemented, and
`chunk_size` / `top_k` / `score_threshold` remain hand-set rather than data-driven. Exact match and
F1 are **deliberately** not reported — see the threats to validity in [EVALUATION.md](EVALUATION.md).

### Tasks

- create benchmark datasets for Telugu and English queries
- add retrieval metrics: Recall@k, MRR, nDCG, MAP
- add answer metrics: Exact Match, F1, groundedness, citation quality
- add per-language evaluations for Telugu, Romanized Telugu, and mixed-script inputs
- store benchmark results in JSON or CSV output
- identify failure cases and fix prompts or chunking heuristics

### Exit criteria

- benchmark results are repeatable and stored in a machine-readable format
- each major language mode has separate evaluation metrics
- retrieval thresholds and chunking settings are data-driven

---

## Milestone 3 — Robustness & Document Handling

### Goal

Increase reliability on real-world, historical and cultural documents: noisy extraction, bilingual
content, and the weakest query path.

### Status

🔄 **In progress.** Two workstreams are active, both chosen because the evaluation identified them as
the measured weak points: **transliteration** (romanized retrieval at 60%) and **reranking** (the
0.45 threshold yields 100% out-of-domain rejection but 30% false abstention, and the in-corpus and
out-of-domain score distributions overlap, so no fixed cutoff can fix it). OCR is **not started** —
no OCR code exists in the repository.

### Tasks

- **[active]** fix the Romanized-Telugu transliteration path — measured 60% recall@5 vs 85% native
- **[active]** add a reranking stage to replace the fixed `score_threshold` cutoff
- improve OCR pipeline for scanned manuscripts and degraded PDFs
- add OCR-first ingestion for manuscripts and images, with page-level metadata and captions
- add fallback behavior when OCR confidence is low
- add document provenance metadata for source attribution
- create language-specific handling for literary, historical, and philosophical texts
- add chunk overlap tuning for long-form verse and prose, and better token estimation
- refine source labeling in prompt construction
- support domain-specific retrieval filters for sources and page ranges
- keep the text-first workflow stable while the OCR path is added (optional, modular, observable)

### Exit criteria

- fragile or noisy documents are normalized and retrievable
- answers reflect source provenance with clear citations
- cultural and historical materials are handled with stronger context retention
- scanned documents become usable without breaking the main system

---

## Milestone 4 — Cultural Intelligence

### Goal

Move beyond retrieval into understanding: inscriptions, historical documents, and the relationships
between concepts, places and texts.

### Status

Planned. No work has started.

### Tasks

- inscription understanding — script identification and epigraphic convention handling
- historical document handling with date, region and dynasty metadata
- build a heritage knowledge graph linking people, places, texts and concepts
- community workflows for contributing and correcting heritage material
- provenance-aware answers that distinguish primary text from editorial annotation

### Exit criteria

- inscriptions and historical documents are retrievable and correctly attributed
- the knowledge graph can be traversed to answer relationship questions
- community contributions are reviewable and traceable

---

## Milestone 5 — Indic Expansion

### Goal

Extend the pipeline from Telugu to the wider Indic language family without redesigning the core.

### Status

Planned. The README lists this publicly; the pipeline is currently Telugu-first.

### Tasks

- add Hindi, Tamil, Kannada and Sanskrit support to normalization and script detection
- add cross-language retrieval (ask in one language, retrieve in another)
- per-language evaluation sets following the milestone 2 methodology
- language-specific transliteration rather than the current Telugu-only rule set
- document the per-language quality so the weakest language is visible, not averaged away

### Exit criteria

- a new language can be added without touching API, UI or storage code
- each supported language has its own measured retrieval numbers
- cross-language retrieval is quantified, not assumed

---

## Milestone 6 — Production Readiness

### Goal

Prepare the platform for real-world deployments and user-facing use.

### Status

Planned.

### Tasks

- add telemetry and audit logging
- support observability for latency, retrieval quality, and answer quality
- implement index health checks and reset/rebuild routines
- add robust API rate limits and request validation
- build admin tooling for ingestion and index stats
- formalize deployment configuration for local and cloud environments

### Exit criteria

- the system is stable under repeated ingestion and querying
- critical operational metrics are visible to maintainers
- failures can be diagnosed quickly from logs and metadata

---

## Priority order

1. Retrieval quality and language handling
2. Source-grounded answer reliability
3. Benchmarking and evaluation workflows
4. Robustness on noisy documents, then OCR and multimodal expansion
5. Cultural intelligence, Indic expansion
6. Production observability and deployment readiness

---

## Suggested immediate tasks

The previous list here (benchmark dataset, evaluation scripts, docs folder, results schema,
normalization regression tests) is **done** — see milestone 2. Current next steps:

- fix the Romanized-Telugu transliteration path, then re-run `eval/run_eval.py --skip-query` to
  measure the delta against the recorded 60% baseline
- add a reranker stage and re-run the threshold study to test whether false abstention falls
  without losing the 10/10 out-of-domain refusal
- implement nDCG and MAP to close out milestone 2's metric list
- add a scanned/degraded PDF fixture so milestone 3's OCR work is testable
- overwrite the hand-set `chunk_size` / `top_k` / `score_threshold` with values derived from a sweep
