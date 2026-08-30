# VāṇiVerse Product Roadmap

## Overview

This roadmap translates the architecture into a sequence of milestones for building a production-grade Indic-language RAG system. It balances research quality, cultural preservation requirements, and engineering reliability.

---

## Milestone 1 — Foundation & Stability

### Goal

Stabilize the core retrieval and generation pipeline for Telugu, Romanized Telugu, and English queries.

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

## Milestone 3 — Cultural & Domain Robustness

### Goal

Increase reliability on historical and cultural documents with noisy extraction and bilingual content.

### Tasks

- improve OCR pipeline for scanned manuscripts and degraded PDFs
- add document provenance metadata for source attribution
- create language-specific handling for literary, historical, and philosophical texts
- add chunk overlap tuning for long-form verse and prose
- refine source labeling in prompt construction
- support domain-specific retrieval filters for sources and page ranges

### Exit criteria

- fragile or noisy documents are normalized and retrievable
- answers reflect source provenance with clear citations
- cultural and historical materials are handled with stronger context retention

---

## Milestone 4 — Multimodal Expansion

### Goal

Support scanned documents and image-based heritage materials without destabilizing the text-first core path.

### Tasks

- add OCR-first ingestion for manuscripts and images
- extract page-level metadata and image captions when needed
- merge multimodal metadata into the same retrieval corpus
- test mixed text + image retrieval quality
- add fallback behavior when OCR confidence is low

### Exit criteria

- text-first workflow remains stable
- scanned documents become usable without breaking the main system
- multimodal pipeline is optional, modular, and observable

---

## Milestone 5 — Production Readiness

### Goal

Prepare the platform for real-world deployments and user-facing use.

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
4. OCR and multimodal expansion
5. Production observability and deployment readiness

---

## Suggested immediate tasks

- add a benchmark dataset for Telugu and Romanized Telugu queries
- add evaluation scripts to compute metrics from stored outputs
- create a documentation folder with architecture and roadmap docs
- define a standard schema for benchmark JSON results
- add a simple regression suite for normalization and transliteration behavior
