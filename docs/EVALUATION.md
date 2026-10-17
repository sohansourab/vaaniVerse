# Evaluation

Measured results for the VāṇiVerse retrieval and generation pipeline. Every number on this page
came from a script in this repository, run against the local stack — none are estimated.

**Headline:** retrieval finds an answer-bearing page for **83%** of in-corpus questions (5.5× the
random baseline), **100%** of out-of-domain questions are refused rather than hallucinated, and the
retrieval stage answers in **~277 ms**. The weak points are romanized-Telugu retrieval (**60%**) and
the 3B local model's generation quality.

---

## What was measured

Two layers are measured separately, because mixing them hides which one is failing:

| Layer | Endpoint | What it isolates | Cost |
| --- | --- | --- | --- |
| **Retrieval** | `GET /search` | normalize → embed → FAISS search | ~280 ms |
| **End-to-end** | `POST /query` | the above + prompt + LLM | ~3–32 s |

Reporting a single "accuracy" number for a RAG system is not meaningful: it conflates whether the
right text was *found* with whether the model *used* it well. These are reported apart.

## Setup

| | |
| --- | --- |
| Corpus | 1 Telugu PDF, 33 pages → **33 chunks** (one per page) |
| Embedding model | `BAAI/bge-m3`, 1024-d, L2-normalized |
| Vector store | FAISS `IndexFlatIP`, 33 vectors |
| LLM | Ollama `qwen2.5:3b` (local, 1.9 GB) |
| Config | chunk 512 / overlap 64 · `top_k=5` · `score_threshold=0.45` |
| GPU | NVIDIA GeForce GTX 1650, 4 GB · driver 610.88 |
| Runtime | Python 3.14.6 · torch 2.14.1+cu130 (CUDA 13.0) |

**Gold set:** `eval/gold_qa.jsonl` — 40 questions.

| Group | Count | Purpose |
| --- | --- | --- |
| Telugu | 20 | native-script retrieval |
| Romanized Telugu | 5 | exercises the transliteration path |
| English | 5 | exercises cross-lingual retrieval + the English guard |
| Out-of-domain | 10 | tests refusal: politics, geography, cooking, history, weather, software, demographics, plus two **fabrication traps** ("Who is the author of this book?", "What is the price of this book?") |

Each in-corpus question carries the page number(s) that legitimately answer it, hand-assigned by
reading the document. An answer counts as retrieved if **any** expected page appears in the top-k.

---

## Results

### 1. Retrieval quality (top_k = 5 of 33 chunks)

| Metric | Value |
| --- | --- |
| **Recall@5** | **0.833** (25/30) |
| Precision@5 | 0.227 |
| **MRR** | **0.636** |
| *Random baseline recall@5* | *0.152 (5/33)* |

**Recall by query language** — the interesting part:

| Language | n | Recall@5 | MRR |
| --- | --- | --- | --- |
| English | 5 | **1.000** | 0.840 |
| Telugu | 20 | 0.850 | 0.618 |
| Romanized Telugu | 5 | **0.600** | 0.500 |

Retrieval latency: **p50 277 ms**, p95 288 ms, max 479 ms (first call 8.1 s — one-time model load).

### 2. Threshold behaviour (`score_threshold = 0.45`)

The threshold is the system's relevance gate. Its behaviour is the single most consequential finding:

| Group | Queries passing the gate |
| --- | --- |
| In-corpus | 21/30 (70%) |
| Out-of-domain | **0/10 (0%)** |

Top-1 similarity score distribution:

| Group | min | median | max |
| --- | --- | --- | --- |
| In-corpus | 0.362 | **0.533** | 0.710 |
| Out-of-domain | 0.295 | **0.346** | 0.447 |

The distributions separate at the median but **overlap** (in-corpus min 0.362 < out-of-domain max
0.447). No threshold cleanly divides them — raising it costs in-corpus recall, lowering it leaks
out-of-domain queries. This is the fundamental limit of a fixed similarity cutoff with a single
small embedding model, and it is why a reranker is the next real improvement.

### 3. End-to-end grounding (40 queries, 0 HTTP errors)

| Metric | Value |
| --- | --- |
| Out-of-domain queries refused | **10/10 (100%)** |
| In-corpus queries that refused (false abstention) | 9/30 (30%) |
| **Grounded answer rate** (answered *and* cited an expected page) | **56.7%** (17/30) |
| Fully grounded (all expected pages cited) | 43.3% (13/30) |
| Script classification correct | 40/40 |
| Citation precision / recall | 0.259 / 0.730 — *see caveat* |

**The end-to-end funnel for the 30 in-corpus questions:**

```
30 in-corpus questions
 ├── 9  (30%) refused        ← no chunk cleared the 0.45 threshold
 ├── 17 (57%) answered        ← ≥1 correct page cited
 └── 4  (13%) answered        ← but from sources that did not contain the answer
```

Latency: p50 **6.16 s** / p95 30.8 s overall. Answered queries p50 **15.8 s**; refused queries p50
**3.1 s** (no generation cost). The LLM, not retrieval, dominates latency.

**Two honest caveats on these numbers:**

1. **Citation precision is not an independent measure of attribution.** `POST /query` returns every
   chunk placed in the prompt as a "source", not just the ones the model relied on. Citation
   precision (0.259) therefore closely tracks retrieval precision (0.227). To measure true
   attribution you would parse the `[SOURCE N]` markers the prompt requests from the answer — the
   3B model emits them inconsistently (and in one case echoed the prompt template verbatim), so
   that measurement would be noisy at this model size.
2. **`scripts/evaluate_rag.py` reports different citation figures** (precision 0.136, recall 0.633)
   because it averages over all 40 rows, scoring refused rows as zero precision and giving empty
   `expected_sources` a recall of 1.0. `eval/run_eval.py` excludes refused rows from citation
   averages, which is the more meaningful denominator. Both are reported above; only one is a
   headline.

### 4. Retrieval misses (5 of 30)

| Query | Expected page(s) | Retrieved | Note |
| --- | --- | --- | --- |
| `t07` విజ్ఞానమయ కోశం అంటే ఏమిటి? | 3 | 2, 30, 8, 7, 22 | returned page 2, which introduces the kosha framework — adjacent, not wrong |
| `t08` ఆనందమయ కోశం అంటే ఏమిటి? | 4 | 2, 30, 23, 22, 7 | same; pages 3–4 continue a list that starts on page 2 |
| `t19` ద్వంద్వం ఎప్పుడు అదృశ్యమవుతుంది? | 23, 29 | 19, 14, 26, 20, 7 | genuine miss |
| `r02` eruka ante emiti? | 15 | 21, 24, 32, 25, 20 | genuine miss (romanized) |
| `r05` nija svarupam ante emiti? | 1, 5 | 31, 6, 24, 26, 12 | genuine miss (romanized) |

---

## Findings

1. **Retrieval works, and works cross-lingually.** 83% recall@5 against a 15% random baseline. An
   **English** question reliably surfaces **Telugu** passages (5/5) — the multilingual embedding
   model carries the cross-lingual load, not the transliterator.
2. **Refusal is the strongest result.** 10/10 out-of-domain questions were refused with an explicit
   "not available in the provided sources" answer — including both fabrication traps (author, price).
   Zero hallucinations observed at the retrieval gate, because the gate rejects them before the LLM
   is ever called.
3. **The threshold is a blunt instrument.** It delivers that 100% precision at a cost of 30% false
   abstention on real questions. This is the clearest actionable finding: the system is
   *conservative*, not *wrong*. Tuning `SCORE_THRESHOLD` trades recall for out-of-domain leakage.
4. **Romanized Telugu is the weakest path** (60% vs 85% Telugu / 100% English). The likely cause is
   the rule-based Roman→Telugu transliteration map producing a query that differs from the indexed
   text; English avoids that step entirely and benefits from BGE-M3's cross-lingual training. Stated
   as a hypothesis — this evaluation does not isolate the transliterator.
5. **Generation quality, not plumbing, is the ceiling.** `qwen2.5:3b` produced one prompt-template
   echo and at least one false refusal on a question the corpus answers. The wiring is correct; this
   is a model-size limit, and the evidence for using the Groq 70B provider for any real use.

---

## Threats to validity

Read these before quoting any number above.

- **Corpus size.** One document, 33 chunks, one language, one domain. Top-5 of 33 is a generous
  setting; recall@5 is not evidence of behaviour on a large multi-document corpus.
- **Chunk-per-page granularity.** One chunk per page makes page-level scoring exact, but it also
  means "adjacent page" misses (t07, t08) are penalised as fully wrong when the neighbouring text is
  contextually relevant for a single contiguous book.
- **Expected pages are author-assigned** by reading the document. Reasonable people could label
  `t19` as also expecting pages 26 or 28. Recall here is a useful signal, not a precise constant.
- **One embedding model, one LLM.** No comparison across models or chunk sizes. Changing
  `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `top_k` or `score_threshold` invalidates these numbers.
- **Single run, no variance estimate.** One pass per query; no confidence intervals.
- **Source text is degraded.** The PDF's Telugu extraction contains font-encoding artifacts
  (e.g. `మ ైన` for `మైన`). This depresses embedding quality and makes token-F1 meaningless — it is
  also why this evaluation does not report F1 or exact match.

## How to reproduce

Requires a running API and an ingested document (the PDF is not in the repository — see the README).

```bash
# start the API (loads BGE-M3 + FAISS)
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# full evaluation: 40 gold questions, retrieval + LLM  (~20 min on a GTX 1650)
.venv/Scripts/python eval/run_eval.py --out eval/results.json

# retrieval only, no LLM calls  (~30 s)
.venv/Scripts/python eval/run_eval.py --skip-query

# re-aggregate an existing run without repeating the LLM calls
.venv/Scripts/python eval/run_eval.py --from-results eval/results.json

# the repository's own metric script can consume the emitted file
.venv/Scripts/python scripts/evaluate_rag.py --results eval/results.jsonl
```

`eval/results.json` and `eval/results.jsonl` contain model answers grounded in the book text and are
**gitignored**, like `data/`. `eval/gold_qa.jsonl` — the questions, which we authored — is tracked.
