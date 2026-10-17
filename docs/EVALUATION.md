# Evaluation

Measured results for the VāṇiVerse retrieval and generation pipeline. Every number on this page
came from a script in this repository, run against the local stack — none are estimated.

**Headline:** retrieval finds an answer-bearing page for **90%** of in-corpus questions (5.9× the
random baseline), **100%** of out-of-domain questions are refused rather than hallucinated, and the
retrieval stage answers in **~277 ms**. The weak points are the 3B local model's generation quality
and the fixed relevance threshold. Romanized-Telugu retrieval rose from **60% to 100%** after the
transliteration fix — see [§5](#5-change-log).

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
| **Recall@5** | **0.900** (27/30) |
| Precision@5 | 0.240 |
| **MRR** | **0.686** |
| *Random baseline recall@5* | *0.152 (5/33)* |

**Recall by query language** — the interesting part:

| Language | n | Recall@5 | MRR |
| --- | --- | --- | --- |
| English | 5 | **1.000** | 0.840 |
| Telugu | 20 | 0.850 | 0.618 |
| Romanized Telugu | 5 | **1.000** | 0.800 |

Retrieval latency: **p50 277 ms**, p95 288 ms, max 479 ms (first call 8.1 s — one-time model load).

### 2. Threshold behaviour (`score_threshold = 0.45`)

The threshold is the system's relevance gate. Its behaviour is the single most consequential finding:

| Group | Queries passing the gate |
| --- | --- |
| In-corpus | 23/30 (76.7%) |
| Out-of-domain | **0/10 (0%)** |

Top-1 similarity score distribution:

| Group | min | median | max |
| --- | --- | --- | --- |
| In-corpus | 0.362 | **0.546** | 0.710 |
| Out-of-domain | 0.295 | **0.346** | 0.447 |

The distributions separate at the median but **overlap** (in-corpus min 0.362 < out-of-domain max
0.447). No threshold cleanly divides them — raising it costs in-corpus recall, lowering it leaks
out-of-domain queries. This is the fundamental limit of a fixed similarity cutoff with a single
small embedding model, and it is why a reranker is the next real improvement.

### 3. End-to-end grounding (40 queries, 0 HTTP errors)

> ⚠️ **This section has not been re-run** since the transliteration fix in [§5](#5-change-log). It
> reflects the previous normalizer, so the refusal and abstention figures below are stale — the
> retrieval gate pass rate already improved from 70% to 76.7%, so false abstention should fall from
> 30% to roughly 23%. Treat these as a pre-fix baseline until re-measured.

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

### 4. Retrieval misses (3 of 30)

| Query | Expected page(s) | Retrieved | Note |
| --- | --- | --- | --- |
| `t07` విజ్ఞానమయ కోశం అంటే ఏమిటి? | 3 | 2, 30, 8, 7, 22 | returned page 2, which introduces the kosha framework — adjacent, not wrong |
| `t08` ఆనందమయ కోశం అంటే ఏమిటి? | 4 | 2, 30, 23, 22, 7 | same; pages 3–4 continue a list that starts on page 2 |
| `t19` ద్వంద్వం ఎప్పుడు అదృశ్యమవుతుంది? | 23, 29 | 19, 14, 26, 20, 7 | genuine miss |

Both previous romanized misses (`r02`, `r05`) are now retrieved at ranks 1 and 2 — see [§5](#5-change-log).
Every remaining miss is native Telugu and `t07`/`t08` are "adjacent page" cases rather than genuine
failures.

---

## Findings

1. **Retrieval works, and works cross-lingually.** 90% recall@5 against a 15% random baseline. An
   **English** question reliably surfaces **Telugu** passages (5/5) — the multilingual embedding
   model carries the cross-lingual load, not the transliterator.
2. **Refusal is the strongest result.** 10/10 out-of-domain questions were refused with an explicit
   "not available in the provided sources" answer — including both fabrication traps (author, price).
   Zero hallucinations observed at the retrieval gate, because the gate rejects them before the LLM
   is ever called.
3. **The threshold is a blunt instrument.** It delivers that 100% precision at a cost of 30% false
   abstention on real questions. This is the clearest actionable finding: the system is
   *conservative*, not *wrong*. Tuning `SCORE_THRESHOLD` trades recall for out-of-domain leakage.
4. **Romanized Telugu was the weakest path, and the hypothesis about why was correct.** The original
   result was 60% (vs 85% Telugu / 100% English), attributed to the rule-based Roman→Telugu map
   producing a query that differed from the indexed text. That was stated as a hypothesis because the
   evaluation did not isolate the transliterator. It has since been tested directly, and the failure
   was more basic than a *bad* transliteration — for pure-Roman input the transliterator **never ran
   at all**. Romanized recall is now 100%. See [§5](#5-change-log).
5. **Generation quality, not plumbing, is the ceiling.** `qwen2.5:3b` produced one prompt-template
   echo and at least one false refusal on a question the corpus answers. The wiring is correct; this
   is a model-size limit, and the evidence for using the Groq 70B provider for any real use.

---

## 5. Change log

### Romanized-Telugu transliteration fix

**Scope:** the retrieval layer was re-measured. The end-to-end section ([§3](#3-end-to-end-grounding-40-queries-0-http-errors))
was **not** re-run and is stale.

Two independent defects in `ingestion/normalizer.py` were found and fixed. The first is a routing
bug; the second is the reason the first existed.

1. **The syllable walk never ran on pure-Roman input.** For `roman`-script text the code applied only
   `_PHRASE_MAP` — 17 exact-substring phrases. Four of the five Romanized gold questions matched
   nothing in it and reached the embedder as **raw ASCII**:

   | id | query | `normalize()` before | after |
   | --- | --- | --- | --- |
   | `r01` | `nenu evaru?` | నేను ఎవరు? *(phrase map)* | నేను ఎవరు? |
   | `r02` | `eruka ante emiti?` | *unchanged* | ఎరుక అన్తె ఎమితి? |
   | `r03` | `ghadha nidralo eruka untunda?` | *unchanged* | ఘధ నిద్రలో ఎరుక ఉన్తున్ద? |
   | `r04` | `samudram mariyu alalu udaharana emiti?` | *unchanged* | సముద్రం మరియు అలలు ఉదహరన ఎమితి? |
   | `r05` | `nija svarupam ante emiti?` | *unchanged* | నిజ స్వరుపం అన్తె ఎమితి? |

2. **The syllable map could not express a consonant followed by a vowel.** It emitted *independent*
   vowel letters after consonants — `eruka` became `ఎరఉకఅ` instead of `ఎరుక` — which is precisely why
   the walk had been gated to `mixed`-script text. It was replaced with an abugida-aware state
   machine: a vowel after a consonant becomes a dependent sign (matra), a consonant after a consonant
   takes a virama, and word-final `m` becomes anusvara. The consonant table was also switched to the
   ISO 15919 default of dental `t`/`d`/`n`, which is what makes `samudram` → `సముద్రం` exact.

**Before/after** — identical gold set, identical config, retrieval layer only:

| Metric | Before | After |
| --- | --- | --- |
| Recall@5 | 0.833 | **0.900** |
| MRR | 0.636 | **0.686** |
| Precision@5 | 0.227 | 0.240 |
| Romanized recall@5 | 0.600 | **1.000** |
| Telugu recall@5 | 0.850 | 0.850 |
| English recall@5 | 1.000 | 1.000 |
| In-corpus gate pass | 70% | 76.7% |
| Out-of-domain gate pass | 0% | **0%** |

Reproduce with the API running: `python eval/run_eval.py --skip-query`.

**How much to trust the 0.600 → 1.000 jump: not much on its own.** The Romanized group is **5
questions**, so that swing is *two* questions, and the interval on n=5 is enormous. The overall
figure moves 25/30 → 27/30 for the same reason. What *is* defensible: the fix is a rule set with **no
benchmark-specific vocabulary added**; the two questions that flipped were the two previously
identified as genuine misses; and there was no regression on the other 35 questions or on
out-of-domain rejection.

**Limitations introduced or left in place:**

- **The English guard is stopword-based** (≥2 English stopwords ⇒ skip transliteration). A
  stopword-free English string is indistinguishable from Romanized Telugu and will be transliterated
  (`"hello"` → `హెల్లొ`). A language identifier, not a rule, is the real fix.
- **Ad-hoc phonetic spellings are not recoverable by any rule set.** The gold set's Romanization is
  not a consistent scheme — `emiti` is used for ఏమిటి and `ante` for అంటే — so they transliterate to
  ఎమితి and అన్తె. These are high-frequency function words, so a future improvement should target
  them, via a general Telugu lexicon rather than by adding this gold set's vocabulary.
- **The accepted input alphabet is lower-case ASCII**, so retroflex (ట/డ/ణ) cannot be expressed.
- **No held-out Romanized set exists.** The fix was written before being measured against this set
  and added no vocabulary from it, but a second, unseen Romanized set is the only way to make the
  figure fully trustworthy.

### Cross-encoder reranking — implemented and verified, not yet measured end-to-end

**Scope:** the retrieval path gained an optional second stage. `RERANK_ENABLED` is **off by
default**, and every number in [§1](#1-retrieval-quality-top_k--5-of-33-chunks) and
[§2](#2-threshold-behaviour-score_threshold--045) was produced with it off. The dense path was
re-measured on the post-change code (`eval/rerank_dense.json`) and is unchanged: recall@5 0.900,
precision@5 0.240, MRR 0.6856, gate pass 0.7667, out-of-domain rejection 0.0 — an exact
reproduction of the 2A figures, which is the evidence that the reranker did not disturb it.

What was verified:

| Check | Result |
| --- | --- |
| Unit tests (`tests/test_reranker.py`) | 17 pass, offline, no model download |
| API smoke test | `reranked: true`; results ordered by rerank score (0.9950 / 0.9796 / 0.9699) with dense scores preserved separately (0.6207 / 0.5967 / 0.5515) |
| Ranking sanity on the real corpus | `నేను ఎవరు?` scores the relevant passage **0.3666** against an unrelated one **0.0000** |
| Missing dependency | degrades to dense-only with a single warning; a model that loads but fails to score raises `RerankError` |

**Measured cost — the main result of this work.** Reranking 20 full ~512-token chunks on CPU
takes **~34 s per query** (~1.4–1.7 s per pair). That is why `RERANK_ENABLED` is off in the
hosted default: it is a real latency cost, and on Streamlit Community Cloud a second ~2.3 GB
model does not fit alongside BGE-M3.

**What was *not* measured, stated plainly.** The before/after retrieval delta for the reranked
configuration was not obtained. Both attempts were lost to an environment fault rather than to
the reranker: on Windows, uvicorn's asyncio accept loop intermittently dies with
`OSError: [WinError 64] The specified network name is no longer available`, which leaves a run
apparently healthy on one surviving keep-alive connection while the listening socket is already
gone — so the run continues, then fails the moment a new connection is opened. The harness now
holds a single connection (`keepalive_expiry=900`) against a server started with
`--timeout-keep-alive 300`, which is what keeps a run alive at all; the third attempt was cut
for time rather than completed. **Do not quote a reranked recall figure: there isn't one.**

To produce it:

```bash
RERANK_ENABLED=true python -m uvicorn api.main:app --port 8000 --timeout-keep-alive 300
python eval/run_eval.py --skip-query --out eval/rerank_on.json           # ~40 min on CPU
python eval/threshold_study.py --out eval/gate_reranked.json             # ~25 min on CPU
```

**Two caveats that would apply to any number from this corpus.**

- `RERANK_CANDIDATES=20` against a **33-chunk** corpus means the reranker sees roughly 61% of it.
  A recall lift measured here would be easy to produce and would not be evidence at scale.
- `RERANK_THRESHOLD=0.5` sits **above** the 0.3666 that the known-relevant passage scores, so the
  shipped default would abstain on a correct answer. It is a placeholder pending
  `eval/threshold_study.py`; treating it as calibrated would be wrong.

### Answer-language, citation and text-layer fixes

These move no retrieval number above, but they change observable behaviour.

- **Romanized Telugu was answered in English.** `detect_script` reports `"roman"` for both English
  and Romanized Telugu, and every `"roman"` query was routed to the English system prompt, whose
  rule 2 forbids Telugu — which made `_SYSTEM_PROMPT_TELUGU`'s own rule for Romanized input
  unreachable. Prompt selection now resolves an answer language from the script *plus* the query,
  reusing the stopword heuristic the normalizer already used to gate transliteration. Verified over
  HTTP against the embedded API: `nenu evaru?` → 85–88% Telugu characters, `who am i?` → 0%,
  `నేను ఎవరు?` → 86–88%.
- **Citations could name a source that was never provided.** The blocks are labelled
  `[SOURCE 1]..[SOURCE N]` but nothing stated that N was a ceiling; against a five-block prompt the
  model cited `[SOURCE 6]` in **1 of 6** sampled runs. The prompt now states the bound numerically.
  After the change: **0 of 6**. With n=6 that is suggestive, not established — the definite part is
  that the bound was genuinely absent before.
- **`fitz` → `pymupdf`.** The dependency was already `pymupdf`; the deprecated alias printed a
  warning on every run and will stop working when it is removed.

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
