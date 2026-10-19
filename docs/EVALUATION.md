# Evaluation

Measured results for the VāṇiVerse retrieval and generation pipeline. Every number on this page
came from a script in this repository, run against the local stack — none are estimated.

**Headline:** retrieval finds an answer-bearing page for **91.9%** of in-corpus questions (6.1× the
random baseline), and the relevance gate refuses **15 of 16** out-of-domain questions before the LLM
is ever called — the one exception is a near-miss "who is the author" probe that scores just over the
Telugu cutoff. The retrieval stage answers in **~295 ms**. The weak points remain the 3B local
model's generation quality and the bluntness of a similarity cutoff. Romanized-Telugu retrieval rose
from **60% to 100%** after the transliteration fix — see [§5](#5-change-log).

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
| Config | chunk 512 / overlap 64 · `top_k=5` · per-script cutoff 0.41 (Telugu) / 0.46 (Roman+English) |
| GPU | NVIDIA GeForce GTX 1650, 4 GB · driver 610.88 |
| Runtime | Python 3.14.6 · torch 2.14.1+cu130 (CUDA 13.0) |

**Gold set:** `eval/gold_qa.jsonl` — 53 questions.

| Group | Count | Purpose |
| --- | --- | --- |
| Telugu | 23 | native-script retrieval |
| Romanized Telugu | 7 | exercises the transliteration path |
| English | 7 | exercises cross-lingual retrieval + the English guard |
| Out-of-domain | 16 | tests refusal: politics, geography, cooking, history, weather, software, demographics, **fabrication traps** (author, price, ISBN, publisher), and **near-miss probes** that share vocabulary with the corpus |

Each in-corpus question carries the page number(s) that legitimately answer it, hand-assigned by
reading the document. An answer counts as retrieved if **any** expected page appears in the top-k.

---

## Results

### 1. Retrieval quality (top_k = 5 of 33 chunks)

| Metric | Value |
| --- | --- |
| **Recall@5** | **0.919** (34/37) |
| Precision@5 | 0.238 |
| **MRR** | **0.696** |
| *Random baseline recall@5* | *0.152 (5/33)* |

**Recall by query language** — the interesting part:

| Language | n | Recall@5 | MRR |
| --- | --- | --- | --- |
| English | 7 | **1.000** | 0.771 |
| Telugu | 23 | 0.870 | 0.646 |
| Romanized Telugu | 7 | **1.000** | 0.786 |

Retrieval latency: **p50 295 ms**, p95 324 ms (first call 13.4 s — one-time model load).

### 2. Threshold behaviour

The threshold is the system's relevance gate, and its behaviour is the single most consequential
finding. Two configurations are reported: the naive single cutoff (`0.45`, what the project shipped
first) and the shipped default, which is calibrated **per script**.

**Single cutoff, `score_threshold = 0.45`:**

| Group | Queries passing the gate |
| --- | --- |
| In-corpus | 30/37 (81.1%) |
| Out-of-domain | 1/16 (6.3%) |

Top-1 similarity score distribution:

| Group | min | median | max |
| --- | --- | --- | --- |
| In-corpus | 0.362 | **0.569** | 0.710 |
| Out-of-domain | 0.295 | **0.352** | 0.472 |

The distributions separate at the median but **overlap** (in-corpus min 0.362 < out-of-domain max
0.472). No single threshold cleanly divides them — raising it costs in-corpus recall, lowering it
leaks out-of-domain queries. This is the fundamental limit of a fixed similarity cutoff with a
single small embedding model.

**Per-script thresholds (the shipped default).** One 0.45 cutoff is really a *Telugu* cutoff applied
to every script, because native-script queries embed closer to the (native-script) passages than
Roman or English queries do at equal relevance. Calibrating a cutoff per script — the value that
maximises in-corpus passage at zero out-of-domain passage (`scripts/calibrate_threshold.py`) — gives:

| Script | Threshold | In-corpus gate pass | Out-of-domain gate pass |
| --- | --- | --- | --- |
| Telugu (`SCORE_THRESHOLD_TELUGU`) | 0.41 | 23/23 | 1/3 |
| Roman / English (`SCORE_THRESHOLD_ROMAN`) | 0.46 | 12/14 | 0/13 |
| **Overall** | — | **94.6%** (35/37) | **6.3%** (1/16) |

That is up from 81.1% in-corpus with the single 0.45 cutoff. It is bought with **one** near-miss
leak: the Telugu **"ఈ పుస్తకం రచయిత ఎవరు?"** (who is the author) probe scores 0.472 against page 5,
clearing the lower Telugu cutoff. The same question in English scores below 0.46 and is refused, so
this is a genuine near-miss rather than an artefact of the single cutoff. The trade is deliberate —
the leaked question cannot be answered from the corpus, so the LLM prompt's own abstention rule is
expected to refuse it. These are the shipped defaults; re-run `scripts/calibrate_threshold.py` after
any corpus or embedder change.

### 3. End-to-end grounding

> ⚠️ **This section is stale on two counts** and must not be quoted as current. It was measured
> before the transliteration fix in [§5](#5-change-log), and before the gold set grew from 40 to 53
> questions. It reflects the previous normalizer and the old set: the retrieval gate pass rate has
> since risen from 70% → 81.1% (single cutoff) → 94.6% (per-script), so false abstention should now be
> lower, and the denominators below no longer match the gold file. Treat it as a pre-fix baseline
> until re-measured with an LLM (`eval/run_eval.py`, without `--offline`).

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

### 4. Retrieval misses (3 of 37)

| Query | Expected page(s) | Retrieved | Note |
| --- | --- | --- | --- |
| `t07` విజ్ఞానమయ కోశం అంటే ఏమిటి? | 3 | 2, 30, 8, 7, 22 | returned page 2, which introduces the kosha framework — adjacent, not wrong |
| `t08` ఆనందమయ కోశం అంటే ఏమిటి? | 4 | 2, 30, 23, 22, 7 | same; pages 3–4 continue a list that starts on page 2 |
| `t19` ద్వంద్వం ఎప్పుడు అదృశ్యమవుతుంది? | 23, 29 | 19, 14, 26, 20, 7 | genuine miss |

Every miss is native Telugu; all Romanized and English questions retrieve (7/7 and 7/7), and
`t07`/`t08` are "adjacent page" cases rather than genuine failures. The two Romanized misses that
existed before the transliteration fix (`r02`, `r05`) are retrieved at ranks 1 and 2 — see
[§5](#5-change-log).

---

## Answer quality (LLM judge)

Everything above measures *retrieval* and *abstention*. It does not measure whether the answer the
model wrote is any good, because there is no way to do that with string matching: the stored Telugu
is font-mangled, so a correct answer never matches the source text, and token-F1 scores extraction
damage rather than answer quality — see [Threats to validity](#threats-to-validity).

`eval/judge_answers.py` fills that gap with a **reference-free LLM judge**. For each answer it is
given the question, the answer, and the passages the system cited — and nothing else. It scores four
dimensions from 1–5 (`relevance`, `faithfulness`, `conciseness`, `overall`) and rules on two
booleans: whether the answer is in the question's language, and whether abstaining was the right
call. Cited passages are recovered from the local index pickle, so the judge sees the actual context
the model was handed. The gold set is used only to tell the judge whether a question *should* be
answerable, never to supply a reference answer.

> **Not yet run.** Producing honest judge numbers requires an LLM, and the box used for the rest of
> this page has no reachable Ollama server or `GROQ_API_KEY`. This section documents the harness, not
> results. Two things make that acceptable rather than hand-wavy: the judge is scored against the
> same run whose retrieval behaviour is already measured above, and `--emulated` plus the unit tests
> exercise the whole pipeline without a model, so the plumbing is covered on every push.

The judge is itself a model and can be wrong. It is a second opinion on the same answers, useful for
tracking regressions between runs, not ground truth. Where it disagrees with a human reading, the
human reading wins — which is why the per-answer verdicts land in `eval/judge.json` rather than only
in a summary mean. A single malformed completion is counted as unscored rather than scored zero
(`n_unscored`), so a bad judge call cannot quietly depress the aggregate.

---

## Findings

1. **Retrieval works, and works cross-lingually.** 91.9% recall@5 against a 15% random baseline. An
   **English** question reliably surfaces **Telugu** passages (7/7) — the multilingual embedding
   model carries the cross-lingual load, not the transliterator.
2. **Refusal is strong at the retrieval gate, and now measured against near-misses.** 15 of 16
   out-of-domain questions — including the author/price/ISBN/publisher fabrication traps — fail the
   gate and abstain before the LLM is called. The single pass is the Telugu "who is the author"
   probe, a near-miss that the LLM prompt's own abstention rule is still expected to refuse.
3. **The threshold is a blunt instrument, and per-script calibration helps.** A single cutoff gives
   81.1% gate pass in-corpus; per-script cutoffs raise that to 94.6% at the cost of one near-miss
   leak. The system is *conservative*, not *wrong*; tuning the cutoff trades recall for
   out-of-domain leakage.
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

**Scope:** the retrieval layer was re-measured. The end-to-end section ([§3](#3-end-to-end-grounding))
was **not** re-run and is stale. The before/after table below was measured on the **original
40-question** gold set, before it was expanded to 53; it is retained as the record of the fix, not as
the current headline (which is 0.919 recall@5 on the expanded set, [§1](#1-retrieval-quality-top_k--5-of-33-chunks)).

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

### Cross-encoder reranking — measured, and off by default

**Scope:** the retrieval path gained an optional second stage. `RERANK_ENABLED` is **off by
default**, and every number in [§1](#1-retrieval-quality-top_k--5-of-33-chunks) and
[§2](#2-threshold-behaviour) was produced with it off. The dense path was re-measured on the
post-change code (`eval/rerank_dense.json`, original 40-question set) and is unchanged: recall@5
0.900, precision@5 0.240, MRR 0.6856, gate pass 0.7667, out-of-domain rejection 0.0 — an exact
reproduction of the pre-reranker figures, which is the evidence that the reranker did not disturb
the dense path.

What was verified:

| Check | Result |
| --- | --- |
| Unit tests (`tests/test_reranker.py`) | 17 pass, offline, no model download |
| API smoke test | `reranked: true`; results ordered by rerank score (0.9950 / 0.9796 / 0.9699) with dense scores preserved separately (0.6207 / 0.5967 / 0.5515) |
| Ranking sanity on the real corpus | `నేను ఎవరు?` scores the relevant passage **0.3666** against an unrelated one **0.0000** |
| Missing dependency | degrades to dense-only with a single warning; a model that loads but fails to score raises `RerankError` |

**Measured end to end (`scripts/measure_reranker.py`).** The before/after delta this document
previously did not have is now measured *in process* — the persisted index is loaded directly and
candidates are re-scored locally, so there is no HTTP connection to lose:

| Metric | Dense | + reranker | Δ |
| --- | --- | --- | --- |
| Recall@5 (30 in-corpus, original set) | 0.900 | **0.9333** | **+0.0333** (one question) |
| Mean latency / query | 973 ms | 77 107 ms | **+77.1 s** |

Latency is the reranker's own scoring time (`eval/reranker.json: delta.latency_ms_added`); the dense
stage still runs first, so a user waits the sum, ~78.1 s. An earlier run of the same script measured
65.0 s, and the 12 s spread is CPU thermal variance on the same machine, not a code change — both
runs agree on the recall figure exactly.

**Verdict: keep `RERANK_ENABLED=false`.** The lift is a single question out of 30 — well inside the
noise of a 33-chunk corpus — and `RERANK_CANDIDATES=20` means the reranker sees only ~61% of that
corpus, so the number is not evidence at scale. Against that, **+77 s/query on CPU is fatal for an
interactive demo.** The reranker is implemented and tested; it is simply not worth its cost here,
and that is now a measured conclusion rather than an assumption.

Reproduce (loads BGE-M3 and the cross-encoder; ~35 min on CPU):

```bash
.venv/Scripts/python -m scripts.measure_reranker --out eval/reranker.json
```

The earlier attempts to obtain this over HTTP were lost to a Windows uvicorn accept-loop fault
(`OSError: [WinError 64] The specified network name is no longer available`), as recorded in the
previous revision of this section; moving the measurement off the HTTP path is what made it
obtainable.

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

Retrieval-only runs (`--offline`) need just an ingested document; end-to-end runs add a running API.
The source PDF is not in the repository, so bootstrap a corpus first:

```bash
# bootstrap a public Telugu PDF and build data/vectors/ (~5 min, downloads BGE-M3 on first run)
.venv/Scripts/python -m scripts.load_sample

# ...or ingest your own
.venv/Scripts/python -m scripts.ingest_documents --input path/to/book.pdf
```

`scripts/load_sample.py` fetches a **different** book from the one measured here, so its numbers will
not match this page. The corpus-specific results reproduce only with the original 33-page document.

```bash
# start the API (loads BGE-M3 + FAISS)
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# full evaluation: 53 gold questions, retrieval + LLM  (~20 min on a GTX 1650)
.venv/Scripts/python eval/run_eval.py --out eval/results.json

# retrieval only, no LLM calls, no server needed  (~30 s)
.venv/Scripts/python eval/run_eval.py --offline

# re-aggregate an existing run without repeating the LLM calls
.venv/Scripts/python eval/run_eval.py --from-results eval/results.json

# the repository's own metric script can consume the emitted file
.venv/Scripts/python scripts/evaluate_rag.py --results eval/results.jsonl

# reference-free LLM judge over the answers from a live run (writes eval/judge.json)
.venv/Scripts/python eval/judge_answers.py --results eval/results.jsonl
.venv/Scripts/python eval/judge_answers.py --emulated      # no model: plumbing check only
```

`eval/results.json` and `eval/results.jsonl` contain model answers grounded in the book text and are
**gitignored**, like `data/`. `eval/gold_qa.jsonl` — the questions, which we authored — is tracked.
