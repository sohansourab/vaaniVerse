# OCR — scanned and image-only pages

VāṇiVerse ingests PDFs. Most PDFs carry a text layer, and for those the pipeline
never touches OCR. A scanned manuscript does not — the page is a picture of
text. This document covers what handles that, why those components were chosen,
and what the measurements say.

**Status:** engine layer, parser fallback, fixture fetching and the evaluation
harness are implemented and tested (49 passing tests). Measurements in
[Results](#results).

---

## 1. Where it plugs in

Exactly one place. `_extract_text_from_page()` in
[`ingestion/parser.py`](../ingestion/parser.py) already fell back from `text` to
`blocks` mode when a page came back empty; it now also falls back to OCR:

```
page → PyMuPDF text layer
         │
         ├── enough text?  → use it, OCR never runs
         │
         └── shorter than OCR_MIN_CHARS?
                    → rasterize with PyMuPDF (no temp file)
                    → ocr_image()
                    → text → existing normalize() → existing chunker
```

Three deliberate properties:

- **The threshold is a character count, not a truthiness check.** A page whose
  only extractable text is a stray page number is a scan for our purposes;
  `len(text) < OCR_MIN_CHARS` catches it, `not text` would not.
- **A page with real text never invokes the engine.** Verified by a test that
  makes the OCR call raise, then parses a normal text PDF.
- **OCR output re-enters the existing normalizer.** Nothing about the chunker,
  embedder or retrieval changes — so the transliteration fix in
  [`EVALUATION.md §5`](EVALUATION.md#5-change-log) applies to OCR text as well.

Rasterizing is free: PyMuPDF was already a dependency and can hand back raw
pixels, so no image is ever written to disk.

---

## 2. Engines

Only engines with **verified** Telugu support are registered. Everything in this
table was checked against the upstream source, not assumed from a feature list:

| Engine | Telugu code | Evidence | Licence | Cost |
| --- | --- | --- | --- | --- |
| **Tesseract** | `tel` | `tel.traineddata` ships in the official `tessdata_fast` repo | Apache-2.0 | CPU, 2.8 MB, 0 VRAM |
| **EasyOCR** | `te` | `easyocr/character/te_char.txt` and `te.txt` ship in the EasyOCR repo | Apache-2.0 | CPU-capable, reuses the existing torch |

### Ruled out, with reasons

| Candidate | Why not |
| --- | --- |
| **PaddleOCR** | No Telugu entry in `ppocr/utils/dict`. Claiming Telugu support would be wrong. |
| **Surya** | No Telugu mention anywhere in the repository. Unverified, so not advertised. |
| **GPU VLM OCR**¹ | Real and likely better on degraded scans, but needs a GPU. The GTX 1650 has 4 GB, already holding BGE-M3 + `qwen2.5:3b`. Cannot co-reside. |

¹ e.g. `SHAON123/indicpage-ocr-telugu` (GOT-OCR2.0 LoRA), `Yagna1/deepseek-ocr-telugu-*`,
`bodhan-ai/indic-ocr`.

They remain a plausible Milestone 3 upgrade behind the same `ocr_image()`
interface — that is the point of making the layer pluggable.

### Why `tesseract` is the default

`auto` resolves in preference order and **short-circuits**: it returns Tesseract
without ever importing `easyocr`. That matters because importing EasyOCR pulls
in torch, which is the single most expensive import in the project. A pipeline
that only ever sees text PDFs should not pay for it — and on a text PDF it
doesn't.

Language codes are canonical ISO 639-1 (`te`, `en`) and translated per engine,
because the two engines spell Telugu differently (`tel` vs `te`). The caller
never has to remember which is which.

---

## 3. Tesseract language data

The stock Windows installer is a dead end for Telugu:

- it ships only `eng` and `osd`,
- its silent install offers no language selection,
- and it writes into `Program Files`, which is **not user-writable** (verified:
  `Access to the path ... is denied`).

So [`scripts/fetch_ocr_tessdata.py`](../scripts/fetch_ocr_tessdata.py) stages a
self-contained `data/tessdata/` (gitignored): it copies what the system install
has, then downloads the missing models from `tessdata_fast`. On Linux and
Streamlit Cloud the equivalent is `packages.txt`, which asks the builder for
`tesseract-ocr` and `tesseract-ocr-tel`.

One trap worth recording, because it cost a debugging cycle: the staged
directory is pointed at with **`TESSDATA_PREFIX`**, not `--tessdata-dir`. The
option has to survive pytesseract's shell splitting, and a quoted path arrives
at Tesseract with the quote characters still attached:

```
Error opening data file "...\data\tessdata"/tel.traineddata
```

---

## 4. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_ENABLED` | `true` | Whether the parser may OCR at all |
| `OCR_ENGINE` | `auto` | `auto` \| `tesseract` \| `easyocr` |
| `OCR_LANG` | `te,en` | ISO 639-1, comma or `+` separated |
| `OCR_DPI` | `200` | Page rasterization resolution |
| `OCR_MIN_CHARS` | `16` | Below this, a page counts as image-only |

Failure policy, so a missing engine can never break ingestion of a text PDF:

| `parse_document(ocr=…)` | Behaviour when no engine is installed |
| --- | --- |
| `None` (default) | Degrade. Record `ocr_unavailable` in `doc_metadata`, keep the text layer |
| `True` | Raise `OCREngineError` — an explicit request must not silently no-op |
| `False` | Never OCR |

---

## 5. Datasets

The roadmap item is **inference-time ingestion**, not model training. So the
datasets below are used to *measure* an engine, not to fit one. Ground truth was
inspected before being trusted.

| Dataset | Licence | What it is | Role |
| --- | --- | --- | --- |
| [`sarvamai/indic-ocr-bench`](https://huggingface.co/datasets/sarvamai/indic-ocr-bench) | Apache-2.0 | 6,909 semantic text blocks, 23 languages, scans from the 19th century to today; ground truth reviewed twice by human language experts | **primary** — recognition on pre-segmented text |
| [`AlbertoChestnut/telugu-ocr`](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr) | CC-BY-SA-4.0 | ~25,565 page scans + human transcriptions, 221 books, ~11 GB, from Telugu Wikisource | **full-page** — tests the path the app actually needs |

Both are fetched on demand into `data/ocr_bench/` and never committed. The
Wikisource corpus is share-alike and large, so only a small page slice is
pulled. That keeps the repository cleanly MIT.

### The dataset we did not use

[`meharuhanzz/OCR-Bench1000-Telugu`](https://huggingface.co/datasets/meharuhanzz/OCR-Bench1000-Telugu)
is the easiest to reach for: MIT-licensed, 1,000 tidy line images, a proper
LICENSE file. It is also **unusable as a benchmark**, because its own card says:

> This corpus was reviewed by someone who cannot personally read or verify Telugu.

The labels are synthetic and unverified. Scoring an engine against them would
produce a precise number with no meaning. It is deliberately absent from the
harness.

Verification performed on the source actually used: all 51 Telugu rows of the
Sarvam sample contain genuine Telugu codepoints, and the sample is balanced at
51 rows per language across 23 languages.

---

## 6. Results

Measured locally on CPU — no GPU — against the fixtures described in §5. CER/WER are
computed on NFC-normalized text with runs of whitespace collapsed, so layout differences are not
scored as recognition errors. The first fixture of each run is dropped as a warm-up, hence n=50
rather than 51.

| Corpus | Engine | n | CER ↓ | WER ↓ | Exact | Mean conf | p50 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sarvam text blocks | **Tesseract** | 50 | **0.137** | **0.448** | 0.020 | 0.78 | **819 ms** |
| Sarvam text blocks | EasyOCR | 50 | 0.238 | 0.717 | 0.000 | 0.65 | 1,692 ms |
| Wikisource page scans | **Tesseract** | 6 | 0.489 | **1.058** | 0.000 | 0.54 | **5,003 ms** |
| Wikisource page scans | EasyOCR | 6 | **0.477** | 1.120 | 0.000 | 0.50 | 16,257 ms |

Raw per-image output, including every hypothesis, is in `eval/ocr_results.json`.

### What this says

**1. Tesseract is the default because it measured better, not because it was first.** On
pre-segmented text blocks it is clearly ahead (0.137 vs 0.238 CER) and about twice as fast. On full
pages the two are effectively tied — a 0.012 CER difference over six pages is noise — while EasyOCR
costs **16.3 s per page against Tesseract's 5.0 s**, a 3.3× penalty for no measured accuracy gain. On
this hardware and these corpora, EasyOCR buys nothing.

**2. The blocks-vs-pages gap is the real finding.** The same engine scores 0.137 on already-segmented
text and 0.489 on a real book page. Segmentation quality and 19th-century print cost far more than
the choice of recognizer does; picking between Tesseract and EasyOCR is a rounding error next to
that.

**3. Neither engine is good enough for faithful transcription of old Telugu print.** A WER above 1.0
means more edits than there are reference words. The text is readable in bursts but not quotable.
That is the concrete, measured case for the deferred GPU VLM in §2 — and the reason it is deferred
rather than dismissed.

**4. Transcription is not the requirement — retrieval is.** VāṇiVerse embeds OCR text in order to
*find* passages, not to reproduce them. A CER of 0.489 still leaves many content words intact, so
retrieval may well survive it. **This is untested.** Measuring recall on OCR-derived text against the
same gold set is a separate experiment, and it belongs on the roadmap rather than in this table as an
assumption.

### Threats to validity

- The page result rests on **six pages from a single 19th-century book**. 0.489 is an anecdote with a
  decimal point, not an estimate of Telugu OCR generally.
- The block corpus is a reasonable size (51 samples, 50 scored) but is **pre-segmented**, so it
  cannot see the layout failures that dominate the page result.
- Both corpora are literary or documentary. Neither represents modern, cleanly typeset pages, which
  is the case commercial OCR usually targets.
- Both engines ran on **CPU**, which is the deployment target (Streamlit Community Cloud has no GPU).
  GPU figures would be faster but would not change the rankings.

---

## 7. Limitations

- **Tesseract on real book scans is the weak link, and the writes are in the
  data, not the opinion.** The gap between block-level and full-page CER below
  is the honest headline — segmentation and 19th-century print both hurt.
- **Only Telugu and English are wired.** The language mapping table covers the
  other major Indic codes, but nothing else has been measured.
- **`OCR_LANG` mixes scripts into one recognizer call** (`tel+eng`). That is
  correct for mixed lines and possibly suboptimal for pure-Telugu pages.
- **No confidence-driven fallback.** `mean_confidence` is recorded per page but
  nothing acts on it yet — low-confidence pages are neither retried nor flagged
  to the user. That is the obvious next step and the reason the field exists.
- **No table, column or reading-order handling.** Blocks come back in
  Tesseract's order.

---

## 8. Reproducing

```bash
pip install -r requirements-ocr.txt        # pytesseract + easyocr
python -m scripts.fetch_ocr_tessdata       # stage Telugu language data
python -m scripts.fetch_ocr_bench          # 51 text blocks + 6 book pages
python eval/run_ocr_eval.py                # CER / WER / latency per engine

pytest tests/test_ocr.py -q                # routing + fallback, no engine needed
```

`--limit`, `--group`, `--engines`, `--device` and `--psm-*` narrow a run. The
raw per-image output lands in `eval/ocr_results.json` (gitignored).
