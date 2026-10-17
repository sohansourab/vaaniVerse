<div align="center">

<img src="logo.png" width="440" alt="VāṇiVerse logo" />

# VāṇiVerse

**Source-grounded retrieval and reasoning for India's linguistic heritage.**

`తెలుగు` · *Romanized Telugu* · English

<br/>

### ▶️ [**Try the live demo →**](https://vaaniverse-27.streamlit.app/)

<br/>

[![Live demo](https://img.shields.io/badge/Streamlit%20Cloud-live-brightgreen?logo=streamlit&logoColor=white)](https://vaaniverse-27.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Embeddings](https://img.shields.io/badge/Embeddings-BGE--M3-orange)](https://huggingface.co/BAAI/bge-m3)
[![Vectors](https://img.shields.io/badge/Vectors-FAISS-yellow)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Groq-8A2BE2)](https://groq.com)
[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)](#testing)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

</div>

---

## What is VāṇiVerse

VāṇiVerse is a **multilingual Retrieval-Augmented Generation (RAG)** platform for Telugu and
other Indic heritage content. Ask a question in **Telugu**, **Romanized Telugu**, or **English**
and get an answer built *only* from retrieved source passages, with the source and page cited.
The answer comes back in the language you asked in: a Romanized Telugu question is answered in
Telugu script, an English one in English.

Modern retrieval systems are built for globally digitised, script-uniform text. Heritage corpora
break that assumption: the same query arrives as `రామ ఎవరు` or `rama evaru`, PDF extraction mangles
Telugu conjuncts, and the "right answer" is often a passage rather than a fact. VāṇiVerse treats
those as first-class problems.

| The problem | What VāṇiVerse does |
| --- | --- |
| Same query, different scripts | Script detection + Romanized-Telugu transliteration before embedding |
| Answers in the wrong language | English and Romanized Telugu are both "roman" to the script detector; they are separated before the prompt is chosen, so each gets its own answer language |
| Telugu PDFs extract badly | Unicode artifact repair (duplicate matras, broken conjuncts, zero-width chars) |
| Some pages are just pictures | OCR fallback (Tesseract / EasyOCR) for pages with no usable text layer |
| Generic splitting destroys context | Sentence-aware, overlap-aware chunking with per-chunk metadata |
| Model answers from memory | Evidence-only prompting; abstains when sources don't cover the question |
| Answers can't be verified | Every response returns `sources_used` with source, page, and score |

> **Live demo:** <https://vaaniverse-27.streamlit.app/> — hosted on Streamlit Community Cloud.
> There is no Ollama there, so it runs the **Groq** provider, and the container disk is ephemeral:
> upload a Telugu PDF in the sidebar and click **Ingest document** to build the index before asking.
> The first ingest downloads BGE-M3 (~2.2 GB) and embeds on CPU, so give it a few minutes.

> **Status:** Phases 1–2 complete — the end-to-end pipeline runs, the API and UI are live, the
> provider switch (Ollama ⇆ Groq) is implemented and tested, and retrieval is measured at
> **90% recall@5** ([docs/EVALUATION.md](docs/EVALUATION.md)). Phase 3 (robustness) is in progress:
> the Romanized-Telugu transliteration fix is **done**, OCR-first ingestion for image-only pages is
> **shipped** as a pluggable engine layer ([docs/OCR.md](docs/OCR.md)), and a cross-encoder reranker
> is implemented but **off by default and not yet measured end-to-end**. Languages beyond
> Telugu/English remain a roadmap item. See [Known limitations](#known-limitations).

---

## Architecture

The codebase is split into four responsibilities — **interface → application → domain → infrastructure** —
so that swapping the vector store, the embedding model, or the LLM never touches HTTP code.

```mermaid
flowchart LR
    UI["🖥️ Streamlit UI<br/>frontend/app.py"] -->|HTTP| API["⚡ FastAPI<br/>api/main.py"]
    CLI["🖥️ Ingestion CLI<br/>scripts/ingest_documents.py"] --> PRS

    subgraph INGEST["📥 Ingest path"]
        direction LR
        PRS["Parser"] --> NRM["Normalizer"] --> CHK["Chunker"] --> EMB["BGE-M3<br/>embeddings.py"]
    end

    subgraph QUERY["🔎 Query path"]
        direction LR
        VEC["Vector store<br/>vector_store.py"] --> PRM["Prompter"] --> GEN["Generator"]
    end

    API --> PRS
    API --> VEC
    EMB --> VEC
    VEC <--> FILES[("💾 faiss + pkl<br/>data/vectors")]
    EMB -.->|weights| HF["🌐 Hugging Face<br/>BAAI/bge-m3"]
    PRM -.->|grounded prompt| GEN
    GEN -.-> OLL["🌐 Ollama"]
    GEN -.-> GRQ["🌐 Groq"]
```

**The layers, and why they are drawn this way:** `api/main.py` owns orchestration — it imports
every service but no service imports it. `VectorStore` hides FAISS vs Qdrant behind one interface,
and `Generator` hides Ollama vs Groq behind another. Nothing in the API or UI knows which backend
is active; that decision lives in `core/config.py` (`LLM_PROVIDER`) and in the per-request payload.

---

## Deployment topologies

The UI talks to the API over HTTP on `http://localhost:8000`. Where that API runs is a deployment
choice, and the app supports all three:

```mermaid
flowchart LR
    subgraph T1["1 · Local development"]
        direction TB
        A1["Streamlit :8501"] -->|HTTP| A2["uvicorn :8000"]
        A2 --- A3["Ollama :11434"]
    end

    subgraph T2["2 · Streamlit Community Cloud"]
        direction TB
        B1["Streamlit :8501"] -->|"HTTP localhost"| B2["uvicorn<br/>daemon thread, :8000"]
        B2 --- B3["Groq API"]
    end

    subgraph T3["3 · Container"]
        direction TB
        C1["uvicorn :$PORT"] --- C2["Groq or Ollama"]
    end
```

| Topology | How the API starts | Typical provider |
| --- | --- | --- |
| **Local** | `uvicorn api.main:app` as a separate process | Ollama (offline, private) |
| **Streamlit Cloud** · [live ↗](https://vaaniverse-27.streamlit.app/) | `frontend/app.py` calls `ensure_api_ready()`, which boots uvicorn on a daemon thread inside the Streamlit process | Groq (no local GPU/Ollama available) |
| **Container** | `Dockerfile` + `docker run`, or any platform that injects `$PORT` | Groq or a networked Ollama |

Streamlit Community Cloud runs **one process**, so a separate backend is impossible there. The UI
therefore detects a missing API and starts one in-process — this is why `frontend/app.py` imports
`uvicorn` and `api.main`. `API_BASE_URL` stays overridable via the `VANIVERSE_API_URL` environment
variable or a Streamlit secret, so pointing at a remote API later needs no code change.

### Streamlit Cloud secrets

The UI copies flat string secrets into the environment *before* the embedded API is
imported, because `core/config.py` reads its configuration at import time — that is why
the bridge sits at the top of `frontend/app.py`. Paste this into
**App → Settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_..."   # required: the only provider reachable with no local server
LLM_PROVIDER = "groq"      # optional: see below
```

`GROQ_API_KEY` is the one that matters. `LLM_PROVIDER` is belt-and-braces: `/health`
reports `ollama_reachable`, and the sidebar drops Ollama from the provider list when
nothing answers on `OLLAMA_BASE_URL`, so the default falls through to Groq on its own.
`GROQ_MODEL` overrides the model id, whose default is listed under
[Configuration](#configuration) and which Groq retires on its own schedule — a retired id
returns `404 model_not_found`, which the API now reports verbatim rather than as a bare
"call failed".

Secrets are never committed: `GROQ_API_KEY` is read from the environment only, and
`.env.example` ships an empty placeholder.

---

## Ingestion pipeline

Turns a PDF into searchable, citation-ready vectors.

```mermaid
flowchart TD
    PDF["📄 PDF file"] --> PARSE["Parse — PyMuPDF<br/>pages, text, metadata"]
    PARSE --> UNI["Unicode NFC<br/>strip control chars"]
    UNI --> REPAIR["Telugu artifact repair<br/>duplicate matras · broken conjuncts<br/>zero-width chars · orphan virama"]
    REPAIR --> SPLIT["Sentence segmentation<br/>Telugu punctuation + spaCy"]
    SPLIT --> PACK["Greedy token packing<br/>512 tokens, 64 overlap"]
    PACK --> META["Attach metadata<br/>source · page · script · chunk_id"]
    META --> EMBED["Embed — BGE-M3<br/>1024-d, L2-normalised"]
    EMBED --> ADD["Index add — FAISS IndexFlatIP"]
    ADD --> SAVE["Persist<br/>vaaniverse.faiss + vaaniverse_meta.pkl"]
```

Because vectors are L2-normalised, FAISS inner product **is** cosine similarity — that is why the
`/search` and `/query` scores in the UI are directly interpretable (0.55–0.65 is a strong match for
this corpus).

Each chunk carries the metadata below, which is what makes citations possible later:

| Field | Example | Purpose |
| --- | --- | --- |
| `chunk_id` | `book_p4_0` | Stable identifier, returned to clients |
| `source` | `book.pdf` | Cited filename |
| `page_number` | `4` | Cited page |
| `script` | `telugu` | Drives answer-language policy |
| `chunk_index` | `0` | Order within the page |
| `text` | `…` | The retrievable passage |

---

## Query pipeline

Turns a question into a grounded, cited answer — or an honest abstention.

```mermaid
flowchart TD
    Q["Query<br/>తెలుగు / romanized / English"] --> PROV{"Provider valid?"}
    PROV -->|"no"| E400["400 — invalid provider"]
    PROV -->|"groq without key"| E503A["503 — GROQ_API_KEY missing"]
    PROV -->|"yes"| IDX{"Index non-empty?"}
    IDX -->|"no"| E503B["503 — index is empty"]
    IDX -->|"yes"| DET["detect_script → normalize<br/>repair / transliterate by script"]
    DET --> EMB["Embed query — BGE-M3"]
    EMB --> SEARCH["FAISS search<br/>top_k, score_threshold"]
    SEARCH --> HITS{"Any chunks<br/>above threshold?"}
    HITS -->|"no"| ABSTAIN["Abstention prompt<br/>no sources attached"]
    HITS -->|"yes"| BUILD["Build grounded prompt<br/>[SOURCE N] labels"]
    BUILD --> LLM["Generator — provider facade"]
    ABSTAIN --> LLM
    LLM --> ANS["Answer + sources_used<br/>+ query_script + model"]
```

The round trip, step by step:

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant UI as Streamlit UI
    participant API as FastAPI /query
    participant N as Normalizer
    participant E as BGE-M3
    participant V as FAISS
    participant P as Prompter
    participant G as Generator
    U->>UI: నేను ఎవరు?
    UI->>API: POST /query {query, top_k, provider}
    API->>API: validate provider + index
    API->>N: detect_script() + normalize()
    N-->>API: normalized query + script
    API->>E: embed_query()
    E-->>API: 1024-d vector
    API->>V: search(top_k, score_threshold)
    V-->>API: RetrievedChunk list
    API->>P: build_prompt(query, chunks, script)
    P-->>API: PromptPackage (system + user prompt)
    API->>G: generate(package, provider)
    G-->>API: GeneratedResponse
    API-->>UI: answer · sources_used · provider · model
    UI-->>U: grounded answer + citations
```

### The two content languages

Normalization is **script-aware**, not one-size-fits-all — this is the part that decides whether a
query is transliterated or left alone:

| Detected script | Trigger | Handling |
| --- | --- | --- |
| `telugu` | Telugu codepoints > 60% of letters | Unicode repair only — already native script |
| `roman` + English stopwords | ≥ 2 English function words | **No transliteration** — protects "who am I" from being mangled |
| `roman` (not English) | otherwise | Phrase map, then the abugida-aware state machine |
| `mixed` | neither threshold met | Full transliteration to Telugu |
| `unknown` | no letters at all | Passed through |

The English-stopword guard exists because transliterating `who am i` syllable-by-syllable produces
garbage. `detect_script()` runs on the **raw** query, before normalization, so the decision is never
based on already-mutated text.

> **Note on the abugida rules.** Telugu consonants carry an inherent `a`, and any other vowel is a
> dependent sign attached to that consonant — so a flat letter-by-letter substitution cannot work.
> The state machine emits matras after consonants, a virama in clusters, and anusvara for word-final
> `m`, which is what turns `eruka` into `ఎరుక` rather than `ఎరఉకఅ`. Plain `t`/`d`/`n` follow the
> ISO 15919 dental default. Spellings that are not a consistent Romanization (the gold set's `emiti`
> for ఏమిటి) are not recoverable by rules — see [Known limitations](#known-limitations).

---

## Scanned documents (OCR)

A PDF with no text layer — a photograph of a page — produces no chunks on its own. VāṇiVerse now
falls back to OCR instead of returning nothing. Rasterizing costs nothing extra: PyMuPDF already
returns raw pixels, so no image is written to disk. The recovered text re-enters the same
`normalize()` step, which means the transliteration and Unicode repair above apply to OCR output too.

Engines are pluggable and **only Telugu-capable ones are registered**. Support was verified against
upstream sources rather than assumed: Tesseract ships `tel.traineddata` in the official `tessdata_fast`
repo, and EasyOCR ships `te_char.txt`. PaddleOCR has no Telugu dictionary in `ppocr/utils/dict` and
Surya's repository has no Telugu entry, so neither is offered. The GPU OCR models that would likely do
much better on degraded scans cannot co-reside with BGE-M3 and `qwen2.5:3b` inside 4 GB of VRAM, so
they sit behind the same interface as a documented upgrade rather than a shipped default.

| Corpus | Tesseract CER | EasyOCR CER |
| --- | --- | --- |
| Expert-verified Telugu text blocks (n=50) | **0.137** | 0.238 |
| Real book page scans (n=6) | 0.489 | **0.477** |

Tesseract is the default **on measurement, not preference**: clearly better on pre-segmented text,
statistically tied on full pages, and 3.3× faster per page (5.0 s vs 16.3 s) on CPU. The blocks-versus-
pages gap is the honest headline — segmentation and 19th-century print cost far more than the choice
of recognizer does. Full detail, including why neither engine is good enough for faithful
transcription of old print, is in [docs/OCR.md](docs/OCR.md).

---

## Component map

| Module | Responsibility | Key entry points |
| --- | --- | --- |
| `api/main.py` | HTTP orchestration; startup singletons for store + generator | `lifespan`, 6 endpoints |
| `core/config.py` | Single source of truth for env-driven settings | `LLM_PROVIDER`, `FAISS_*_PATH`, `TOP_K`, `SCORE_THRESHOLD` |
| `core/exceptions.py` | Typed domain errors with structured `details` | `VaaniverseBaseError` + 12 subclasses |
| `ingestion/parser.py` | PDF → pages with layout metadata; OCR fallback for scans | `parse_document()` |
| `ingestion/ocr.py` | Pluggable Telugu OCR for image-only pages | `ocr_image()`, `resolve_engine()`, `available_engines()` |
| `ingestion/normalizer.py` | Script detection, Unicode repair, transliteration | `detect_script()`, `normalize()`, `repair_telugu_unicode()` |
| `ingestion/chunker.py` | Sentence-aware packing with overlap + metadata | `chunk_document()`, `chunk_page()` |
| `retrieval/embeddings.py` | Lazy-loaded BGE-M3 dense encoder | `get_model()`, `embed_chunks()`, `embed_query()` |
| `retrieval/vector_store.py` | Backend-agnostic store; FAISS + Qdrant adapters | `VectorStore`, `FAISSVectorStore`, `QdrantVectorStore` |
| `reasoning/prompter.py` | Evidence-only prompts with `[SOURCE N]` labels | `build_prompt()`, `build_no_result_prompt()` |
| `reasoning/generator.py` | Provider-agnostic LLM facade | `Generator`, `OllamaProvider`, `GroqProvider` |
| `frontend/app.py` | Streamlit UI + embedded-API bootstrap | `ensure_api_ready()`, `query_api()` |
| `scripts/*` | Batch ingestion, benchmarking, evaluation | see [CLI](#cli) |

---

## API reference

| Method | Endpoint | Purpose | Failure modes |
| --- | --- | --- | --- |
| `GET` | `/health` | Liveness + index size + active provider + `groq_configured` | `503` store uninitialised |
| `POST` | `/ingest` | Upload a PDF → parse → chunk → embed → persist | `422` not a PDF / unparseable / zero chunks · `500` save, chunk, or index failure |
| `POST` | `/query` | End-to-end RAG with citations | `400` unknown provider · `503` Groq key missing or index empty · `502` LLM call failed |
| `GET` | `/search` | Retrieval only — no LLM, useful for debugging recall | `503` index empty · `500` retrieval failure |
| `GET` | `/index/stats` | Vector count and readiness | — |
| `DELETE` | `/index/reset` | Drop the in-memory index | — |

<details>
<summary><b>Example — <code>POST /query</code></b></summary>

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "నేను ఎవరు?",
        "top_k": 5,
        "score_threshold": 0.45,
        "provider": "groq"
      }'
```

```jsonc
{
  "answer": "…grounded answer with [SOURCE 3] citations…",
  "query": "నేను ఎవరు?",
  "query_script": "telugu",
  "model": "qwen/qwen3.8-27b",
  "provider": "groq",
  "backend": "groq",
  "sources_used": [
    { "chunk_id": "book_p1_0", "source": "book.pdf",
      "page_number": 1, "score": 0.995, "retrieval_score": 0.6207,
      "script": "telugu" }
  ],
  "total_chunks_retrieved": 5,
  "reranked": true
}
```

</details>

The API also serves interactive docs at **`/docs`** (OpenAPI/Swagger) whenever it is running.

---

## Choosing an LLM provider

**Ollama is the default. Groq is strictly opt-in, per request.**

| Rule | Behaviour |
| --- | --- |
| Default | `LLM_PROVIDER` (env), falling back to `ollama` |
| Legacy alias | `LLM_BACKEND` is still read if `LLM_PROVIDER` is unset |
| Per-request override | `provider` field in `POST /query` (`"ollama"` or `"groq"`) |
| Unknown provider | `400`, before any work is done |
| Groq without a key | `503`, with a message naming `GROQ_API_KEY` |
| Startup safety | Providers are built lazily and never touch the network at construction, so a missing key or an offline Ollama **cannot** crash the server — it fails only when that provider is actually asked to generate |

The API key is read from the environment only. It is never hardcoded, never logged, and never
committed — `.env.example` ships an empty placeholder.

---

## Configuration

Every setting is environment-driven (`core/config.py`). Copy `.env.example` → `.env` and edit.

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `groq` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `llama3` | `.env.example` ships `qwen2.5:3b` (fits a 4 GB GPU) |
| `OLLAMA_TEMPERATURE` | `0.2` | |
| `GROQ_API_KEY` | *(empty)* | Supply via environment/secret store only |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq retires model ids; list current ones in their console |
| `GROQ_MAX_TOKENS` / `GROQ_TEMPERATURE` | `1024` / `0.2` | |
| `VECTOR_BACKEND` | `faiss` | `faiss` \| `qdrant` |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | |
| `EMBEDDING_DEVICE` | `cpu` | Set `cuda` for GPU ingestion/query |
| `EMBEDDING_BATCH_SIZE` | `32` | |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` | `localhost` / `6333` / `vaaniverse` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Tokens |
| `SPACY_MODEL` | `en_core_web_sm` | Telugu path is rule-based and does not need it |
| `TOP_K` / `SCORE_THRESHOLD` | `5` / `0.45` | Retrieval defaults, overridable per request |
| `RERANK_ENABLED` | `false` | Second-stage cross-encoder. **Off by default** — ~34 s/query on CPU |
| `RERANK_MODEL` / `RERANK_DEVICE` | `BAAI/bge-reranker-v2-m3` / `cpu` | Device pinned to CPU so it stays off the embedder's GPU |
| `RERANK_CANDIDATES` / `RERANK_THRESHOLD` | `20` / `0.5` | Candidates retrieved before reranking; threshold is **not calibrated** yet |
| `OCR_ENABLED` / `OCR_ENGINE` | `true` / `auto` | OCR runs only on pages with no usable text layer; `auto` = Tesseract, EasyOCR fallback |
| `OCR_LANG` / `OCR_DPI` / `OCR_MIN_CHARS` | `te,en` / `200` / `16` | Telugu + English; below `OCR_MIN_CHARS` a page counts as image-only |
| `API_HOST` / `API_PORT` / `API_RELOAD` | `0.0.0.0` / `8000` / `true` | |

Built-in art-direction (dark "ink & turmeric" theme) lives in `.streamlit/config.toml`.

---

## Quickstart

**Prerequisites:** Python 3.12 or newer (developed and tested on 3.14), and either
[Ollama](https://ollama.com) (local, default) or a
[Groq API key](https://console.groq.com) (hosted).

```bash
git clone https://github.com/sohansourab/vaaniVerse.git
cd vaaniVerse

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
cp .env.example .env             # then edit if needed
```

**1 — Start a model (for the default Ollama provider):**

```bash
ollama pull qwen2.5:3b
ollama serve
```

**2 — Build an index from your own PDF:**

```bash
python -m scripts.ingest_documents --input path/to/book.pdf
```

The index is written to `data/vectors/`. The first run downloads BGE-M3 (~2.2 GB) from Hugging Face.

**3 — Run the app:**

```bash
streamlit run frontend/app.py
```

That is enough. If no API is listening on port 8000, the UI starts one itself and the sidebar turns
**Online**. To run the backend separately instead:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000   # then: streamlit run frontend/app.py
```

Open <http://localhost:8501> and ask away. Sidebar controls: **Top K**, **Score threshold**,
**Filter by source**, **LLM provider**, and PDF upload → **Ingest document**.

### CLI

```bash
# Batch-ingest a file or a whole directory, with optional dry run
python -m scripts.ingest_documents --input docs/ --chunk-size 512 --overlap 64 --dry-run

# Latency/recall benchmark against a running API
python -m scripts.benchmark_retrieval --help

# Answer quality: exact match, F1, citation precision/recall
python -m scripts.evaluate_rag --help
```

### Docker

```bash
docker build -t vaaniverse .
docker run -p 8000:8000 -v vaaniverse-data:/app/data vaaniverse
```

The image is a two-stage `python:3.12-slim` build that runs as non-root (`appuser`, uid 10001) and
honours `$PORT`. Mount `/app/data` to persist the index — without it, ingestions vanish on restart.

> **Windows note:** non-ASCII CLI output needs UTF-8. Run `$env:PYTHONUTF8=1` (PowerShell) before
> Python commands.

---

## Project structure

```text
vaaniVerse/
├── api/
│   └── main.py                 # FastAPI app, endpoints, lifespan singletons
├── core/
│   ├── config.py               # environment-driven settings
│   └── exceptions.py           # typed domain errors
├── ingestion/
│   ├── parser.py               # PDF → pages (+ layout metadata)
│   ├── normalizer.py           # script detection · Unicode repair · transliteration
│   └── chunker.py              # sentence-aware, overlap-aware chunking
├── retrieval/
│   ├── embeddings.py           # BGE-M3 dense encoder (1024-d, L2-normalised)
│   └── vector_store.py         # VectorStore facade + FAISS / Qdrant backends
├── reasoning/
│   ├── prompter.py             # source-grounded prompt construction
│   └── generator.py            # provider facade — Ollama | Groq
├── frontend/
│   └── app.py                  # Streamlit UI + embedded-API bootstrap
├── scripts/
│   ├── ingest_documents.py     # batch ingestion CLI
│   ├── benchmark_retrieval.py  # retrieval latency benchmark
│   └── evaluate_rag.py         # EM / F1 / citation metrics
├── eval/
│   ├── gold_qa.jsonl           # gold set: 30 in-corpus + 10 out-of-domain questions
│   └── run_eval.py             # retrieval · grounding · latency evaluation
├── tests/
│   ├── test_api_providers.py   # provider routing at the HTTP layer
│   └── test_llm_providers.py   # provider behaviour in isolation
├── docs/
│   ├── ARCHITECTURE.md         # full design rationale
│   ├── DESIGN-NOTES.md         # long-form design rationale
│   ├── EVALUATION.md           # measured results and methodology
│   └── ROADMAP.md              # milestones and exit criteria
├── .streamlit/config.toml      # UI theme
├── Dockerfile                  # two-stage, non-root container
├── requirements.txt            # runtime deps
├── requirements-dev.txt        # pytest
└── .env.example                # config template (no secrets)
```

`data/` is **gitignored** and created at runtime — it holds uploaded documents and the vector index.
The `.pkl` metadata file embeds source text, so it is deliberately never committed.

---

## Evaluation

Measured on the local stack — 40 gold questions (30 in-corpus across Telugu, romanized Telugu and
English; 10 out-of-domain probes), `qwen2.5:3b` + BGE-M3 + FAISS, `top_k=5`.

| Layer | Metric | Result |
| --- | --- | --- |
| Retrieval | **Recall@5** | **0.900** (random baseline 0.152) |
| Retrieval | MRR | **0.686** |
| Retrieval | Latency (embed + search) | **277 ms** p50 |
| Retrieval | Latency with reranking on | **~34 s** p50 — CPU-only, off by default |
| Retrieval | In-corpus queries clearing the 0.45 gate | 76.7% |
| Retrieval | Out-of-domain queries clearing the 0.45 gate | **0/10 (0%)** |
| Grounding | Out-of-domain questions refused | **10/10 (100%)** ¹ |
| Grounding | Grounded answer rate | **56.7%** (17/30) ¹ |
| Grounding | False abstention on in-corpus questions | 30% ¹ |
| End-to-end | Latency | 6.2 s p50 · 30.8 s p95 ¹ |

Recall by query language: English **1.00** · Telugu **0.85** · romanized Telugu **1.00**.

¹ Measured **before** the romanized-transliteration fix, and not yet re-run — these rows will move
(fewer false abstentions) once they are. Every retrieval-layer row, including both gate rates, was
re-measured after it. See [docs/EVALUATION.md](docs/EVALUATION.md) for the before/after detail.

**Cross-encoder reranking is implemented, tested, and available — but off by default and not
measured end-to-end.** `RERANK_ENABLED=true` adds a second stage that re-scores the top
`RERANK_CANDIDATES` (20) passages with `BAAI/bge-reranker-v2-m3` before the top 5 are returned. It
is unit-tested (17 tests, offline), verified to reorder results while preserving the dense score
separately as `retrieval_score`, and sanity-checked on the real corpus (`నేను ఎవరు?` scores the
relevant passage 0.3666 against 0.0000 for an unrelated one). It is off by default for two measured
reasons — **~34 s per query** on CPU, since each candidate is a full ~512-token chunk, and a second
~2.3 GB model that does not fit next to BGE-M3 on Streamlit Cloud. **No reranked recall figure is
claimed:** the before/after run was lost twice to a Windows uvicorn accept-loop fault and cut for
time, and `RERANK_CANDIDATES=20` covers ~61% of a 33-chunk corpus, so a lift here would be easy to
inflate anyway. See [docs/EVALUATION.md](docs/EVALUATION.md) for how to reproduce it.

**Honest reading.** Retrieval is solid and works *cross-lingually* — an English question reliably
surfaces Telugu passages. Out-of-domain questions are refused rather than answered: zero
hallucinations across 10 probes, including the fabrication traps "Who is the author of this book?"
and "What is the price of this book?". The cost is a conservative system — the `0.45` score threshold
also stops **23.3%** of genuine questions at the retrieval gate. And generation quality, not plumbing,
is the ceiling at 3B parameters.

```bash
python eval/run_eval.py --skip-query     # retrieval only, ~30 s, no LLM calls
python eval/run_eval.py                  # full run, ~20 min on a GTX 1650
```

Full methodology, per-query results, miss analysis and **threats to validity**:
**[docs/EVALUATION.md](docs/EVALUATION.md)**.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
# 35 passed
```

The suite covers the provider contract and normalization end to end without needing Ollama, Groq, or
a GPU:

- default provider is `ollama`; a per-request override is honoured
- an unknown provider returns **400** with no downstream work
- `groq` without a key returns **503** rather than crashing
- `/health` reports `llm_default` and `groq_configured` accurately
- Ollama/Groq providers raise typed errors on empty responses and connection failures

---

## Known limitations

These are real and worth knowing before you rely on the system:

1. **OCR is a fallback, not a cure.** Image-only pages are now transcribed instead of yielding nothing
   (see [docs/OCR.md](docs/OCR.md)), but Tesseract on real Telugu book scans is weak, and its accuracy
   is well below what a current GPU model would give. It also needs the optional OCR dependencies and
   the Telugu language data; without them the parser degrades to text-only rather than failing. A
   scanned PDF with no engine installed still yields zero chunks and `422`.
2. **PDF extraction quality caps retrieval quality.** Telugu extracted from many PDFs contains
   spacing artifacts (e.g. `మ ైన` for `మైన`). Normalization repairs structural damage, but it cannot
   recover text that was never extracted correctly. This measurably weakens embeddings.
3. **The local default model is small.** `qwen2.5:3b` produces repetitive, awkward Telugu. Wiring is
   correct; generation quality is a model-size question. `qwen/qwen3.8-27b` via Groq is
   substantially better.
4. **Qdrant is implemented but not integration-tested.** FAISS is the tested path.
5. **FAISS is in-memory and single-process.** No concurrency story, no scale-out, and no native
   metadata filtering — `filter_by` is honoured on Qdrant only and ignored on FAISS.
6. **Ingestion appends.** Both the CLI and `POST /ingest` add to an existing index; re-ingesting the
   same PDF duplicates vectors. Use `DELETE /index/reset` (in-memory only) or remove
   `data/vectors/*` and rebuild.
7. **The index does not travel with the repo.** `data/` is gitignored, so any hosted deployment starts
   with an empty index and must ingest before it can answer.
8. **Streamlit Community Cloud is a tight fit.** No GPU, modest RAM, ephemeral disk. BGE-M3 needs
   ~2.2 GB in fp32, and the index must be rebuilt after each cold start.
9. **BGE-M3 loads lazily.** The first ingest or query pays the model-load cost (minutes on CPU, and a
   one-time ~2.2 GB download). The UI shows a spinner; timeouts are raised accordingly.

---

## Roadmap

| Phase | Focus | State |
| --- | --- | --- |
| **1 · Foundation** | Multilingual RAG pipeline, FastAPI, Streamlit, FAISS, provider switch, tests | ✅ Complete |
| **2 · Evaluation** | 40-question gold set, recall@k, MRR, per-language and threshold analysis | ✅ Complete |
| **3 · Robustness** | Transliteration, reranking, richer PDF layout handling, token estimation, OCR, provenance | 🔄 In progress |
| **4 · Cultural intelligence** | Inscription understanding, historical documents, heritage knowledge graph, community workflows | Planned |
| **5 · Indic expansion** | Hindi, Tamil, Kannada, Sanskrit, cross-language retrieval | Planned |
| **6 · Production readiness** | Telemetry, health checks, rate limits, admin tooling, deployment | Planned |

[docs/ROADMAP.md](docs/ROADMAP.md) is the source of truth — detailed tasks, exit criteria and status
for each milestone live there, and this table is its summary.

---

## Documentation

| Document | Contents |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Clean-architecture design, layer contracts, Indic-language policy, evaluation design |
| [docs/DESIGN-NOTES.md](docs/DESIGN-NOTES.md) | Long-form design rationale (architecture blueprint, multilingual trade-offs, evaluation metrics) |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Measured retrieval / grounding / latency results, methodology, and threats to validity |
| [docs/OCR.md](docs/OCR.md) | OCR for scanned pages: verified engine support, datasets, measured CER/WER, limitations |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestone-based plan with engineering priorities and exit criteria |
| [DEPLOY.md](DEPLOY.md) | Streamlit Cloud deployment, secrets, provider behaviour, troubleshooting |
| [VERIFY.md](VERIFY.md) | Verified local run: commands and observed output |
| [ANALYSIS.md](ANALYSIS.md) | Repository analysis and endpoint verification |

---

## Contributing

Contributions are welcome from developers, researchers, linguists, and anyone working on cultural
preservation.

1. Fork the repository and create a feature branch.
2. Keep the boundaries intact — new storage or model backends belong behind `VectorStore` /
   `Generator`, not in `api/main.py`.
3. Run `pytest -q` before opening a pull request.

Especially useful: Indic NLP, OCR, multilingual retrieval, knowledge graphs, and historical document
processing.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## Acknowledgements

[BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3) · [FastAPI](https://fastapi.tiangolo.com) ·
[Streamlit](https://streamlit.io) · [PyMuPDF](https://pymupdf.readthedocs.io) ·
[FAISS](https://github.com/facebookresearch/faiss) · [Qdrant](https://qdrant.tech) ·
[Groq](https://groq.com) · [Ollama](https://ollama.com)

<div align="center">
<br/>
<em>Preserving India's knowledge systems through multilingual AI.</em>
</div>
