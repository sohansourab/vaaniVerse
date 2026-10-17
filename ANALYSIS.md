# VāṇiVerse — Codebase Audit & Analysis

> Audit date: 2026-10-16 · Auditor: automated repository audit
> Scope: static reading of all tracked files + a live local install/run attempt.
> Legend: **[VERIFIED]** = confirmed by reading code or executing it · **[UNVERIFIED]** = not confirmed in this audit.

---

## 1. Inventory

### Top-level layout

| Path | Type | Purpose |
| --- | --- | --- |
| `api/` | dir | FastAPI application (`main.py`) |
| `core/` | dir | Config (`config.py`) and exception hierarchy (`exceptions.py`) |
| `ingestion/` | dir | PDF parsing, normalization/transliteration, chunking |
| `retrieval/` | dir | BGE-M3 embeddings + FAISS/Qdrant vector stores |
| `reasoning/` | dir | Prompt construction + Groq/Ollama LLM generation |
| `frontend/` | dir | Streamlit chat UI (`app.py`) |
| `scripts/` | dir | CLI: batch ingest, RAG evaluation, retrieval benchmark |
| `docs/` | dir | `ARCHITECTURE.md`, `ROADMAP.md` (design docs, not code) |
| `.streamlit/` | dir | Streamlit theme config (`config.toml`) |
| `README.md`, `summary.md`, `interview.md` | files | Documentation / notes |
| `requirements.txt` | file | Dependency list (unpinned) |
| `.env.example` | file | Example environment variables |
| `.gitignore` | file | Ignores `.venv/`, `.env`, `__pycache__/`, `indexes/`, `data/documents/`, `*.faiss`, `*.pkl` |
| `logo`, `logo.png` | files | Branding assets |

### Languages, frameworks, tooling

- **Language:** Python only (20 `.py` files: 14 real modules + 6 empty `__init__.py`). No JS/TS/Go/Rust.
  - `__init__.py` files present in `api/`, `core/`, `ingestion/`, `retrieval/`, `reasoning/`, `scripts/` (all empty). `frontend/` has **no** `__init__.py`. **[VERIFIED]**
- **Frameworks:** **FastAPI** (API, with Starlette + Uvicorn), **Pydantic v2** (request/response models), **Streamlit** (UI). **[VERIFIED]**
- **Package manager:** `pip` via `requirements.txt`. No `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, or `poetry.lock`. No version pins. The 18 declared deps: `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`, `python-multipart`, `pymupdf`, `Pillow`, `spacy`, `indic-nlp-library`, `FlagEmbedding`, `torch`, `sentence-transformers`, `faiss-cpu`, `qdrant-client`, `groq`, `httpx`, `numpy`, `tqdm`. **[VERIFIED]**
- **Build tools:** none (no `Makefile`, no build/package step, not a packaged library). **[VERIFIED]**
- **Notable:** `streamlit` is **not** listed in `requirements.txt` even though the frontend requires it. `indic-nlp-library` and `sentence-transformers` are declared but **never imported** anywhere in the codebase (dead dependencies that pull in `torch`, `datasets`, `scikit-learn`, `pandas`, `sphinx`, etc.). **[VERIFIED]**

### Module-by-module map (all read in full)

| Module | Responsibility |
| --- | --- |
| `api/main.py` (441 lines) | FastAPI app, lifespan, Pydantic schemas, 6 endpoints, error mapping |
| `core/config.py` (54) | Env-driven settings; creates `data/documents` + `data/vectors` at import |
| `core/exceptions.py` (50) | `VaaniverseBaseError` + 12 typed exceptions |
| `ingestion/parser.py` (165) | PyMuPDF text extraction (+fallback), per-page image extraction, calls normalizer |
| `ingestion/normalizer.py` (256) | Script detection, Unicode NFC + Telugu artifact repair, Romanized-Telugu transliteration |
| `ingestion/chunker.py` (250) | spaCy/rule-based sentence split, greedy sentence packing with overlap, `TextChunk` |
| `retrieval/embeddings.py` (270) | BGE-M3 dense embeddings (singleton), sparse optional, cosine utils |
| `retrieval/vector_store.py` (517) | `FAISSVectorStore`, `QdrantVectorStore`, unified `VectorStore` |
| `reasoning/prompter.py` (207) | Script-specific system prompts, source-block formatting, `PromptPackage` |
| `reasoning/generator.py` (289) | `GroqGenerator`, `OllamaGenerator`, unified `Generator` |
| `frontend/app.py` (487) | Streamlit chat UI + ingest widget (heavy custom CSS) |
| `scripts/ingest_documents.py` (311) | Batch CLI ingestion with dry-run/no-persist |
| `scripts/evaluate_rag.py` (169) | EM/F1/citation metrics from JSON/JSONL |
| `scripts/benchmark_retrieval.py` (96) | Latency benchmark against `/query` |

---

## 2. Entry points

| Entry point | Location | Run command |
| --- | --- | --- |
| **Backend API** | `api/main.py` → `app = FastAPI(...)`, `lifespan()` | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| API (module form) | `api/main.py` `__main__` block | `python api/main.py` → runs `uvicorn.run("api.main:app", ...)` with **hardcoded `reload=True`** (ignores `API_RELOAD`) **[VERIFIED]** |
| **Frontend UI** | `frontend/app.py` (top-level Streamlit script) | `streamlit run frontend/app.py` |
| Ingestion CLI | `scripts/ingest_documents.py` | `python -m scripts.ingest_documents --input <file-or-dir> [--dry-run] [--chunk-size N] [--overlap N] [--backend faiss\|qdrant] [--no-persist]` |
| Evaluation CLI | `scripts/evaluate_rag.py` | `python -m scripts.evaluate_rag --results <json\|jsonl> [--output <path>]` |
| Benchmark CLI | `scripts/benchmark_retrieval.py` | `python scripts/benchmark_retrieval.py --dataset <jsonl> --api-url http://localhost:8000` |

- Backend host/port come from `API_HOST`/`API_PORT` (`0.0.0.0:8000` defaults) but only when launched via `python api/main.py`; the `uvicorn` CLI form ignores them unless passed explicitly. **[VERIFIED]**
- Frontend has a **hardcoded** `API_BASE_URL = "http://localhost:8000"` (no env var). **[VERIFIED]**

---

## 3. Architecture (end-to-end request flow)

The README describes a "clean architecture" (domain/application/infrastructure/interface) but the code is a **flat, module-by-concern package layout** (api/core/ingestion/retrieval/reasoning/frontend/scripts). Those clean-architecture layers do **not** exist as directories or classes. **[VERIFIED]**

### Query flow (`POST /query`)

```
Streamlit UI (frontend/app.py)  ──HTTP──▶  FastAPI (api/main.py::query)
   │                                            │
   │ POST /query {query, top_k, score_threshold, filter_source}
   │                                            ▼
   │                              _normalize_query()  →  ingestion/normalizer.detect_script() + normalize()
   │                                            ▼
   │                              VectorStore.query()  ──▶ retrieval/embeddings.embed_query()
   │                                                        (BGE-M3 dense 1024-d)
   │                                            ▼
   │                              backend.search()  (FAISSVectorStore | QdrantVectorStore)
   │                                            ▼
   │                              reasoning/prompter.build_prompt()  → script-specific system prompt + [SOURCE n] blocks
   │                                            ▼
   │                              reasoning/generator.Generator.generate()  ──▶ Groq Chat API  |  Ollama /api/chat
   │                                            ▼
   └──────────────  QueryResponse {answer, query_script, model, backend, sources_used[]}
```

### Ingestion flow (`POST /ingest`)

```
Upload (multipart) ─▶ save to data/documents/ ─▶ parse_document() (PyMuPDF, normalize per page, extract images)
   ─▶ chunk_document() (script-aware sentence split + greedy overlap packing)
   ─▶ VectorStore.index_chunks() ─▶ embed_chunks() (BGE-M3) ─▶ backend.add() ─▶ FAISS save() (.faiss + .pkl)
```

### Main modules and what each does

- **`api/main.py`** — orchestration and HTTP surface. Singletons `_vector_store` / `_generator` created in the `lifespan` startup hook. Endpoints: `GET /health`, `POST /ingest`, `POST /query`, `GET /search`, `GET /index/stats`, `DELETE /index/reset`.
- **`ingestion/normalizer.py`** — the linguistic core: `detect_script` (telugu/roman/mixed/unknown by codepoint ratios), `unicode_normalize` (NFC), `repair_telugu_unicode` (zero-width, duplicate matras/diacritics, broken conjuncts, orphan virama), `transliterate_roman_to_telugu` (phrase map + syllable longest-match), and `is_english_query` (stopword heuristic ≥2 hits to avoid mangling English). **[VERIFIED working — see §7]**
- **`ingestion/chunker.py`** — `_estimate_tokens` (1.4× words for Telugu else 1.1×), Telugu danda-aware split vs spaCy `sentencizer`, greedy pack with overlap window. **Note:** the code is mis-indented in the source as read (the `_build_chunks` body appears at a different level); it executes correctly in practice (dry-run passed), so this is cosmetic.
- **`retrieval/embeddings.py`** — singleton `BGEM3FlagModel`, `embed_texts`/`embed_chunks`/`embed_query`, always `max_length=512`, dense-only indexing (sparse/ColBERT available but unused).
- **`retrieval/vector_store.py`** — `FAISSVectorStore` (IndexFlatIP, disk persistence), `QdrantVectorStore` (HTTP, metadata filters), unified `VectorStore` facade.
- **`reasoning/prompter.py`** — three system prompts (Telugu/English/Mixed) enforcing evidence-only answers + `[SOURCE n]` citations; context capped at 1500 chars/chunk.
- **`reasoning/generator.py`** — Groq (requires key) or Ollama (requires reachable server) backends.

---

## 4. Data layer

- **No relational database, no ORM, no migrations, no SQL schema.** Persistence is file-based. **[VERIFIED]**
- **Active vector store = FAISS `IndexFlatIP`** (inner product over L2-normalized BGE-M3 vectors = cosine similarity).
  - Index file: `data/vectors/vaaniverse.faiss`; metadata: `data/vectors/vaaniverse_meta.pkl` (a Python `pickle` of a list of dicts, positionally aligned with vectors). **[VERIFIED]**
- **Optional Qdrant** backend (`VECTOR_BACKEND=qdrant`), collection `vaaniverse`, COSINE distance, default `localhost:6333`. README marks it "implemented, untested". **[VERIFIED in code; runtime UNVERIFIED — no Qdrant instance in this audit]**
- **External APIs / services:**
  - **Groq** Chat Completions (`groq` SDK) — defaults `llama3-70b-8192` (code) vs `llama-3.3-70b-versatile` (`.env.example`). **[VERIFIED]**
  - **Ollama** local `POST /api/chat` (default `http://localhost:11434`, model `llama3`). **[VERIFIED in code]**
  - **Hugging Face Hub** — downloads `BAAI/bge-m3` on first embed. **[VERIFIED in code; download not performed]**
- **ML models:** `BAAI/bge-m3` dense embeddings (1024-dim; `EMBEDDING_DIM=1024`); spaCy `en_core_web_sm` with graceful fallback to `spacy.blank("en")` + `sentencizer` (Telugu uses rule-based splitting). Models are loaded **lazily**, cached per process. **[VERIFIED]**
- **Persisted artifacts are runtime-only:** `data/` is created at import; there is currently no `data/` content or index on disk in this checkout. `indexes/` is listed in `.gitignore` but the code actually uses `data/vectors/` — a minor inconsistency (`data/vectors/` itself is not ignored; only its `*.faiss`/`*.pkl` files are). **[VERIFIED]**

---

## 5. Config and secrets

### Config files
- `core/config.py` — reads env via `python-dotenv` `load_dotenv()`. **[VERIFIED]**
- `.env.example` — template only.
- `.streamlit/config.toml` — dark theme, `toolbarMode = "minimal"`.

### Environment variables

| Variable | Default in `core/config.py` | In `.env.example`? |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | yes |
| `EMBEDDING_DEVICE` | `cpu` | yes |
| `EMBEDDING_BATCH_SIZE` | `32` | yes |
| `VECTOR_BACKEND` | `faiss` | yes |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` | `localhost` / `6333` / `vaaniverse` | **no** |
| `LLM_BACKEND` | `groq` | yes |
| `GROQ_API_KEY` | `""` (empty) | yes (placeholder) |
| `GROQ_MODEL` | `llama3-70b-8192` | yes (`llama-3.3-70b-versatile`) |
| `GROQ_MAX_TOKENS` / `GROQ_TEMPERATURE` | `1024` / `0.2` | **no** |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_TEMPERATURE` | `http://localhost:11434` / `llama3` / `0.2` | **no** |
| `CHUNK_SIZE` | `512` | yes (`256`) |
| `CHUNK_OVERLAP` | `64` | yes (`32`) |
| `SPACY_MODEL` | `en_core_web_sm` | **no** |
| `TOP_K` | `5` | yes |
| `SCORE_THRESHOLD` | `0.45` | yes |
| `API_HOST` / `API_PORT` / `API_RELOAD` | `0.0.0.0` / `8000` / `true` | yes |

### Secrets assessment
- **No hardcoded secrets or API keys found anywhere in the repo.** `GROQ_API_KEY` is read from env with an empty default, and the code fails explicitly if it is missing (`GroqGenerator.__init__`). `.env` is gitignored; only `.env.example` (placeholder `your_groq_api_key_here`) is committed. **[VERIFIED]**
- Better-than-average hygiene: `.gitignore` excludes `.env`. **[VERIFIED]**
- Config drift to note: `.env.example` values differ from code defaults (`CHUNK_SIZE` 256 vs 512, `CHUNK_OVERLAP` 32 vs 64, `GROQ_MODEL` differs), and several supported variables are absent from the example. **[VERIFIED]**
- `API_RELOAD` is defined but never used (the `__main__` runner hardcodes `reload=True`). **[VERIFIED]**

---

## 6. Tests

- **No tests exist.** No `test_*.py`, `*_test.py`, `conftest.py`, `pytest.ini`, `tox.ini`, `pyproject.toml`, or `tests/` directory in the working tree or in git history. **[VERIFIED]**
- `pytest` is installed (v9.1.1) so running it is possible, but there is nothing to collect.
- Command run and result:

  ```
  $ python -m pytest -q
  no tests ran in 0.02s
  ```

- The README's Phase-1 claim of "Pre-commit quality gates: ruff, mypy, pytest" is **not backed by any config**: no `.pre-commit-config.yaml`, no `ruff`/`mypy` config, no `pyproject.toml`, and none of `ruff`/`mypy`/`pytest` appear in `requirements.txt`. **[VERIFIED]**
- The `docs/ROADMAP.md` itself lists "add a simple regression suite for normalization and transliteration behavior" as a suggested task, confirming this gap. **[VERIFIED]**

**Test status: NONE — cannot run.**

---

## 7. Run it locally

Environment used: Windows 11, **Python 3.14.6** (project README targets 3.12; install worked on 3.14).

### Commands that worked

```powershell
# 1. Create venv
python -m venv .venv

# 2. Install deps  → SUCCESS, all 18 requirements resolved & installed
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Import check → all modules import OK
.\.venv\Scripts\python.exe -c "import api.main; import retrieval.vector_store; import reasoning.generator"
#   (only warning: `fitz` API is deprecated, use `import pymupdf`)

# 4. CLI ingest (dry-run, no embedding/LLM needed) → SUCCESS
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input probe.pdf --dry-run
#   probe.pdf — 1 pages, 1 chunks [0.2s] ; exit 0

# 5. Evaluation CLI → SUCCESS
.\.venv\Scripts\python.exe -m scripts.evaluate_rag --results eval.jsonl
#   {"total_examples": 2, "exact_match": 0.5, "f1": 0.8333, ...}
```

### Endpoint verification (server started with a dummy `GROQ_API_KEY`)

| Request | Result |
| --- | --- |
| `GET /health` | `200` → `{"status":"ok","total_vectors":0,"vector_backend":"faiss","llm_backend":"groq"}` |
| `GET /index/stats` | `200` → `{"total_vectors":0,"vector_backend":"faiss","index_ready":false}` |
| `GET /search?q=rama` | `503` → `{"detail":"Index is empty. Ingest documents first."}` |
| `POST /query` (empty index) | `503` → `{"detail":"Index is empty. Ingest documents via POST /ingest first."}` |
| `POST /query` (blank query) | `422` → Pydantic `string_too_short` validation error |
| `POST /ingest` (non-PDF) | `422` → `{"detail":"Only PDF files are supported. Upload a .pdf file."}` |

### Errors / blockers encountered

1. **The API cannot start without an LLM backend.** With no `.env`, `uvicorn api.main:app` failed at startup:
   ```
   INFO:  Waiting for application startup.
   INFO:retrieval.vector_store:VectorStore initialized with backend: 'faiss'
   WARNING:api.main:No vector index found on disk. ...
   ERROR:    Traceback (most recent call last):
     File ".../api/main.py", line 63, in lifespan
       _generator = Generator()
     File ".../reasoning/generator.py", line 49, in __init__
       raise LLMInferenceError("GROQ_API_KEY is not set. Add it to your .env file.")
   ERROR:    Application startup failed. Exiting.
   ```
   There is **no degraded/offline startup path**: `LLM_BACKEND=groq` requires a key; `LLM_BACKEND=ollama` requires a reachable Ollama server (`OllamaGenerator._verify_connection`). A dummy non-empty `GROQ_API_KEY` was enough to boot, because the key is only validated on an actual `/query` call.
2. **`streamlit` is not in `requirements.txt`** — `python -m streamlit` in the venv → `No module named streamlit`. The frontend must install Streamlit separately.
3. `fitz` (PyMuPDF) deprecation warning on every run.
4. `python -m scripts.evaluate_rag` rejects files that begin with a UTF-8 BOM (`load_records` uses `encoding="utf-8"`, not `utf-8-sig`).

### Not verified in this audit

- **Full `/query` and `/ingest` (embed → index → LLM answer) — UNVERIFIED.** These require a **valid Groq API key** and would download the **~2 GB `BAAI/bge-m3`** model on first embed; neither was available/attempted.
- **Qdrant backend end-to-end — UNVERIFIED** (no Qdrant instance).
- **Ollama backend — UNVERIFIED** (no Ollama instance).
- **Streamlit UI rendering — UNVERIFIED** (Streamlit absent from the venv; not installed separately).
- **`scripts/benchmark_retrieval.py` — UNVERIFIED** (needs a live API + dataset).

---

## 8. Risks

### Performance hotspots
- **Blocking work inside `async def` handlers.** `query` and `ingest_document` are `async` but call synchronous, CPU-bound code (`embed_query`, `embed_chunks`, PyMuPDF parsing, spaCy) and blocking network calls (`groq` SDK, `httpx.post`). This blocks the event loop and serializes concurrent requests. **[VERIFIED]**
- **Whole upload read into memory.** `content = await file.read()` with **no size limit** — a large PDF is fully buffered in RAM. **[VERIFIED]**
- **Wasted image extraction during ingest.** `/ingest` calls `parse_document(..., extract_images=True)`, decoding every embedded image to a PIL object per page, but **nothing downstream ever reads `ParsedPage.images`** — pure wasted CPU/RAM. **[VERIFIED]** (`scripts/ingest_documents.py` correctly uses `extract_images=False`.)
- **O(n²) image scan.** `_extract_images_from_page` calls `page.get_image_info()` inside the loop over `get_images()`. **[VERIFIED]**
- **FAISS `IndexFlatIP` is exact brute-force** (O(N) per query). No ANN (IVF/HNSW) → poor scaling beyond modest corpora. **[VERIFIED]**
- **Cold-start latency.** BGE-M3 and spaCy load lazily on first use (model download + load), and the vector index is loaded at startup. First query after boot is slow; no warmup. **[VERIFIED]**
- **No caching / batching / queue.** Every query re-embeds; no embedding cache or request coalescing. **[VERIFIED]**
- **Large prompts.** `top_k` up to 20 × up to 1500 chars/chunk + system prompt; no token-budget enforcement before calling the LLM. **[VERIFIED]**
- **Non-streaming LLM responses** (`stream=False`, Ollama `stream:False`) — full answer buffered before return. **[VERIFIED]**

### Security issues
- **Path traversal on ingest.** `destination = DATA_DIR / file.filename` uses the client-supplied filename directly; a crafted name (e.g. `..\..\foo`) can write outside `data/documents/`. **[VERIFIED]**
- **No authentication/authorization** on any endpoint — `/ingest` (write), `/index/reset` (destructive), `/query` (costly) are open to anyone who can reach the service. **[VERIFIED]**
- **Permissive CORS:** `allow_origins=["*"]` **combined with** `allow_credentials=True`; methods/headers also `*`. **[VERIFIED]**
- **No rate limiting** → Groq spend / DoS exposure. **[VERIFIED]**
- **Insecure deserialization:** metadata is loaded via `pickle.load` — arbitrary code execution if the `.pkl` is tampered with (local-trust assumption). **[VERIFIED]**
- **No upload size/type hardening** beyond a `.pdf` extension check on the filename. **[VERIFIED]**
- **`/index/reset` leaves disk files** (`data/vectors/*.faiss`/`.pkl`) intact, so stale data can reappear on restart. **[VERIFIED]**

### Correctness bugs
- **`filter_source` is silently ignored on the default FAISS backend.** `api/main.py` builds `filter_by={"source": ...}`, but `VectorStore.query` only forwards it to Qdrant; FAISS receives no filter. Users can pass a filter and get unfiltered results. **[VERIFIED]**
- **Unstable Qdrant point IDs.** `id = abs(hash(chunk.chunk_id)) % 2**63` uses Python's per-process randomized string hash, so IDs differ across runs → duplicate points on re-ingest. **[VERIFIED]**
- **Document normalization transliterates on ingest** for every page (`transliterate=True`), including English documents; safe by design (phrase map only) but worth noting. **[VERIFIED]**

### Missing error handling
- **No global exception handlers** for `VaaniverseBaseError`; each route wraps calls in `try/except` individually. An unhandled exception in the lifespan hook crashes the whole app (as observed in §7). **[VERIFIED]**
- **Citation grounding is not validated** — the code trusts the LLM to emit valid `[SOURCE n]` labels; there is no post-hoc check that answers are supported by retrieved chunks. **[VERIFIED]**
- **No input guard for very large documents/text** in `normalize()`/chunking (memory blow-ups possible). **[VERIFIED]**
- Partial-ingest cleanup: a saved upload that later fails parsing/embedding is left on disk. **[VERIFIED]**

---

## 9. Deployment readiness

### Present
- `GET /health` endpoint (returns status + vector count + backends). **[VERIFIED]**
- Environment-driven configuration with a committed `.env.example`, and `.env` correctly gitignored. **[VERIFIED]**
- A `.gitignore`. **[VERIFIED]**
- Dependency list that resolves cleanly on Python 3.14 (verified), though README targets 3.12. **[VERIFIED]**

### Missing / gaps
- **No `Dockerfile`, `.dockerignore`, or `docker-compose.yml`.** **[VERIFIED]**
- **No CI** (no `.github/workflows`, no `.gitlab-ci.yml`, no equivalent). **[VERIFIED]**
- **No tests / no quality gates** despite README claiming ruff+mypy+pytest (see §6). **[VERIFIED]**
- **No `pyproject.toml`** (no project metadata, no console-script entry points, no tool config). **[VERIFIED]**
- **No pinned dependency versions** → non-reproducible builds; `requirements.txt` is a bare list. **[VERIFIED]**
- **No `LICENSE` file** even though the README states "Released under the MIT License." **[VERIFIED]**
- **Health endpoint is shallow** — no readiness vs liveness split, no dependency checks (LLM key, model warm, index state are only partially reflected). **[VERIFIED]**
- **Logging** is bare `logging.basicConfig(level=INFO)`: no structured/JSON logs, no request IDs, no access-log customization, no rotation, no log levels via env. **[VERIFIED]**
- **No metrics/tracing/observability** (no OpenTelemetry, Prometheus, or health dashboards), despite `docs/ROADMAP.md` Milestone 6 calling for it. **[VERIFIED]**
- **No auth, no rate limiting, permissive CORS** (see §8). **[VERIFIED]**
- **Startup hard-depends on an LLM backend** — a container without a Groq key or Ollama will crash-loop (observed). **[VERIFIED]**
- **Frontend API URL is hardcoded** to `http://localhost:8000` (not env-configurable). **[VERIFIED]**
- **`streamlit` missing from `requirements.txt`** → frontend not installable from the declared deps. **[VERIFIED]**

### Untracked artifacts noticed in the working tree (not created by this audit's intentional work)
- `9.nenu-evarini.pdf` (4.4 MB, untracked, **not** gitignored, mtime preserved from June) — appears to be user test data. Left untouched. Its presence means `git status` is not clean. **[VERIFIED it exists; provenance UNVERIFIED]**
- A `.venv/` and empty `data/documents/`, `data/vectors/` directories exist from this audit's install/run (`data/` is created by importing `core.config`). `.venv/` is gitignored. **[VERIFIED]**

---

### Summary of highest-impact findings
1. **App cannot start without `GROQ_API_KEY`** — no degraded/offline boot path (verified crash).
2. **`streamlit` missing from `requirements.txt`** — frontend dependency not declared.
3. **No tests, no Dockerfile, no CI, no LICENSE** despite README claims of quality gates + MIT.
4. **Path traversal + unauthenticated destructive endpoints + wildcard CORS** — security posture needs work.
5. **Blocking work in async handlers + brute-force FAISS + lazy model load** — performance/scaling concerns.
6. **`filter_source` silently ignored on FAISS**; **unstable Qdrant point IDs** — correctness bugs.
