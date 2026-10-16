# VERIFY.md — Local GPU run, provider switch, and end-to-end checks

Machine: Windows, Python **3.14.6** in `.venv` (README targets 3.12).
All commands below were run from the repo root with the venv Python.

> PowerShell note: several Python log lines go to stderr, so the harness prints
> `Exited with code 1` for commands that actually succeeded. The **real** exit
> codes are shown explicitly (e.g. `INGEST_EXIT:0`, `PYTEST_EXIT:0`).

---

## Phase 1 — Environment & GPU

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
```

**Result**

| Item | Value |
|------|-------|
| GPU | **NVIDIA GeForce GTX 1650** |
| VRAM | **4096 MiB (4 GB)** |
| Driver / CUDA UMD | 610.88 / 13.3 |

Initial torch was **CPU-only** (`2.14.1+cpu`, `cuda_available False`). Fixed by
force-installing the CUDA 13.0 wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.14.1" `
  --index-url https://download.pytorch.org/whl/cu130
```

After install:

```
torch 2.14.1+cu130
cuda_available True
device NVIDIA GeForce GTX 1650
cuda_version 13.0
matmul_ok True          # 1000x1000 matmul executed on the GPU
```

`streamlit` was added to `requirements.txt` and installed (`streamlit-1.66.0`).
Dev-only test deps live in `requirements-dev.txt` (`pytest-9.1.1`).

---

## Phase 2 — Ollama (default provider)

```powershell
ollama list
ollama run qwen2.5:3b "hello"
ollama ps
```

* Starting state: only `qwen2.5:7b` (4.7 GB) installed — too big for 4 GB VRAM.
  Per decision, pulled **`qwen2.5:3b`** (1.9 GB).
* `ollama run qwen2.5:3b "hello"` → *"Hello! How can I assist you today?..."*, exit 0.
* `ollama ps` →

```
NAME          ID              SIZE      PROCESSOR    CONTEXT    RUNNER
qwen2.5:3b    357c53fb659c    2.2 GB    100% GPU     4096       llamacpp
```

`.env` (git-ignored) sets: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://localhost:11434`,
`OLLAMA_MODEL=qwen2.5:3b`, `VECTOR_BACKEND=faiss`, `EMBEDDING_DEVICE=cuda`,
`EMBEDDING_BATCH_SIZE=16`. **No `GROQ_API_KEY`.**

**API starts with no Groq key** (verified with `GROQ_API_KEY` removed from the env):

```
GET /health -> 200
{"status":"ok","total_vectors":0,"vector_backend":"faiss",
 "llm_backend":"ollama","llm_default":"ollama","groq_configured":false}
```

> Note: after the Phase-4 refactor, startup no longer touches any LLM backend, so
> the API starts even if Ollama is down or `GROQ_API_KEY` is unset.

---

## Phase 3 — Ingest the book on GPU

Dry run:

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input "9.nenu-evarini.pdf" --dry-run
```

```
Found 1 PDF(s) to ingest.
  Parsing   -> 9.nenu-evarini.pdf
  Chunking  -> 9.nenu-evarini.pdf (33 pages)
  Chunks    -> 33 produced
  Dry run   -> skipping embed + index
  OK 9.nenu-evarini.pdf - 33 pages, 33 chunks [0.4s]
```

Real ingestion:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input "9.nenu-evarini.pdf"
```

```
Loading BGE-M3 model 'BAAI/bge-m3' on device 'cuda'...      <-- embeddings ran on the GPU
BGE-M3 model loaded successfully.
Embedding batches: 100%|##########| 2/2 [01:07<00:00, 33.58s/batch]
FAISS: added 33 vectors. Total: 33
OK 9.nenu-evarini.pdf - 33 pages, 33 chunks [17m 11.7s]
FAISS index saved -> ...\data\vectors\vaaniverse.faiss
Metadata saved    -> ...\data\vectors\vaaniverse_meta.pkl
INGEST_EXIT:0
```

* **Pages/chunks/vectors: 33 / 33 / 33.**
* **Duration 17m 11.7s**, of which **~15m 50s was the one-time `BAAI/bge-m3`
  download** (30 files); actual GPU embedding was **~1m 7s**.
* **Device: `cuda`** (logged by `retrieval/embeddings.py`). Confirmed by GPU
  usage during the run.

Index files:

| File | Size |
|------|------|
| `data/vectors/vaaniverse.faiss` | 132 KB |
| `data/vectors/vaaniverse_meta.pkl` | 86 KB |

These are git-ignored (`data/`).

### Clean rebuild (model already cached)

Because the CLI **appends** to an existing index, a second run would produce
duplicates. To measure a duplicate-free run with the model cached, the index was
backed up, removed, and rebuilt:

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input "9.nenu-evarini.pdf"
```

```
Loading BGE-M3 model 'BAAI/bge-m3' on device 'cuda'...
Embedding batches: 100%|##########| 2/2 [01:03<00:00, 31.79s/batch]
FAISS: added 33 vectors. Total: 33
OK 9.nenu-evarini.pdf - 33 pages, 33 chunks [1m 9.1s]
INGEST_EXIT:0        # wall clock: 1m 30s (includes model load from cache)
```

* **33 chunks / 33 vectors**, device **`cuda`**, **1m 9.1s** (first GPU batch ~62s
  includes CUDA/torch warm-up; second batch ~1.3s).

---

## Phase 4 — Provider switch

Where the LLM call happens (`reasoning/generator.py`, before the change): a single
`Generator` picked one backend from `LLM_BACKEND` at startup and delegated to
`GroqGenerator.generate` (`client.chat.completions.create(...)`) or
`OllamaGenerator.generate` (`httpx.post(f"{base}/api/chat", ...)`).
`api/main.py` called `generator.generate(prompt)` in `/query`.

After the change: `LLMProvider` interface + `OllamaProvider` / `GroqProvider`,
selected per request. See the "Files changed" section of the final summary.

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```
...............                                                          [100%]
15 passed, 1 warning in 15.91s
PYTEST_EXIT:0
```

The 1 warning is a Starlette/httpx deprecation notice (non-blocking). Tests are
fully mocked — no Ollama, Groq, GPU, or network access.

---

## Phase 5 — End-to-end checks (live API)

Started with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Endpoint results

| # | Check | Result | Verdict |
|---|-------|--------|---------|
| 1 | `GET /health` | `200` `{total_vectors:33, llm_default:"ollama", groq_configured:false}` | ✅ |
| 2 | `GET /index/stats` | `200` `{total_vectors:33, index_ready:true}` | ✅ |
| 3 | `GET /search` Telugu *"నేను ఎవరు? ... ఆత్మ విచారణ"* | 3 hits; **top = page 1, score 0.7513** | ✅ relevant |
| 4 | `GET /search` English *"Who am I? What is self inquiry?"* | 3 hits; **top = page 1, score 0.6577** | ✅ relevant |
| 5 | `GET /search` Telugu *"శరీరం నేను కాదు"* | 3 hits; **top = page 11, score 0.6523** | ✅ relevant |
| 6 | `POST /query` no `provider` | `200`; `provider:"ollama"`, `model:"qwen2.5:3b"`, 3 chunks, cites `[SOURCE 1]` | ✅ |
| 7 | `POST /query` `provider:"groq"`, no key | **`503`** `{"detail":"Provider 'groq' selected but GROQ_API_KEY is not configured on the server."}` | ✅ |
| 8 | `POST /query` `provider:"bogus"` | **`400`** `{"detail":"Invalid provider 'bogus'. Choose one of: ollama, groq."}` | ✅ |
| 9 | `POST /query` `provider:"groq"` with a real key | **SKIPPED** — `GROQ_API_KEY` was not set in the shell | ⏭️ |

### Check 6 raw output

```
provider: ollama | backend: ollama | model: qwen2.5:3b
query_script: telugu | chunks_retrieved: 3
sources: [('9.nenu-evarini.pdf', 1, 0.6207), ('9.nenu-evarini.pdf', 5, 0.5967), ('9.nenu-evarini.pdf', 4, 0.5773)]
answer:
 నేను ఎవరు? ... [SOURCE 1] ...
```

The answer is **source-grounded** (cites `[SOURCE 1]` = page 1) and returned via
the `ollama` provider. **Quality caveat:** qwen2.5:3b produces repetitive/awkward
Telugu, and the PDF's Telugu text carries OCR spacing artefacts — this is a model
quality limitation, not a wiring problem. A larger model (or Groq) is recommended
for production answer quality.

### VRAM during `/query`

With both models resident: `3434 MiB / 4096 MiB` used; `ollama ps` shows
qwen2.5:3b **100% GPU**. They fit, but headroom is small (~660 MiB) — if you add
longer contexts, reduce `EMBEDDING_BATCH_SIZE` or unload one side.

### Optional Streamlit UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.headless true --server.port 8501
```

Boots successfully — `GET http://127.0.0.1:8501/` → **HTTP 200**. The sidebar now
has an **LLM provider** dropdown (`ollama` / `groq`) whose value is sent as the
request's `provider` field.

---

## Failures, quirks, and workarounds

1. **`python scripts/ingest_documents.py` fails** with
   `ModuleNotFoundError: No module named 'core'` — the repo root isn't on
   `sys.path` when running the file directly. **Workaround:** use the module form
   `python -m scripts.ingest_documents ...` (this is what the script's own usage
   text documents). Not patched (works via `-m`).
2. **`--help` crashes on Windows** with
   `UnicodeEncodeError: 'charmap' codec ...` because the help text contains
   non-ASCII and the console is cp1252. **Workaround:** `$env:PYTHONUTF8=1`
   (applies to all ingestion runs).
3. **`fitz` deprecation warning** on every run (`ingestion/parser.py` uses the
   old `fitz` alias). Harmless; not changed.
4. **HF symlink warning** (Windows without Developer Mode) — harmless.
5. **`EXIT:0` vs "Exited with code 1"** — PowerShell treats native stderr as an
   error record; the real process exit codes were 0.
6. **Dockerfile not verified** — `docker` is not installed on this machine, so the
   image was not built. Validate with `docker build -t vaaniverse-api .` before
   deploying.
7. **Groq live path not exercised** (no key available). Unit tests cover the wired
   behaviour (200 with key, 503 without).

---

## Reproduce the working run

```powershell
# 1. venv + deps (CUDA torch)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 2. Ollama (default provider)
ollama serve
ollama pull qwen2.5:3b

# 3. Ingest (GPU)
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input "9.nenu-evarini.pdf"

# 4. Test + run
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# UI (optional):
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```
