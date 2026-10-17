# DEPLOY.md — Running and deploying VāṇiVerse

VāṇiVerse ships as **one Streamlit app**. The public instance is
**<https://vaaniverse-27.streamlit.app/>**.

There is no separately hosted API. `frontend/app.py` bootstraps `api.main:app`
(uvicorn) on a daemon thread inside its own process and then talks to it over
`http://127.0.0.1:8000` — see `ensure_api_ready()` at the bottom of that file.
That is what makes the whole system a single deployable unit, and it is also why
a fresh deploy needs nothing provisioned alongside it.

Everything the app needs is driven by environment variables. **No secret is ever
committed**: `.env` is gitignored and `.env.example` ships with an empty
`GROQ_API_KEY` placeholder.

---

## 1. Deploying to Streamlit Community Cloud

| Setting | Value |
|---------|-------|
| Repository | `sohansourab/vaaniVerse` |
| Branch | `main` |
| Main file path | `frontend/app.py` |
| Python version | 3.12+ (declared in `runtime.txt`/the Cloud UI) |
| Apt packages | `packages.txt` → `tesseract-ocr`, `tesseract-ocr-tel` |
| Python packages | `requirements.txt` |

Cloud builds `requirements.txt` at deploy time and installs the apt packages in
`packages.txt`. Nothing else is required to *boot* — the app is designed to start
with no corpus and no configured provider, and to tell you what is missing rather
than crashing.

### Secrets — the one step that matters

Open **App → Settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_..."   # required: nothing else can serve inference on Cloud
LLM_PROVIDER = "groq"      # optional, see below
```

The bridge at the top of `frontend/app.py` copies flat string secrets into the
environment *before* the embedded API is imported, because `core/config.py` reads
its configuration at import time. A secret that is not a plain top-level string
will not be picked up.

`GROQ_API_KEY` is the one that actually matters.

`LLM_PROVIDER` is belt-and-braces. `/health` reports `ollama_reachable`, and the
sidebar only lists providers the server can serve, so with no local Ollama the
default falls through to Groq on its own.

> **This is the fix for the most common failure.** With no `GROQ_API_KEY` the UI
> has no working provider, and the very first query fails with
> `API error: Ollama API call failed.` — because Ollama is the product default
> and there is no Ollama on Cloud. The message now names the endpoint it could
> not reach and the UI suggests switching provider.

`GROQ_MODEL` can override the model id. Groq retires ids on its own schedule and a
retired one returns `404 model_not_found`, which the API reports verbatim along
with the model it tried. The current default is in the Configuration table in
[README.md](README.md).

---

## 2. The corpus, and why you re-ingest after every cold start

The container disk is **ephemeral**: `data/` is wiped whenever the app sleeps,
restarts, or redeploys. `data/` is also gitignored, so the index is never in the
repository — deliberately, because `vaaniverse_meta.pkl` embeds source text and
the corpus is a licensed book.

So the deployed flow is:

1. Open the app.
2. In the sidebar, upload a Telugu PDF and click **Ingest document**.
3. Wait. The first ingest downloads `BAAI/bge-m3` (~2.2 GB) and embeds on **CPU**
   (there is no GPU on Cloud), then writes `data/vectors/`. Give it a few minutes.
4. Ask questions.

Subsequent ingests in the same container session are much faster — the model is
already in memory — but they do not survive a restart.

If you need a warm index without waiting, ingest once, then re-upload the same PDF
and use the same container session; a sleeping app will need a fresh ingest.

---

## 3. Environment variables

Set these as Streamlit secrets (Cloud) or in `.env` (local).

| Variable | Example | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` (default) or `groq` | Provider used when a request omits one. |
| `GROQ_API_KEY` | *(secret)* | Required for the `groq` provider. Empty ⇒ Groq requests return **503**. |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model id. |
| `GROQ_TEMPERATURE` | `0.2` | Sampling temperature. |
| `GROQ_MAX_TOKENS` | `1024` | Completion cap. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint. Probed by `/health` (1 s timeout, cached 30 s). |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model id. Fits a 4 GB GPU. |
| `VECTOR_BACKEND` | `faiss` | `faiss` (default, tested) or `qdrant`. |
| `EMBEDDING_DEVICE` | `cpu` | **Cloud has no GPU — keep `cpu`.** |
| `EMBEDDING_BATCH_SIZE` | `16` | Lower if memory-constrained. |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking. Off by default — see below. |
| `RERANK_DEVICE` | `cpu` | Pinned explicitly; the library claims every CUDA device otherwise. |
| `RERANK_THRESHOLD` | `0.5` | Abstention threshold applied to rerank scores. |
| `OCR_ENGINE` | `tesseract` | `tesseract` (default), `easyocr`, or `none`. |
| `TESSDATA_PREFIX` | `data/tessdata` | Telugu/English tessdata shipped in-repo. |

**Leave `RERANK_ENABLED` off on Cloud.** Reranking adds a second ~2.3 GB model
next to BGE-M3, which does not fit comfortably in the Cloud memory budget, and it
is CPU-only there. It is measured and available, not part of the hosted default.

> The API is designed to **start even when a backend is unavailable**. Selecting
> `groq` without `GROQ_API_KEY` returns `503` at request time; it never crashes
> boot. `GET /health` reports `llm_default`, `groq_configured` and
> `ollama_reachable` so configuration can be confirmed from outside.

---

## 4. `GET /health` — the configuration contract

```json
{
  "status": "ok",
  "total_vectors": 33,
  "vector_backend": "faiss",
  "llm_backend": "groq",
  "llm_default": "groq",
  "groq_configured": true,
  "ollama_reachable": false
}
```

`total_vectors: 0` means no corpus is loaded yet — ingest one. `ollama_reachable`
is a live probe, not a flag, so a client can avoid offering a provider this host
cannot actually serve.

---

## 5. Provider behaviour

| Request | `LLM_PROVIDER` | `GROQ_API_KEY` | Result |
|---------|----------------|----------------|--------|
| *(omitted)* | `ollama` | unset | Ollama used; `provider: "ollama"` |
| *(omitted)* | `groq` | set | Groq used; `provider: "groq"` |
| *(omitted)* | `groq` | unset | **503** — explicit "GROQ_API_KEY not configured" |
| `"ollama"` | any | any | Ollama used; **502** if unreachable |
| `"groq"` | any | set | Groq used |
| `"groq"` | any | unset | **503** |
| `"bogus"` | any | any | **400** invalid provider |

Per-request `provider` always wins over the default, so one deployment can serve
both.

---

## 6. Running locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill in as needed

python -m scripts.ingest_documents                  # build data/vectors/
streamlit run frontend/app.py                       # UI + embedded API
```

The UI boots the API itself, so a separate `uvicorn` process is optional. To work
on the API alone, run `uvicorn api.main:app --reload --port 8000` and point the UI
at it with `VANIVERSE_API_URL`.

With a local Ollama on a 4 GB card, use `OLLAMA_MODEL=qwen2.5:3b`; larger local
models exceed 4 GB of VRAM.

---

## 7. Optional — container self-host

Still supported for running the API on your own hardware, and independent of any
hosting platform.

```bash
docker build -t vaaniverse-api .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen2.5:3b \
  -v "$PWD/data:/app/data" \
  vaaniverse-api
# then: curl http://127.0.0.1:8000/health
```

The image is a two-stage `python:3.12-slim` build running as non-root `appuser`,
and honours `PORT` (falling back to `8000`). `.dockerignore` keeps `data/`,
`.venv/`, the PDF, `.env`, tests and docs out of the build context so none of
them can be shipped by accident.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `API error: Ollama API call failed.` | No `GROQ_API_KEY` secret, so the default provider is Ollama and there is no Ollama on Cloud | Add the secret in §1 |
| `API error: Groq API call failed for model '…': … 404 model_not_found` | Groq retired that model id | Set `GROQ_MODEL` to a current id |
| `/health` shows `total_vectors: 0` | Cold start wiped the index | Ingest a PDF (§2) |
| First ingest takes minutes | ~2.2 GB BGE-M3 download + CPU embedding | Expected, once per container session |
| No OCR on a scanned PDF | Tesseract apt packages missing | Confirm `packages.txt` was picked up, then redeploy |
| App restarts while answering | Cloud memory pressure | Keep `RERANK_ENABLED=false`; lower `EMBEDDING_BATCH_SIZE` |
