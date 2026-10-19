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

For a local clone, skip the upload entirely — this downloads a small public Telugu
PDF, ingests it, and leaves a working index:

```bash
python -m scripts.load_sample            # download + ingest + persist
python -m scripts.load_sample --dry-run  # parse/chunk only, no embedding
```

The PDF is intentionally **not** committed (third-party, and `data/` is
gitignored), which is why a script does the fetching.

> **Do not ingest the sample into an index you evaluate against.** The gold set in
> `eval/gold_qa.jsonl` matches answers by **page number** only, and the sample is a
> different book that also starts at page 1. Ingesting it silently makes "page 1"
> ambiguous and inflates recall. Use a separate index (or `DELETE /index/reset`
> first) before re-running `eval/run_eval.py`.

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
| `SCORE_THRESHOLD_TELUGU` | `0.41` | Score floor for native-Telugu queries. Measured — see `docs/EVALUATION.md`. |
| `SCORE_THRESHOLD_ROMAN` | `0.46` | Score floor for Romanized-Telugu and English queries. |
| `SCORE_THRESHOLD_MIXED` | `0.45` | Score floor for mixed-script queries. |
| `SCORE_THRESHOLD` | `0.45` | Fallback for an undetected script. |
| `HYBRID_ENABLED` | `false` | BM25 + dense fusion (RRF). Off by default. |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking. Off by default — see below. |
| `RERANK_DEVICE` | `cpu` | Pinned explicitly; the library claims every CUDA device otherwise. |
| `RERANK_THRESHOLD` | `0.5` | Abstention floor applied to rerank scores (only used when reranking). |
| `OCR_ENABLED` | `true` | OCR only runs for pages whose text layer is empty or vestigial. |
| `OCR_ENGINE` | `auto` | `auto`, `tesseract`, or `easyocr`. |
| `OCR_LANG` | `te,en` | ISO 639-1 codes; mapped to engine-specific codes internally. |
| `OCR_PREPROCESS` | `true` | Grayscale + autocontrast + Otsu before recognition (helps degraded scans). |
| `OCR_PSM` | `6` | Tesseract page-segmentation mode; 6 = uniform block of text. |
| `MAX_HISTORY_TURNS` | `4` | Conversation turns sent back to the model (client-supplied, stateless server). |
| `CITATION_VERIFY_ENABLED` | `true` | Flag claims whose cited passage does not support them. |
| `VANIVERSE_API_KEY` | *(secret)* | Enables `X-API-Key` auth on `/ingest`, `DELETE /index/reset`, `/admin/requests`. Unset ⇒ those routes are **open**. |
| `REQUIRE_AUTH_FOR_ALL` | `false` | Also require the key on `/query` and `/search`. |
| `RATE_LIMIT_ENABLED` | `true` | In-process rate limiting. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-client limit on `/query` and `/search`. |
| `INGEST_LIMIT_PER_HOUR` | `10` | Per-client limit on `/ingest`. |
| `MAX_UPLOAD_MB` | `50` | Reject larger uploads before reading them into memory. |
| `API_RELOAD` | `false` | uvicorn's reloader doubles model memory. Opt in locally. |

**Leave `RERANK_ENABLED` off on Cloud.** Reranking adds a second ~2.3 GB model
next to BGE-M3, which does not fit comfortably in the Cloud memory budget, and it
is CPU-only there. It has since been measured end-to-end: recall@5 moves
`0.900 → 0.9333` (one extra question of 30) for **+64.2 s per query** on CPU. It
is available and measured, not part of the hosted default — see
`docs/EVALUATION.md`.

> **Set `VANIVERSE_API_KEY` on any host that is reachable from outside.** Without
> it, anyone who can reach the port can ingest documents into the index or drop
> it entirely. Read-only routes stay open by default, so the public demo keeps
> working with the key set. `/health` reports `auth_configured` so this can be
> verified from outside.

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
  "ollama_reachable": false,
  "auth_configured": true,
  "hybrid_enabled": false,
  "rerank_enabled": false,
  "groq_model": "qwen/qwen3.8-27b",
  "groq_model_available": true,
  "sources": ["9.nenu-evarini.pdf"]
}
```

`total_vectors: 0` means no corpus is loaded yet — ingest one. `ollama_reachable`
is a live probe, not a flag, so a client can avoid offering a provider this host
cannot actually serve. `auth_configured` is `false` when no `VANIVERSE_API_KEY` is
set, i.e. the destructive routes are open — check it before exposing a port.
`groq_model_available` is `null` when the check cannot be made (no key, or the
catalogue call failed), which must not be read as "unavailable". `sources` lists
the distinct documents in the index; it backs the UI's source filter.

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

python -m scripts.load_sample                       # public sample PDF → data/vectors/
streamlit run frontend/app.py                       # UI + embedded API
```

`scripts/ingest_documents.py` needs an explicit input:

```bash
python -m scripts.ingest_documents --input path/to/book.pdf
python -m scripts.ingest_documents --input path/to/folder/ --dry-run
```

The UI boots the API itself, so a separate `uvicorn` process is optional. To work
on the API alone, run `uvicorn api.main:app --port 8000` (add
`API_RELOAD=true` for auto-reload) and point the UI at it with `VANIVERSE_API_URL`.

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
| `401` from `/ingest` or `DELETE /index/reset` | `VANIVERSE_API_KEY` is set and the request carried no/wrong `X-API-Key` | Send the header, or unset the variable to reopen the route |
| `429` from `/query` or `/search` | Per-client rate limit | Raise `RATE_LIMIT_PER_MINUTE`, or set `RATE_LIMIT_ENABLED=false` locally |
| `/query` refuses everything | Index empty or rebuilt with a different embedder | Re-ingest; thresholds are calibrated per embedder + corpus |
| Answers cite an unexpected page | Sample corpus was added to an evaluated index | `DELETE /index/reset`, then re-ingest one document (§2) |
