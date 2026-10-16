# DEPLOY.md — Deploying VāṇiVerse to Railway (prepared, **not deployed**)

This document describes how to deploy the **API** (`api/main.py`) to
[Railway](https://railway.app). Nothing here has been executed — no deploy was run.

Everything the app needs is driven by environment variables. **No secret is ever
committed**; `.env` is gitignored and `.env.example` ships with an empty
`GROQ_API_KEY` placeholder.

---

## 0. What gets deployed

| Component | Source | Notes |
|-----------|--------|-------|
| API image | `Dockerfile` | `python:3.12-slim`, two-stage build, **non-root** user `appuser`, honours `PORT` |
| Build/deploy config | `railway.json` | Dockerfile builder, `/health` healthcheck, restart policy |
| Ignore list | `.dockerignore` | excludes `data/`, `.venv/`, the PDF, `.env`, tests, docs |
| Index | Not in the image | must be provided at runtime — see §2 |

Build the image locally to sanity-check it before deploying:

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

---

## 1. Environment variables (set in Railway → **Variables**)

Set these in the Railway dashboard (or `railway variables --set ...`). **Never**
put them in `Dockerfile`, `railway.json`, or any committed file.

| Variable | Example | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `groq` or `ollama` | **Default** provider when a request omits `provider`. |
| `GROQ_API_KEY` | *(secret)* | Required only for the `groq` provider. Empty ⇒ Groq requests return **503**. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id. |
| `OLLAMA_BASE_URL` | `https://<tunnel-host>` (Option A) | Reachable Ollama endpoint (local or tunnelled). |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model id. |
| `VECTOR_BACKEND` | `faiss` | Vector store backend. |
| `EMBEDDING_DEVICE` | `cpu` | Railway has no GPU; keep `cpu`. |
| `EMBEDDING_BATCH_SIZE` | `16` | Lower if memory-constrained. |
| `PORT` | *(injected by Railway)* | The container binds `${PORT:-8000}`. |

> The API is designed to **start even when a backend is unavailable**. Selecting
> `groq` without `GROQ_API_KEY` returns `503` at request time; it never crashes boot.
> `GET /health` reports `llm_default` and `groq_configured` (boolean, no
> reachability check) so you can confirm configuration from the dashboard.

---

## 2. Index strategy — **recommendation: ship the prebuilt index to a volume**

The FAISS index (`data/vectors/*.faiss` + `*.pkl`, plus `data/documents/`) is the
app's state. Two ways to get it onto Railway:

### ✅ Recommended — build locally, ship to a Railway **volume**

Why: the index is small (the book is 4.4 MB; BGE-M3 vectors are 1024-dim, so a few
thousand chunks is tens of MB). Rebuilding inside Railway would require the source
PDF *and* a ~2 GB `BAAI/bge-m3` download, then slow **CPU** embedding on every fresh
container. Shipping once and persisting is faster, cheaper, and deterministic.

```bash
# 1. Build the index locally (GPU) — see VERIFY.md for the exact commands.
python -m scripts.ingest_documents --input 9.nenu-evarini.pdf

# 2. Inspect what to ship
ls data/vectors data/documents

# 3. Create a Railway project + service, then attach a volume at /app/data
railway init
railway link                     # select the project/service
railway volume add --mount-path /app/data

# 4. Copy the index into the volume.
#    (a) If your plan supports it, open a shell into the running service and
#        upload the tarball, e.g.:
tar -czf index.tgz -C data .
railway ssh
#   ...inside the container:
#     mkdir -p /app/data && tar -xzf /tmp/index.tgz -C /app/data

#    (b) Alternative without SSH: temporarily bake `data/` into a one-off image by
#        removing `data/` from .dockerignore, build+push it, run its start command
#        as a one-shot job (`cp -r /app/data/* /volume/`), then revert .dockerignore.
```

The container's `/app/data` must be writable by the non-root `appuser`; the
Dockerfile pre-creates `/app/data/documents` and `/app/data/vectors` and chowns
them. If you mount a fresh Railway volume, ensure the mount is writable by
`appuser` (Railway volumes are mounted as root by default — run the copy as root
once and `chown -R 10001:10001 /app/data`).

### Alternative — rebuild the index on Railway

Only if you must avoid shipping artefacts. You would need to:

1. Put the source PDF on the volume, and
2. Run the ingestion **inside** the container once, persisting `/app/data`:

```bash
railway run -- python -m scripts.ingest_documents --input /app/data/9.nenu-evarini.pdf
```

This downloads `BAAI/bge-m3` (~2 GB) on first run and embeds on CPU (slow).
The result persists in the volume, so it is a one-off cost — but it is heavier and
more error-prone than Option §2-recommended.

---

## 3. Option A — Groq as default, **Ollama reachable per request via a tunnel**

The API runs on Railway; Ollama keeps running on your PC. `LLM_PROVIDER=groq` makes
Groq the default; a client can still ask for `"provider": "ollama"` per request and
it will reach your PC through the tunnel.

> ⚠️ **Your PC must be on** (and Ollama running + tunnel up) for any `ollama`
> request to succeed. If it is off, `ollama` requests fail with a `502`.

### Tunnel with cloudflared (quick)

```bash
# On your PC, with Ollama listening on 11434:
cloudflared tunnel --url http://localhost:11434
# → prints something like https://random-words.trycloudflare.com
```

### Or a stable tunnel with Tailscale

```bash
# On your PC:
tailscale up
# Expose Ollama to your tailnet (either use the Tailscale IP directly, or):
tailscale serve --bg 11434 http://127.0.0.1:11434
tailscale funnel 11434          # only if you need public exposure
# OLLAMA_BASE_URL = http://<pc-tailscale-name>:11434
```

### Railway variables (Option A)

```
LLM_PROVIDER=groq
GROQ_API_KEY=<set in dashboard, never committed>
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_BASE_URL=https://<your-tunnel-host>
OLLAMA_MODEL=qwen2.5:3b
EMBEDDING_DEVICE=cpu
```

### Example calls (Option A)

```bash
BASE=https://<your-service>.up.railway.app

# Default provider (Groq, because LLM_PROVIDER=groq)
curl -s "$BASE/query" -H 'Content-Type: application/json' \
  -d '{"query":"రాముడు ఎవరు?","top_k":5}' | jq '.provider, .answer'

# Explicitly use the tunnelled Ollama on your PC
curl -s "$BASE/query" -H 'Content-Type: application/json' \
  -d '{"query":"రాముడు ఎవరు?","top_k":5,"provider":"ollama"}' | jq '.provider, .answer'
```

---

## 4. Option B — **Groq only** (no tunnel)

Simplest and most reliable production shape: no dependency on your PC.

### Railway variables (Option B)

```
LLM_PROVIDER=groq
GROQ_API_KEY=<set in dashboard>
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_DEVICE=cpu
```

### Example calls (Option B)

```bash
BASE=https://<your-service>.up.railway.app

# Default provider = Groq
curl -s "$BASE/query" -H 'Content-Type: application/json' \
  -d '{"query":"Who is Rama?","top_k":5}' | jq '.provider, .answer'

# Explicit Groq
curl -s "$BASE/query" -H 'Content-Type: application/json' \
  -d '{"query":"Who is Rama?","provider":"groq"}' | jq '.provider'

# Ollama is not reachable here → expect 502 (inference failure), not a crash
curl -s -o /dev/null -w '%{http_code}\n' "$BASE/query" \
  -H 'Content-Type: application/json' -d '{"query":"hi","provider":"ollama"}'
```

---

## 5. Deploy steps (run these only when you are ready)

```bash
railway login
railway init                      # or: railway link
railway volume add --mount-path /app/data   # see §2
railway variables                 # set the vars from §1 / §3 / §4
railway up                        # build the Dockerfile and deploy
railway domain                    # get the public URL
curl -s https://<your-domain>/health | jq
```

Expected `/health`:

```json
{
  "status": "ok",
  "total_vectors": <N>,
  "vector_backend": "faiss",
  "llm_backend": "groq",
  "llm_default": "groq",
  "groq_configured": true
}
```

---

## 6. Quick reference — provider behaviour

| Request | `LLM_PROVIDER` | `GROQ_API_KEY` | Result |
|---------|----------------|----------------|--------|
| *(no `provider`) | `ollama` | unset | Ollama used; `provider: "ollama"` |
| *(no `provider`) | `groq` | set | Groq used; `provider: "groq"` |
| *(no `provider`) | `groq` | unset | **503** — clear "GROQ_API_KEY not configured" message |
| `provider: "ollama"` | any | any | Ollama used (must be reachable, else 502) |
| `provider: "groq"` | any | set | Groq used |
| `provider: "groq"` | any | unset | **503** |
| `provider: "bogus"` | any | any | **400** invalid provider |
