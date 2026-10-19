import json
import os
import sys
import threading
import time
from pathlib import Path

import requests
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Streamlit Cloud runs this file from frontend/, so put the repo root on sys.path
# before we try to launch the bundled FastAPI app (see start_embedded_api).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="VāṇiVerse",
    page_icon="🛕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Configuration ──────────────────────────────────────────────────────────────
# Default to a backend on localhost. Override with the VANIVERSE_API_URL env var
# or an API_BASE_URL Streamlit secret to point at an externally hosted API.
#
# Flat secrets are also bridged into os.environ because core/config.py reads its
# configuration (GROQ_API_KEY, LLM_PROVIDER, …) from the environment at import
# time — i.e. before the embedded API is imported further down.
LOCAL_API_URL = "http://localhost:8000"
API_BASE_URL  = os.getenv("VANIVERSE_API_URL", LOCAL_API_URL)
try:  # st.secrets raises when no secrets.toml is configured
    for _key, _val in st.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
    if "API_BASE_URL" in st.secrets:
        API_BASE_URL = st.secrets["API_BASE_URL"]
except Exception:
    pass

# ── Design tokens & global style ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tiro+Telugu:ital@0;1&family=Work+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:           #14110F;
    --ink-raised:    #1C1815;
    --ink-line:      #242019;
    --parchment:     #EAE2D0;
    --parchment-dim: #9C9485;
    --turmeric:      #D9A441;
    --copper:        #B5603C;
    --banyan:        #6E8F6B;
    --hairline:      rgba(234, 226, 208, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Work Sans', sans-serif;
}

.stApp {
    background: var(--ink);
    color: var(--parchment);
}

#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--ink-raised);
    border-right: 1px solid var(--hairline);
}
[data-testid="stSidebar"] * {
    color: var(--parchment);
}

/* ── Wordmark & headings ────────────────────────────────────────────────── */
.vv-wordmark {
    font-family: 'Tiro Telugu', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--turmeric);
    letter-spacing: 0.01em;
    line-height: 1.15;
    margin: 0;
}
.vv-script {
    font-family: 'Tiro Telugu', serif;
    font-size: 0.95rem;
    color: var(--parchment-dim);
    margin: 2px 0 0 0;
}
.vv-eyebrow {
    font-size: 0.72rem;
    color: var(--parchment-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 6px 0;
}
.vv-tagline {
    font-size: 0.92rem;
    color: var(--parchment-dim);
    margin: 4px 0 18px 0;
}
.vv-divider {
    text-align: center;
    color: var(--turmeric);
    font-size: 1rem;
    opacity: 0.5;
    margin: 16px 0;
    letter-spacing: 0.4em;
}

/* ── Ledger (sidebar status block) ─────────────────────────────────────── */
.vv-ledger {
    border: 1px solid var(--hairline);
    border-radius: 4px;
    padding: 2px 14px;
}
.vv-ledger-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 8px 0;
    font-size: 0.82rem;
}
.vv-ledger-row + .vv-ledger-row {
    border-top: 1px solid var(--hairline);
}
.vv-ledger-label {
    color: var(--parchment-dim);
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
}
.vv-ledger-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.85rem;
}
.vv-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 7px;
}
.vv-hint {
    font-size: 0.78rem;
    color: var(--parchment-dim);
    margin-top: 8px;
}

/* ── Chips (script / status labels) ────────────────────────────────────── */
.vv-chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 3px;
    border: 1px solid currentColor;
}

/* ── Answer metadata & footnote apparatus ──────────────────────────────── */
.vv-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--parchment-dim);
    margin: 10px 0 2px 0;
}
.vv-meta b { color: var(--parchment); font-weight: 500; }

.vv-footnotes { margin-top: 4px; }
.vv-footnote {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 14px;
    padding: 8px 2px;
    border-top: 1px solid var(--hairline);
    font-size: 0.78rem;
}
.vv-footnote:first-child { border-top: none; }
.vv-footnote-ref {
    color: var(--parchment-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vv-footnote-ref b {
    color: var(--parchment);
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}
.vv-footnote-score {
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
}

/* The cited passage itself: the reader must be able to check the claim. */
.vv-passage {
    font-family: 'Tiro Telugu', 'Work Sans', serif;
    font-size: 0.84rem;
    line-height: 1.65;
    color: var(--parchment-dim);
    border-left: 2px solid var(--hairline);
    padding: 2px 0 2px 12px;
    margin: 0 0 10px 2px;
    white-space: pre-wrap;
}

/* ── Inputs & controls ──────────────────────────────────────────────────── */
.stButton button {
    background: transparent;
    border: 1px solid var(--turmeric);
    color: var(--turmeric) !important;
    border-radius: 4px;
    font-weight: 500;
    transition: background 0.15s ease, color 0.15s ease;
}
.stButton button:hover {
    background: var(--turmeric);
    color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--ink-line);
    border: 1px dashed var(--hairline);
}
.stTextInput input {
    background: var(--ink-line) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--hairline) !important;
}
[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--turmeric) !important;
}
[data-testid="stExpander"] {
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: var(--parchment-dim) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_health() -> dict | None:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=3).json()
    except Exception:
        return None


# ── Embedded API bootstrap ─────────────────────────────────────────────────────
# Streamlit Community Cloud provides a single process — the Streamlit server — so
# nothing is listening on localhost:8000 and every API call above fails. This
# starts api.main:app (uvicorn) on a daemon thread inside that same process, so
# the UI works standalone with no second host.

def api_is_up(timeout: float = 2.0) -> bool:
    """True when something answers /health at API_BASE_URL."""
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def start_embedded_api() -> tuple[bool, str]:
    """Launch uvicorn on 127.0.0.1:8000 in a background thread; wait until ready."""
    try:
        import uvicorn
        from api.main import app as api_app
    except Exception as exc:
        return False, f"could not import the API: {exc}"

    server = uvicorn.Server(
        uvicorn.Config(
            api_app,
            host       = "127.0.0.1",
            port       = 8000,
            log_level  = "info",
            access_log = False,
        )
    )
    threading.Thread(target=server.run, daemon=True, name="vaaniverse-api").start()

    for _ in range(180):  # up to ~3 minutes
        if api_is_up(timeout=1.0):
            return True, "started"
        time.sleep(1)
    return False, "the API did not become ready in time"


@st.cache_resource(show_spinner=False)
def ensure_api() -> tuple[bool, str]:
    """(ready, detail). Starts the embedded API only when nothing else answers."""
    if api_is_up():
        return True, "external"
    return start_embedded_api()


def ensure_api_ready() -> bool:
    """UI-facing wrapper — shows a spinner while a cold start happens."""
    if api_is_up():
        return True
    with st.spinner("Starting the VāṇiVerse engine…"):
        ready, _detail = ensure_api()
    return ready


def score_color(score: float) -> str:
    if score >= 0.65:
        return "var(--banyan)"
    if score >= 0.50:
        return "var(--turmeric)"
    return "var(--copper)"


_SCRIPT_COLORS = {
    "telugu":  "var(--banyan)",
    "roman":   "var(--copper)",
    "mixed":   "var(--turmeric)",
    "unknown": "var(--parchment-dim)",
}


def script_badge(script: str) -> str:
    color = _SCRIPT_COLORS.get(script, "var(--parchment-dim)")
    return f'<span class="vv-chip" style="color:{color};">{script}</span>'


def render_sources(sources: list[dict], *, key: str = "") -> None:
    """
    Cited passages, not just filenames.

    A citation the reader cannot check is not a citation. Each entry shows the
    source, page and score, plus the passage the model was given, collapsed by
    default so a long answer stays readable.
    """
    if not sources:
        return
    with st.expander(f"Sources cited — {len(sources)}", expanded=False):
        for i, s in enumerate(sources, 1):
            score = s.get("score", 0)
            color = score_color(score)
            st.markdown(
                f'<div class="vv-footnote">'
                f'<span class="vv-footnote-ref">'
                f'<b>[{i}]</b> &nbsp;{s.get("source", "unknown")} · '
                f'p.{s.get("page_number", "?")}'
                f'&nbsp;&nbsp;{script_badge(s.get("script", "?"))}'
                f'</span>'
                f'<span class="vv-footnote-score" style="color:{color};">'
                f'{score:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            text = (s.get("text") or "").strip()
            if text:
                excerpt = text if len(text) <= 700 else text[:700] + "…"
                st.markdown(
                    f'<div class="vv-passage">{excerpt}</div>',
                    unsafe_allow_html=True,
                )


def render_verification(verification: dict | None) -> None:
    """Surface the claim↔citation check without turning it into a verdict."""
    if not verification:
        return
    if verification.get("passed"):
        st.markdown(
            '<p class="vv-meta" style="color:var(--banyan);">'
            '✓ every citation points at a supplied source</p>',
            unsafe_allow_html=True,
        )
        return

    problems = []
    invalid = verification.get("invalid_citations") or []
    if invalid:
        problems.append(
            "cited " + ", ".join(f"[SOURCE {n}]" for n in invalid)
            + " but those sources were not provided"
        )
    unsupported = verification.get("unsupported_claims") or 0
    if unsupported:
        problems.append(f"{unsupported} claim(s) did not overlap the passage they cited")
    st.warning(
        "**Citation check flagged this answer.** " + "; ".join(problems) + ".",
        icon="⚠️",
    )


def render_feedback(entry: dict, key: str) -> None:
    """Two buttons that append a verdict to the server's feedback log."""
    state_key = f"fb_{key}"
    if st.session_state.get(state_key):
        st.markdown(
            f'<p class="vv-meta">Thanks — recorded as '
            f'<b>{st.session_state[state_key]}</b>.</p>',
            unsafe_allow_html=True,
        )
        return

    left, right, _ = st.columns([1, 1, 8])
    verdict = None
    if left.button("👍", key=f"up_{key}", help="This answer was useful"):
        verdict = "up"
    if right.button("👎", key=f"down_{key}", help="This answer was wrong or unsupported"):
        verdict = "down"
    if verdict:
        payload = {
            "verdict":  verdict,
            "query":    entry.get("query", ""),
            "answer":   entry.get("content", ""),
            "provider": entry.get("meta", {}).get("provider"),
            "model":    entry.get("meta", {}).get("model"),
            "sources":  entry.get("sources", []),
        }
        try:
            requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=15)
        except requests.exceptions.RequestException:
            pass  # feedback is best-effort; never block the reading room on it
        st.session_state[state_key] = verdict
        st.rerun()


def query_api(
    query: str,
    top_k: int,
    score_threshold: float | None,
    filter_source: str | None,
    provider: str,
    history: list[dict] | None = None,
) -> dict | None:
    payload = {
        "query":    query,
        "top_k":    top_k,
        "provider": provider,
    }
    # None means "use the server's calibrated per-script default", which is what
    # the UI offers when the threshold box is left alone.
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold
    if filter_source:
        payload["filter_source"] = filter_source
    if history:
        payload["history"] = history[-8:]
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json    = payload,
            timeout = 300,  # first call may load BGE-M3 — slow on CPU
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


def ingest_file(file) -> dict | None:
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        resp  = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=900)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


def fetch_sources() -> list[str]:
    """Source filenames currently in the index, for the sidebar filter."""
    try:
        resp = requests.get(f"{API_BASE_URL}/index/sources", timeout=15)
        if resp.ok:
            return resp.json().get("sources", [])
    except requests.exceptions.RequestException:
        pass
    return []


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="vv-wordmark">VāṇiVerse</p>', unsafe_allow_html=True)
    st.markdown('<p class="vv-script">వాణివర్స్ · the archive</p>', unsafe_allow_html=True)

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    # ── Health ─────────────────────────────────────────────────────────────────
    api_ready = ensure_api_ready()
    health    = get_health() if api_ready else None
    if health:
        st.markdown(f"""
<div class="vv-ledger">
    <div class="vv-ledger-row">
        <span class="vv-ledger-label"><span class="vv-dot" style="background:var(--banyan);"></span>Status</span>
        <span class="vv-ledger-value" style="color:var(--banyan);">Online</span>
    </div>
    <div class="vv-ledger-row">
        <span class="vv-ledger-label">Vectors</span>
        <span class="vv-ledger-value">{health.get('total_vectors', 0)}</span>
    </div>
    <div class="vv-ledger-row">
        <span class="vv-ledger-label">Backend</span>
        <span class="vv-ledger-value">{health.get('vector_backend', '?').upper()}</span>
    </div>
    <div class="vv-ledger-row">
        <span class="vv-ledger-label">LLM</span>
        <span class="vv-ledger-value">{health.get('llm_backend', '?').upper()}</span>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="vv-ledger">
    <div class="vv-ledger-row">
        <span class="vv-ledger-label"><span class="vv-dot" style="background:var(--copper);"></span>Status</span>
        <span class="vv-ledger-value" style="color:var(--copper);">Offline</span>
    </div>
</div>
<p class="vv-hint">The embedded API failed to start — check the app logs.</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    # ── Upload & Ingest ────────────────────────────────────────────────────────
    st.markdown('<p class="vv-eyebrow">Add to archive</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a Telugu PDF",
        type    = ["pdf"],
        help    = "Text PDFs use their text layer. Scanned pages are OCR'd when an engine is available.",
    )

    if st.button("Ingest document", use_container_width=True, disabled=uploaded is None):
        ensure_api_ready()
        # Ingest is one blocking request (parse → chunk → embed → persist), so a
        # real progress bar is not available without server streaming. Say what
        # is happening and why it is slow instead of leaving a bare spinner.
        progress = st.progress(0.0, text="Uploading…")
        with st.spinner("Parsing → chunking → embedding…"):
            progress.progress(0.15, text="Parsing pages…")
            result = ingest_file(uploaded)
            progress.progress(0.95, text="Saving index…")
            if result and "total_chunks" in result:
                progress.progress(1.0, text="Done")
                replaced = result.get("replaced_chunks", 0)
                note = f" (replaced {replaced} old chunks)" if replaced else ""
                st.success(
                    f"{result['source']} — {result['total_pages']} pages → "
                    f"{result['total_chunks']} chunks → {result['total_vectors']} vectors{note}"
                )
                st.rerun()
            elif result:
                progress.empty()
                st.error(f"Ingestion failed: {result.get('detail', result)}")
            else:
                progress.empty()
                st.error("Cannot reach the API.")

    if health and health.get("total_vectors", 0) == 0:
        st.markdown(
            '<p class="vv-hint">The index is empty. Upload a PDF above to build '
            'it — the first ingest also downloads BGE-M3 (~2.2 GB) and embeds on '
            'CPU, so give it a few minutes.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    # ── Retrieval Settings ─────────────────────────────────────────────────────
    st.markdown('<p class="vv-eyebrow">Retrieval settings</p>', unsafe_allow_html=True)

    top_k = st.slider(
        "Top K chunks",
        min_value = 1,
        max_value = 10,
        value     = 5,
        help      = "Number of source chunks to retrieve per query.",
    )

    # The server applies a calibrated cutoff per script (0.41 Telugu / 0.46 Roman
    # and English). Overriding it here is possible, but the default is the
    # measured value, so "automatic" is the first-class choice rather than an
    # invisible constant.
    threshold_mode = st.radio(
        "Score threshold",
        options = ["Automatic (per script)", "Manual"],
        index   = 0,
        help    = "Automatic uses the calibrated cutoff for the query's script "
                  "(0.41 Telugu, 0.46 Roman/English). Manual pins one value.",
    )
    score_threshold = None
    if threshold_mode == "Manual":
        score_threshold = st.slider(
            "Manual threshold",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.45,
            step      = 0.01,
            help      = "Minimum similarity for the score that ranks results. "
                        "Lower = more results, less precise.",
        )

    indexed_sources = fetch_sources() if api_ready else []
    if indexed_sources:
        filter_source = st.selectbox(
            "Filter by source",
            options = ["All sources"] + indexed_sources,
            index   = 0,
            help    = "Restrict retrieval to a specific document.",
        )
        filter_source = "" if filter_source == "All sources" else filter_source
    else:
        filter_source = st.text_input(
            "Filter by source",
            placeholder = "e.g. ramayanam.pdf",
            help        = "Restrict retrieval to a specific document.",
        )

    # Only offer providers this server can actually serve. Where there is no
    # local Ollama — Streamlit Community Cloud, for example — listing it just
    # produces a confusing failure at query time. Filtering it out here also lets
    # the default fall through to Groq, if a key is configured.
    health = health or {}
    ollama_ok = bool(health.get("ollama_reachable", True))
    groq_ok   = bool(health.get("groq_configured", False))

    provider_options = [
        name for name, usable in (("ollama", ollama_ok), ("groq", groq_ok)) if usable
    ]
    if not provider_options:
        # Neither works; keep both listed so the server's error message, not the
        # UI, explains why.
        provider_options = ["ollama", "groq"]

    if not ollama_ok and not groq_ok:
        # Say this before the user asks something, not after it fails. A hosted
        # deployment with no secret configured has no working provider at all,
        # and "Ollama API call failed" (the product default) explains nothing to
        # someone who has never heard of Ollama.
        st.warning(
            "**No LLM provider is configured.** No Ollama server answered on this "
            "host, and `GROQ_API_KEY` is not set — so answers will fail.\n\n"
            "On Streamlit Community Cloud, add it under "
            "**App → Settings → Secrets**:\n\n"
            '```toml\nGROQ_API_KEY = "gsk_..."\n```'
        )

    provider_default = health.get("llm_default", "ollama")
    provider_choice = st.selectbox(
        "LLM provider",
        options = provider_options,
        index   = provider_options.index(provider_default)
                  if provider_default in provider_options else 0,
        help    = "Only providers this server can serve are listed. "
                  "Ollama needs a local server; Groq needs GROQ_API_KEY.",
    )

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

st.markdown('<p class="vv-wordmark" style="font-size:1.6rem;">Reading room</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="vv-tagline">'
    'Query in తెలుగు, romanized Telugu, or English — every answer is grounded in cited source text.'
    '</p>',
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Starter questions. A first-time visitor gets an empty reading room and an empty
# input box; these are drawn from the gold set so they are known to be answerable
# from the sample book, and they cover all three query modes.
_SAMPLE_QUESTIONS = [
    "నేను ఎవరు?",
    "eruka ante emiti?",
    "What is awareness?",
]

if not st.session_state.messages:
    st.markdown(
        '<p class="vv-hint">Nothing asked yet. Try one of these, or write your '
        'own below.</p>',
        unsafe_allow_html=True,
    )
    _cols = st.columns(len(_SAMPLE_QUESTIONS))
    for _col, _question in zip(_cols, _SAMPLE_QUESTIONS):
        if _col.button(_question, key=f"starter_{_question}", use_container_width=True):
            st.session_state.pending_prompt = _question
            st.rerun()

# Render existing chat history
for _idx, msg in enumerate(st.session_state.messages):
    avatar = "🛕" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            if meta:
                st.markdown(
                    f'<p class="vv-meta">'
                    f'{script_badge(meta.get("query_script", "unknown"))}&nbsp;&nbsp;'
                    f'{meta.get("provider", "?")}&nbsp;/&nbsp;<b>{meta.get("model", "?")}</b>&nbsp;&nbsp;'
                    f'{meta.get("total_chunks_retrieved", 0)} chunks retrieved'
                    f'</p>',
                    unsafe_allow_html=True,
                )
            render_verification(msg.get("verification"))
            render_sources(msg.get("sources", []))
            render_feedback(msg, key=f"msg{_idx}")

# Chat input — a starter button can also seed the prompt.
prompt = st.chat_input(
    "అర్జునుడికి విషాదం ఎందుకు కలిగింది? / Who is Rama? / What is dharma?"
) or st.session_state.pop("pending_prompt", None)

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API and show response
    with st.chat_message("assistant", avatar="🛕"):
        ensure_api_ready()

        # Prior turns let a follow-up ("and what did that mean?") resolve. The
        # server treats them as chat context only, never as citable sources.
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.spinner("Retrieving context and reasoning…"):
            data = query_api(
                query           = prompt,
                top_k           = top_k,
                score_threshold = score_threshold,
                filter_source   = filter_source.strip() or None,
                provider        = provider_choice,
                history         = history,
            )

        if data is None:
            st.error(f"Cannot reach the VāṇiVerse API at {API_BASE_URL}.")
            st.stop()

        if "detail" in data:
            detail = data["detail"]
            st.error(f"API error: {detail}")
            if "Ollama" in detail:
                probe = get_health() or {}
                if not probe.get("ollama_reachable", True):
                    st.info(
                        "This deployment has no local Ollama server. Pick **groq** "
                        "as the LLM provider in the sidebar — the server needs a "
                        "`GROQ_API_KEY` secret for it to work."
                    )
            st.stop()

        answer  = data.get("answer", "No answer generated.")
        sources = data.get("sources_used", [])
        script  = data.get("query_script", "unknown")
        model   = data.get("model", "?")
        provider_used = data.get("provider", provider_choice)
        chunks  = data.get("total_chunks_retrieved", 0)
        threshold_used = data.get("score_threshold_used")
        verification   = data.get("verification")

        # Render answer
        st.markdown(answer)

        # Metadata row
        threshold_note = (
            f'&nbsp;&nbsp;cutoff {threshold_used:.2f}'
            if isinstance(threshold_used, (int, float)) else ""
        )
        st.markdown(
            f'<p class="vv-meta">'
            f'{script_badge(script)}&nbsp;&nbsp;'
            f'{provider_used}&nbsp;/&nbsp;<b>{model}</b>&nbsp;&nbsp;'
            f'{chunks} chunks retrieved{threshold_note}'
            f'</p>',
            unsafe_allow_html=True,
        )

        render_verification(verification)

        # Source citations, with the cited passages
        render_sources(sources)

        # Save to session
        entry = {
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "verification": verification,
            "meta":    {
                "query_script":          script,
                "provider":              provider_used,
                "model":                 model,
                "total_chunks_retrieved": chunks,
                "score_threshold_used":  threshold_used,
            },
        }
        st.session_state.messages.append(entry)
        render_feedback(entry, key=f"msg{len(st.session_state.messages) - 1}")