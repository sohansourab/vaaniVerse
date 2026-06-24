import json
import streamlit as st
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="VāṇiVerse",
    page_icon="🛕",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources cited — {len(sources)}", expanded=False):
        rows = ""
        for i, s in enumerate(sources, 1):
            score = s.get("score", 0)
            color = score_color(score)
            rows += f"""
<div class="vv-footnote">
    <span class="vv-footnote-ref">
        <b>[{i}]</b> &nbsp;{s.get('source', 'unknown')} · p.{s.get('page_number', '?')}
        &nbsp;&nbsp;{script_badge(s.get('script', '?'))}
    </span>
    <span class="vv-footnote-score" style="color:{color};">{score:.4f}</span>
</div>"""
        st.markdown(f'<div class="vv-footnotes">{rows}</div>', unsafe_allow_html=True)


def query_api(
    query: str,
    top_k: int,
    score_threshold: float,
    filter_source: str | None,
) -> dict | None:
    payload = {
        "query":           query,
        "top_k":           top_k,
        "score_threshold": score_threshold,
    }
    if filter_source:
        payload["filter_source"] = filter_source
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json    = payload,
            timeout = 60,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


def ingest_file(file) -> dict | None:
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        resp  = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=300)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="vv-wordmark">VāṇiVerse</p>', unsafe_allow_html=True)
    st.markdown('<p class="vv-script">వాణివర్స్ · the archive</p>', unsafe_allow_html=True)

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    # ── Health ─────────────────────────────────────────────────────────────────
    health = get_health()
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
<p class="vv-hint">Run <code>uvicorn api.main:app</code> to reconnect.</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="vv-divider">॥</div>', unsafe_allow_html=True)

    # ── Upload & Ingest ────────────────────────────────────────────────────────
    st.markdown('<p class="vv-eyebrow">Add to archive</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a Telugu PDF",
        type    = ["pdf"],
        help    = "Text-based PDFs only. Scanned documents need OCR (phase 2).",
    )

    if st.button("Ingest document", use_container_width=True, disabled=uploaded is None):
        with st.spinner("Parsing → chunking → embedding…"):
            result = ingest_file(uploaded)
            if result and "total_chunks" in result:
                st.success(
                    f"{result['source']} — {result['total_pages']} pages → "
                    f"{result['total_chunks']} chunks → {result['total_vectors']} vectors"
                )
                st.rerun()
            elif result:
                st.error(f"Ingestion failed: {result.get('detail', result)}")
            else:
                st.error("Cannot reach the API.")

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

    score_threshold = st.slider(
        "Score threshold",
        min_value = 0.0,
        max_value = 1.0,
        value     = 0.45,
        step      = 0.05,
        help      = "Minimum cosine similarity. Lower = more results, less precise.",
    )

    filter_source = st.text_input(
        "Filter by source",
        placeholder = "e.g. ramayanam.pdf",
        help        = "Restrict retrieval to a specific document.",
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

# Render existing chat history
for msg in st.session_state.messages:
    avatar = "🛕" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            if meta:
                st.markdown(
                    f'<p class="vv-meta">'
                    f'{script_badge(meta.get("query_script", "unknown"))}&nbsp;&nbsp;'
                    f'<b>{meta.get("model", "?")}</b>&nbsp;&nbsp;'
                    f'{meta.get("total_chunks_retrieved", 0)} chunks retrieved'
                    f'</p>',
                    unsafe_allow_html=True,
                )
            render_sources(msg.get("sources", []))

# Chat input
if prompt := st.chat_input(
    "అర్జునుడికి విషాదం ఎందుకు కలిగింది? / Who is Rama? / What is dharma?"
):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API and show response
    with st.chat_message("assistant", avatar="🛕"):
        with st.spinner("Retrieving context and reasoning…"):
            data = query_api(
                query           = prompt,
                top_k           = top_k,
                score_threshold = score_threshold,
                filter_source   = filter_source.strip() or None,
            )

        if data is None:
            st.error("Cannot reach the VāṇiVerse API. Is uvicorn running?")
            st.stop()

        if "detail" in data:
            st.error(f"API error: {data['detail']}")
            st.stop()

        answer  = data.get("answer", "No answer generated.")
        sources = data.get("sources_used", [])
        script  = data.get("query_script", "unknown")
        model   = data.get("model", "?")
        chunks  = data.get("total_chunks_retrieved", 0)

        # Render answer
        st.markdown(answer)

        # Metadata row
        st.markdown(
            f'<p class="vv-meta">'
            f'{script_badge(script)}&nbsp;&nbsp;'
            f'<b>{model}</b>&nbsp;&nbsp;'
            f'{chunks} chunks retrieved'
            f'</p>',
            unsafe_allow_html=True,
        )

        # Source citations
        render_sources(sources)

        # Save to session
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "meta":    {
                "query_script":          script,
                "model":                 model,
                "total_chunks_retrieved": chunks,
            },
        })