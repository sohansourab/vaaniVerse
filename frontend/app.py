import json
import streamlit as st
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title   = "VāṇiVerse",
    page_icon    = "🏛️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .vv-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E8C547;
        letter-spacing: 0.03em;
    }
    .vv-subheader {
        font-size: 0.95rem;
        color: #888;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    /* Source citation card */
    .source-card {
        background: #1E1E2E;
        border-left: 3px solid #E8C547;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .source-score {
        display: inline-block;
        background: #2A2A3E;
        color: #E8C547;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .source-script {
        display: inline-block;
        background: #2A3A2A;
        color: #7EC87E;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.8rem;
        margin-left: 6px;
    }

    /* Query script badge */
    .badge-telugu  { color: #7EC87E; font-weight: 600; }
    .badge-roman   { color: #7EB8EC; font-weight: 600; }
    .badge-mixed   { color: #E8C547; font-weight: 600; }
    .badge-unknown { color: #888;    font-weight: 600; }

    /* Chat messages */
    .stChatMessage { border-radius: 10px; }

    /* Sidebar */
    .sidebar-stat {
        background: #1E1E2E;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
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
        return "#7EC87E"
    if score >= 0.50:
        return "#E8C547"
    return "#E87E7E"


def script_badge(script: str) -> str:
    colors = {
        "telugu":  "#7EC87E",
        "roman":   "#7EB8EC",
        "mixed":   "#E8C547",
        "unknown": "#888888",
    }
    color = colors.get(script, "#888888")
    return f'<span style="color:{color};font-weight:600;">{script.upper()}</span>'


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, 1):
            score = s.get("score", 0)
            color = score_color(score)
            st.markdown(f"""
<div class="source-card">
    <b>[SOURCE {i}]</b> &nbsp;
    {s.get('source', 'unknown')} &nbsp;|&nbsp;
    Page <b>{s.get('page_number', '?')}</b> &nbsp;|&nbsp;
    <span class="source-score" style="color:{color};">
        score: {score:.4f}
    </span>
    <span class="source-script">{s.get('script', '?')}</span>
</div>
""", unsafe_allow_html=True)


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
    st.markdown('<div class="vv-header">🏛️ VāṇiVerse</div>', unsafe_allow_html=True)
    st.markdown('<div class="vv-subheader">Cultural Knowledge Retrieval</div>', unsafe_allow_html=True)

    # ── Health ─────────────────────────────────────────────────────────────────
    health = get_health()
    if health:
        st.markdown(f"""
<div class="sidebar-stat">
    🟢 <b>API Online</b><br>
    Vectors &nbsp;: <b>{health.get('total_vectors', 0)}</b><br>
    Backend &nbsp;: <b>{health.get('vector_backend', '?').upper()}</b><br>
    LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{health.get('llm_backend', '?').upper()}</b>
</div>
""", unsafe_allow_html=True)
    else:
        st.error("🔴 API Offline — run uvicorn api.main:app")

    st.divider()

    # ── Upload & Ingest ────────────────────────────────────────────────────────
    st.subheader("📥 Add to Archive")
    uploaded = st.file_uploader(
        "Upload Telugu PDF",
        type    = ["pdf"],
        help    = "Text-based PDFs only. Scanned docs require OCR (Phase 2).",
    )

    if st.button("⚡ Ingest Document", use_container_width=True, disabled=uploaded is None):
        with st.spinner("Parsing → Chunking → Embedding…"):
            result = ingest_file(uploaded)
            if result and "total_chunks" in result:
                st.success(
                    f"✅ **{result['source']}**\n\n"
                    f"{result['total_pages']} pages → "
                    f"{result['total_chunks']} chunks → "
                    f"{result['total_vectors']} total vectors"
                )
                st.rerun()
            elif result:
                st.error(f"Ingestion failed: {result.get('detail', result)}")
            else:
                st.error("Cannot reach API.")

    st.divider()

    # ── Retrieval Settings ─────────────────────────────────────────────────────
    st.subheader("⚙️ Retrieval Settings")

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
        "Filter by source (optional)",
        placeholder = "e.g. ramayanam.pdf",
        help        = "Restrict retrieval to a specific document.",
    )

    st.divider()

    # ── Clear chat ─────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

st.markdown('<div class="vv-header">💬 Knowledge Retrieval</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vv-subheader">'
    'Query in Telugu (తెలుగు), Romanized Telugu, or English'
    '</div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Render metadata row
            meta = msg.get("meta", {})
            if meta:
                script = meta.get("query_script", "unknown")
                model  = meta.get("model", "?")
                chunks = meta.get("total_chunks_retrieved", 0)
                st.markdown(
                    f"<small>"
                    f"Script: {script_badge(script)} &nbsp;|&nbsp; "
                    f"Model: <b>{model}</b> &nbsp;|&nbsp; "
                    f"Chunks retrieved: <b>{chunks}</b>"
                    f"</small>",
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
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and reasoning…"):
            data = query_api(
                query           = prompt,
                top_k           = top_k,
                score_threshold = score_threshold,
                filter_source   = filter_source.strip() or None,
            )

        if data is None:
            st.error("Cannot reach VāṇiVerse API. Is uvicorn running?")
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
            f"<small>"
            f"Script: {script_badge(script)} &nbsp;|&nbsp; "
            f"Model: <b>{model}</b> &nbsp;|&nbsp; "
            f"Chunks retrieved: <b>{chunks}</b>"
            f"</small>",
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