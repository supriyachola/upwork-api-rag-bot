"""
app.py — Streamlit Web Interface for the Upwork API Support Bot
Run with:  streamlit run app.py
"""

import streamlit as st
from rag import initialise, answer_query

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS for polished UI ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background & font ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06);
    color: #c9d1d9;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    text-align: left;
    font-size: 0.82rem;
    padding: 8px 12px;
    transition: all 0.2s ease;
    width: 100%;
    white-space: normal;
    word-wrap: break-word;
    height: auto;
    min-height: 40px;
    line-height: 1.4;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.25);
    border-color: #6366f1;
    color: #ffffff;
    transform: translateX(3px);
}

/* ── Main card container ── */
.main-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Ask button ── */
div[data-testid="stHorizontalBlock"] .stButton > button,
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

/* ── Answer box ── */
.answer-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 20px 24px;
    color: #e6edf3;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ── Metric card ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 20px;
}
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* ── Source expanders ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.88rem !important;
}
.streamlit-expanderContent {
    background: rgba(0,0,0,0.2) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Code blocks inside expanders ── */
.stCode code {
    background: transparent !important;
    font-size: 0.8rem !important;
    color: #a5b4fc !important;
}

/* ── Warning ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Section headers ── */
h3 { color: #e6edf3 !important; font-weight: 600 !important; }

/* ── Badge pills ── */
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in [("current_query", ""), ("auto_run", False), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load RAG pipeline (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading RAG pipeline — first run ~30 s…")
def load_pipeline():
    return initialise(pdf_path="data/upwork_api.pdf")

embed_model, faiss_index, chunks = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Upwork API Bot")
    st.markdown(
        "<span class='badge'>RAG</span>"
        "<span class='badge'>FAISS</span>"
        "<span class='badge'>LLaMA 3.1</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**💡 Example Questions**")
    st.caption("Click any to get an instant answer")

    example_questions = [
        "How long is an OAuth access token valid for?",
        "What are the supported OAuth 2.0 grant types?",
        "Can I use Client Credentials Grant for private contract details?",
        "What is the X-Upwork-API-TenantId header used for?",
        "How do I obtain an authorization code?",
        "What HTTP status code does GraphQL always return?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"btn_{q[:20]}"):
            st.session_state["current_query"] = q
            st.session_state["query_box"] = q       # directly update textarea
            st.session_state["auto_run"] = True
            st.session_state["result"] = None

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;line-height:1.6'>"
        "🔒 Answers grounded in docs only<br>"
        "⚡ Powered by DeepInfra<br>"
        "🧠 Meta-Llama-3.1-8B"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e6edf3;font-size:2rem;font-weight:700;margin-bottom:4px'>"
    "🤖 Upwork API Support Bot"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e;font-size:0.95rem;margin-bottom:20px'>"
    "Ask questions about the <b style='color:#a5b4fc'>Upwork API</b> — "
    "answers are retrieved directly from official documentation."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Input area ────────────────────────────────────────────────────────────────
# Sync sidebar click into the text widget state
if st.session_state["auto_run"]:
    st.session_state["query_box"] = st.session_state["current_query"]

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_area(
        "Your Question",
        placeholder="e.g. How long is an OAuth access token valid for?",
        height=90,
        key="query_box",
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
should_run = (ask_button and user_query.strip()) or st.session_state["auto_run"]

if should_run:
    active_query = (
        st.session_state["current_query"]
        if st.session_state["auto_run"]
        else user_query.strip()
    )
    st.session_state["auto_run"] = False   # reset flag

    if active_query.strip():
        with st.spinner("🤔 Searching documentation and generating answer…"):
            result = answer_query(
                query=active_query.strip(),
                embed_model=embed_model,
                index=faiss_index,
                chunks=chunks,
            )
        st.session_state["result"] = result
    else:
        st.warning("Please type a question or click one from the sidebar.")

elif ask_button and not user_query.strip():
    st.warning("Please enter a question before clicking Ask.")

# ── Display result ────────────────────────────────────────────────────────────
if st.session_state["result"]:
    result = st.session_state["result"]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Answer
    st.markdown("### 💬 Answer")
    st.markdown(
        f"<div class='answer-box'>{result['answer']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("⏱️ API Latency", f"{result['latency']:.2f} s")
    with m2:
        st.metric("📄 Chunks Retrieved", len(result["sources"]))
    with m3:
        st.metric("📝 Answer Length", f"{len(result['answer'])} chars")

    st.divider()

    # Sources
    st.markdown("### 📄 Sources Used")
    st.caption(
        "These exact snippets from the Upwork API docs were passed to the LLM as context."
    )
    for i, source in enumerate(result["sources"], start=1):
        with st.expander(f"📌 Source {i} — click to expand", expanded=(i == 1)):
            st.code(source, language=None)