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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
[data-testid="stHeader"] { background: transparent; }
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
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.answer-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 20px 24px;
    color: #e6edf3;
    line-height: 1.7;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 20px;
}
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 1.6rem !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# ── Load RAG pipeline (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading RAG pipeline — first run ~30 s…")
def load_pipeline():
    return initialise()

embed_model, faiss_index, chunks = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Upwork API Bot")
    st.divider()
    st.markdown("**💡 Example Questions**")
    st.caption("Click any to instantly get an answer ↓")

    example_questions = [
        "How long is an OAuth access token valid for?",
        "What are the supported OAuth 2.0 grant types?",
        "Can I use Client Credentials Grant for private contract details?",
        "What is the X-Upwork-API-TenantId header used for?",
        "How do I obtain an authorization code?",
        "What HTTP status code does GraphQL always return?",
    ]

    for q in example_questions:
        # Each button directly stores the question AND result in one click
        if st.button(q, use_container_width=True, key=f"sidebar_{q[:30]}"):
            st.session_state.selected_question = q
            # Run the pipeline immediately here — no auto_run flag needed
            with st.spinner("🤔 Searching and generating answer…"):
                st.session_state.result = answer_query(
                    query=q,
                    embed_model=embed_model,
                    index=faiss_index,
                    chunks=chunks,
                )

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;line-height:1.8'>"
        "🔒 Answers grounded in docs only<br>"
        "⚡ Powered by DeepInfra<br>"
        "🧠 Meta-Llama-3.1-8B<br>"
        "🗄️ FAISS Vector Search"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e6edf3;font-size:2rem;font-weight:700;margin-bottom:4px'>"
    "🤖 Upwork API Support Bot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e;font-size:0.95rem;margin-bottom:20px'>"
    "Ask questions about the <b style='color:#a5b4fc'>Upwork API</b> — "
    "answers retrieved directly from official documentation."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Manual input area ─────────────────────────────────────────────────────────
# Show the selected question from sidebar (if any) as placeholder
placeholder_text = (
    st.session_state.selected_question
    if st.session_state.selected_question
    else "e.g. How long is an OAuth access token valid for?"
)

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_area(
        "Your Question",
        value=st.session_state.selected_question,
        placeholder=placeholder_text,
        height=90,
        key="manual_input",
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

# ── Handle manual Ask button ──────────────────────────────────────────────────
if ask_btn:
    if user_query.strip():
        st.session_state.selected_question = user_query.strip()
        with st.spinner("🤔 Searching and generating answer…"):
            st.session_state.result = answer_query(
                query=user_query.strip(),
                embed_model=embed_model,
                index=faiss_index,
                chunks=chunks,
            )
    else:
        st.warning("⚠️ Please enter a question or click one from the sidebar.")

# ── Display result ────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### 💬 Answer")
    st.markdown(
        f"<div class='answer-box'>{result['answer']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("⏱️ API Latency",      f"{result['latency']:.2f} s")
    with m2:
        st.metric("📄 Chunks Retrieved", len(result["sources"]))
    with m3:
        st.metric("📝 Answer Length",    f"{len(result['answer'])} chars")

    st.divider()
    st.markdown("### 📄 Sources Used")
    st.caption("Exact snippets from Upwork API docs passed to the LLM as context.")
    for i, source in enumerate(result["sources"], start=1):
        with st.expander(f"📌 Source {i}", expanded=(i == 1)):
            st.code(source, language=None)
