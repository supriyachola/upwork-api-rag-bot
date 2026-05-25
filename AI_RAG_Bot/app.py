"""
app.py — Streamlit Web Interface for the Upwork API Support Bot
===============================================================
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

# ── Title & Description ────────────────────────────────────────────────────────
st.title("🤖 Upwork API Technical Support Bot")
st.markdown(
    "Ask any question about the **Upwork API** documentation. "
    "Answers are grounded only in the provided docs — no hallucinations."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Initialise the RAG pipeline ONCE and cache it in Streamlit's session state.
# st.cache_resource ensures the model and FAISS index are loaded only on the
# first request, not on every page interaction.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading RAG pipeline (first run may take ~30 s)…")
def load_pipeline():
    """Load embed model + FAISS index once; reuse across all sessions."""
    return initialise()  # path resolved automatically inside rag.py


embed_model, faiss_index, chunks = load_pipeline()

# ── Sidebar: quick-access example questions ────────────────────────────────────
with st.sidebar:
    st.header("💡 Example Questions")
    example_questions = [
        "How long is an OAuth access token valid for?",
        "What are the supported OAuth 2.0 grant types?",
        "Can I use a Client Credentials Grant to access a user's private contract details?",
        "What is the X-Upwork-API-TenantId header used for?",
        "How do I obtain an authorization code?",
        "What HTTP status code does GraphQL always return?",
    ]
    for q in example_questions:
        # Clicking a button pre-fills the question box
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    st.caption("Powered by DeepInfra · Meta-Llama-3.1-8B · FAISS · LangChain")

# ── Main Input ─────────────────────────────────────────────────────────────────
# If a sidebar button was clicked, pre-fill the text area
default_query = st.session_state.pop("prefill", "")

user_query = st.text_area(
    "Your Question",
    value=default_query,
    placeholder="e.g. How long is an OAuth access token valid for?",
    height=80,
)

ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# ── Process Query ──────────────────────────────────────────────────────────────
if ask_button and user_query.strip():
    with st.spinner("Thinking…"):
        result = answer_query(
            query=user_query.strip(),
            embed_model=embed_model,
            index=faiss_index,
            chunks=chunks,
        )

    # ── B3: Display Requirements ───────────────────────────────────────────────

    # 1. AI Answer
    st.subheader("💬 Answer")
    st.markdown(result["answer"])

    st.divider()

    # 2. Latency metric
    st.metric(
        label="⏱️ API Latency",
        value=f"{result['latency']:.2f} s",
        help="Time taken for the DeepInfra API to return a response",
    )

    st.divider()

    # 3. Retrieved source chunks
    st.subheader("📄 Sources Used")
    st.caption(
        "These are the exact snippets retrieved from the Upwork API documentation "
        "that were passed to the LLM as context."
    )
    for i, source in enumerate(result["sources"], start=1):
        with st.expander(f"Source {i}", expanded=(i == 1)):
            st.code(source, language=None)   # monospace makes it easy to read

elif ask_button and not user_query.strip():
    st.warning("Please enter a question before clicking Ask.")
