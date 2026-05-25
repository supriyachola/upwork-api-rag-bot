"""
rag.py — Core RAG (Retrieval-Augmented Generation) Pipeline
============================================================
This module handles everything except the UI:
  1. Loading the PDF
  2. Splitting into chunks
  3. Embedding chunks with a local model
  4. Storing/loading a FAISS vector index
  5. Retrieving relevant chunks for a query
  6. Calling the DeepInfra LLM API and returning an answer
"""

import os
import time
import requests
import pickle
from pathlib import Path

# PyMuPDF for reading PDFs (imported as fitz)
import fitz  # pip install pymupdf

# sentence-transformers runs locally — no API key needed
from sentence_transformers import SentenceTransformer

# FAISS is a fast vector similarity search library from Meta
import faiss
import numpy as np

from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────────
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

# ── Constants — tweak these if you want to experiment ─────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # characters of overlap between consecutive chunks
TOP_K         = 3     # how many chunks to retrieve per query
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
API_URL       = "https://api.deepinfra.com/v1/openai/chat/completions"

# Paths where we cache the FAISS index so we don't re-embed every run
FAISS_INDEX_PATH  = "faiss_index.bin"
CHUNKS_CACHE_PATH = "chunks_cache.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Knowledge Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(pdf_path: str) -> str:
    """
    A1. Data Ingestion
    ------------------
    Read every page of the PDF and concatenate the text.
    We use PyMuPDF (fitz) because it handles multi-column layouts better
    than many alternatives.

    Returns the full document as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()   # extract plain text from each page
    doc.close()
    return full_text


def sanity_check(text: str):
    """
    A1. Sanity Check
    ----------------
    Print the total character count and a 300-character preview so we can
    confirm the PDF was read correctly before doing anything expensive.
    """
    print("=" * 60)
    print(f"[Sanity Check] Total characters loaded: {len(text):,}")
    print(f"[Sanity Check] Sample (first 300 chars):\n{text[:300]}")
    print("=" * 60)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    A2. Document Chunking
    ---------------------
    Slide a window of `chunk_size` characters across the text, advancing by
    (chunk_size - overlap) each step.

    WHY OVERLAP?
    Technical docs often contain multi-line code blocks and parameter
    descriptions that span chunk boundaries.  Without overlap, a chunk might
    end in the middle of an OAuth parameter description; the next chunk would
    start mid-sentence, giving the retriever incomplete context.  By repeating
    the last `overlap` characters in the next chunk, we guarantee that every
    important sentence appears *complete* in at least one chunk.

    Example (chunk_size=10, overlap=3):
      Text: "abcdefghijklmno"
      Chunk 1: "abcdefghij"   (0-9)
      Chunk 2: "hijklmnopq"   (7-16)   ← "hij" is the overlap
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():          # skip chunks that are only whitespace
            chunks.append(chunk)
        # Advance by (chunk_size - overlap) so next chunk shares `overlap` chars
        start += chunk_size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# PART A3 — Embedding & Vector Storage
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    """
    Load the local sentence-transformer model.
    'all-MiniLM-L6-v2' is small (~80 MB), fast, and good enough for
    technical documentation retrieval.  It runs entirely on your CPU —
    no API key required.
    """
    print(f"[Embeddings] Loading local model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    return model


def build_faiss_index(chunks: list[str],
                      embed_model: SentenceTransformer):
    """
    A3. Vector Storage
    ------------------
    1. Embed every chunk → numpy array of shape (N, 384)
    2. Build a FAISS flat L2 index (brute-force exact search — fine for
       a few hundred chunks).
    3. Save the index and chunk list to disk so we can reload without
       re-embedding next time.

    FAISS stores only the *vectors*; we keep the original text in a
    parallel list (`chunks`) so we can return the source text later.
    """
    print(f"[FAISS] Embedding {len(chunks)} chunks — this may take a minute…")
    # encode() returns a 2-D numpy array: one row per chunk
    embeddings = embed_model.encode(chunks, show_progress_bar=True,
                                    convert_to_numpy=True)

    # Dimension of each embedding vector
    dim = embeddings.shape[1]

    # IndexFlatL2 = exact nearest-neighbour search using Euclidean distance.
    # Good choice when the dataset is small (< ~100k vectors).
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)   # add all vectors to the index

    # Persist to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[FAISS] Index built with {index.ntotal} vectors, saved to disk.")
    return index, chunks


def load_faiss_index():
    """
    Load a previously saved FAISS index and chunk list from disk.
    Returns (index, chunks) or (None, None) if not found.
    """
    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_CACHE_PATH).exists():
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_CACHE_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"[FAISS] Loaded existing index ({index.ntotal} vectors).")
        return index, chunks
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# PART B1 — Semantic Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_top_chunks(query: str,
                        embed_model: SentenceTransformer,
                        index,
                        chunks: list[str],
                        top_k: int = TOP_K) -> list[str]:
    """
    B1. Semantic Retrieval
    ----------------------
    Convert the user's question into an embedding, then ask FAISS for the
    `top_k` nearest stored vectors.  Return the corresponding text chunks.

    HOW RETRIEVAL WORKS:
    1. The user types "How long is an OAuth token valid?"
    2. We embed that question → a 384-dim vector Q.
    3. FAISS computes the L2 distance between Q and every stored chunk vector.
    4. It returns the indices of the 3 smallest distances (most similar).
    5. We look those indices up in our `chunks` list → raw text snippets.
    6. Those snippets become the *context* we feed to the LLM.
    """
    # Embed the query (shape: (1, 384))
    query_vec = embed_model.encode([query], convert_to_numpy=True)

    # search() returns (distances, indices) — we only need indices
    _, indices = index.search(query_vec, top_k)

    # indices[0] is a 1-D array of length top_k
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved


# ══════════════════════════════════════════════════════════════════════════════
# PART B2 — LLM API Integration & Prompting
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with 10+ years of experience.
Your job is to answer developer questions about the Upwork API clearly and accurately.

STRICT RULES:
1. Answer ONLY using the information in the CONTEXT provided below.
2. If the context does not contain the answer, you MUST respond with exactly:
   "I'm sorry, but the provided documentation does not contain that information."
3. Do NOT guess, invent, or use any knowledge outside the provided context.
4. Format code examples with proper code blocks.
5. Be concise and professional.
"""


def call_llm(query: str, context_chunks: list[str]) -> tuple[str, float]:
    """
    B2. API Integration
    -------------------
    Build a prompt from the retrieved chunks, send it to the DeepInfra
    LLM endpoint (OpenAI-compatible format), and return the answer + latency.

    We use the `requests` library directly (no extra SDK needed) so you can
    see exactly what is being sent over the wire.

    Returns:
        answer  (str)   — the LLM's text response
        latency (float) — seconds the API call took
    """
    if not DEEPINFRA_API_KEY:
        raise ValueError("DEEPINFRA_API_KEY not set. Check your .env file.")

    # Join the retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    # The user message contains both the context and the question.
    # Putting context here (rather than the system prompt) keeps the system
    # prompt reusable across multiple turns.
    user_message = f"""CONTEXT (from Upwork API documentation):
{context}

QUESTION:
{query}

Answer the question using only the context above."""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,   # low temperature = more factual, less creative
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",   # API key goes HERE
    }

    start = time.time()
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    latency = time.time() - start

    # Raise an exception if the HTTP status is 4xx or 5xx
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    return answer, latency


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Full RAG Pipeline (Entry Point)
# ══════════════════════════════════════════════════════════════════════════════

def answer_query(query: str,
                 embed_model: SentenceTransformer,
                 index,
                 chunks: list[str]) -> dict:
    """
    Orchestrate the full RAG pipeline for a single user question:
      retrieve → prompt → call LLM → return structured result

    Returns a dict with:
        answer   — the LLM's response string
        sources  — list of retrieved chunk strings
        latency  — API response time in seconds
    """
    # Step 1: Find the most relevant chunks from our vector store
    top_chunks = retrieve_top_chunks(query, embed_model, index, chunks)

    # Step 2: Call the LLM with those chunks as context
    answer, latency = call_llm(query, top_chunks)

    return {
        "answer":  answer,
        "sources": top_chunks,
        "latency": latency,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation helper — called once when Streamlit starts
# ══════════════════════════════════════════════════════════════════════════════

def initialise(pdf_path: str = "data/upwork_api.pdf"):
    """
    Load or build everything needed for the RAG pipeline.
    If a cached FAISS index exists we skip the expensive embedding step.
    Returns (embed_model, index, chunks).
    """
    embed_model = load_embedding_model()

    # Try loading from disk first (fast path)
    index, chunks = load_faiss_index()

    if index is None:
        # First run: read PDF, chunk it, embed, save
        text = load_pdf(pdf_path)
        sanity_check(text)
        chunks = split_into_chunks(text)
        print(f"[Chunking] Created {len(chunks)} chunks "
              f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        index, chunks = build_faiss_index(chunks, embed_model)
    else:
        print("[Init] Using cached index — skipping PDF processing.")

    return embed_model, index, chunks
