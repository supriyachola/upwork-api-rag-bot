"""
rag.py — Core RAG (Retrieval-Augmented Generation) Pipeline
===========================================================
This module handles:
  1. Loading the PDF
  2. Splitting it into chunks
  3. Embedding chunks with a local model
  4. Storing/loading a FAISS vector index
  5. Retrieving relevant chunks for a query
  6. Calling the Google Gemini API and returning an answer
"""

import os
import pickle
import time
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer


# Load .env from this file's directory, regardless of the terminal's cwd.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Use the model ID recommended by the error returned by your Gemini API.
# If Google reports that it is unavailable, choose a model listed in AI Studio.
LLM_MODEL = "gemini-3.8-flash"

# Retrieval settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cached FAISS index and chunk list
FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index.bin")
CHUNKS_CACHE_PATH = str(BASE_DIR / "chunks_cache.pkl")


# ---------------------------------------------------------------------------
# Part A: Knowledge Engineering
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str) -> str:
    """Read all pages of a PDF and return the extracted text."""
    with fitz.open(pdf_path) as doc:
        return "".join(page.get_text() for page in doc)


def sanity_check(text: str) -> None:
    """Print text length and a short preview to confirm PDF extraction."""
    print("=" * 60)
    print(f"[Sanity Check] Total characters loaded: {len(text):,}")
    print(f"[Sanity Check] Sample (first 300 chars):\n{text[:300]}")
    print("=" * 60)


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Part A3: Embedding and Vector Storage
# ---------------------------------------------------------------------------

def load_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformer model."""
    print(f"[Embeddings] Loading local model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def build_faiss_index(
    chunks: list[str],
    embed_model: SentenceTransformer,
):
    """Embed chunks, build a FAISS index, and cache the index and chunks."""
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"[FAISS] Embedding {len(chunks)} chunks — this may take a minute...")
    embeddings = embed_model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_CACHE_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print(f"[FAISS] Index built with {index.ntotal} vectors and saved.")
    return index, chunks


def load_faiss_index():
    """Load the cached FAISS index and chunks, if both exist."""
    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_CACHE_PATH).exists():
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_CACHE_PATH, "rb") as file:
            chunks = pickle.load(file)
        print(f"[FAISS] Loaded existing index ({index.ntotal} vectors).")
        return index, chunks

    return None, None


# ---------------------------------------------------------------------------
# Part B1: Semantic Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_chunks(
    query: str,
    embed_model: SentenceTransformer,
    index,
    chunks: list[str],
    top_k: int = TOP_K,
) -> list[str]:
    """Retrieve the most similar text chunks for a question."""
    query_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype(np.float32)

    k = min(top_k, len(chunks))
    if k <= 0:
        return []

    _, indices = index.search(query_vec, k)
    return [
        chunks[i]
        for i in indices[0]
        if 0 <= i < len(chunks)
    ]


# ---------------------------------------------------------------------------
# Part B2: Gemini API Integration and Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Upwork API documentation assistant.

Answer the user's question using the supplied documentation context.

Rules:
1. Read all supplied context chunks before answering.
2. Use relevant facts from any chunk, even if the answer is spread across multiple chunks.
3. Explain the information clearly and directly.
4. You may reorganize or summarize documented facts, but do not invent details.
5. If the context does not contain enough information to answer the question,
   say exactly:
   "I'm sorry, but the provided documentation does not contain that information."
6. If only part of the answer is documented, explain that part and identify
   what the documentation does not specify.
7. Format code examples using Markdown code blocks.
"""


def call_llm(query: str, context_chunks: list[str]) -> tuple[str, float]:
    """Call Gemini with retrieved context and return the answer plus latency."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to the .env file in AI_RAG_Bot."
        )

    if not context_chunks:
        return (
            "I'm sorry, but the provided documentation does not contain that information.",
            0.0,
        )

    context = "\n\n---\n\n".join(context_chunks)
    user_message = f"""CONTEXT (from Upwork API documentation):
{context}

QUESTION:
{query}

Answer the question using only the context above."""

    client = genai.Client(api_key=GEMINI_API_KEY)
    start = time.time()
    response = None

    # Retry temporary Gemini server errors (e.g. HTTP 503) up to 3 attempts.
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=user_message,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
            )
            break
        except errors.ServerError as exc:
            if attempt == 2:
                raise RuntimeError(
                    "Gemini is temporarily unavailable after 3 attempts. "
                    "Please wait a little and try again."
                ) from exc
            time.sleep(2 ** (attempt + 1))  # wait 2 seconds, then 4 seconds

    latency = time.time() - start

    answer = response.text if response else None
    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response. Check model availability and API quota."
        )

    return answer, latency


# ---------------------------------------------------------------------------
# Part B: Full RAG Pipeline
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    embed_model: SentenceTransformer,
    index,
    chunks: list[str],
) -> dict:
    """Retrieve relevant chunks, call Gemini, and return a structured result."""
    top_chunks = retrieve_top_chunks(query, embed_model, index, chunks)
    answer, latency = call_llm(query, top_chunks)

    return {
        "answer": answer,
        "sources": top_chunks,
        "latency": latency,
    }


# ---------------------------------------------------------------------------
# Initialization helper — called once when Streamlit starts
# ---------------------------------------------------------------------------

def initialise(pdf_path: str = None):
    """Load or build the local embedding model and FAISS index."""
    if pdf_path is None:
        pdf_path = str(BASE_DIR / "data" / "upwork_api.pdf")

    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}. "
            "Ensure data/upwork_api.pdf exists inside AI_RAG_Bot."
        )

    embed_model = load_embedding_model()
    index, chunks = load_faiss_index()

    if index is None:
        text = load_pdf(pdf_path)
        sanity_check(text)
        chunks = split_into_chunks(text)
        print(
            f"[Chunking] Created {len(chunks)} chunks "
            f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
        )
        index, chunks = build_faiss_index(chunks, embed_model)
    else:
        print("[Init] Using cached index — skipping PDF processing.")

    return embed_model, index, chunks
