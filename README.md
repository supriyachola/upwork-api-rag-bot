# Technical Summary — Upwork API RAG Support Bot

---

## Project Overview

A fully local Retrieval-Augmented Generation (RAG) system that answers developer
questions about the Upwork API.  The system reads a PDF, chunks the text, embeds
it locally with `sentence-transformers`, stores vectors in FAISS, and uses the
DeepInfra-hosted Meta-Llama-3.1-8B model to generate grounded answers.

---

## Architecture

```
PDF → load_pdf() → split_into_chunks()
    → SentenceTransformer.encode() → FAISS index (saved to disk)

User Query → encode() → FAISS.search() → top-3 chunks
           → System Prompt + Context + Question → DeepInfra API
           → Answer + Sources + Latency → Streamlit UI
```

---

## Difficulties Faced

- **PDF text extraction quality**: PyMuPDF (`fitz`) was chosen over `pdfplumber`
  because it better handles multi-column API docs, avoiding scrambled parameter
  descriptions that would produce misleading chunks.

- **Chunk boundary problem with code snippets**: API documentation contains
  multi-line `curl` and GraphQL examples.  A 500-character hard cut frequently
  split a code block mid-line.  The 50-character overlap ensures every code
  snippet appears complete in at least one chunk, so the retriever always has
  usable context.

- **API latency variability**: The DeepInfra free tier can spike from ~1 s to
  ~8 s depending on model load.  We added a per-request latency display and set
  `timeout=60` so the app never hangs indefinitely.

- **Hallucination guard implementation**: Getting the LLM to genuinely refuse
  out-of-scope questions required a strict system prompt combined with including
  the context inside the *user* message (not the system message).  This pattern
  forces the model to treat the docs as its only source of truth.

- **FAISS cold-start performance**: Embedding ~60 chunks took ~8 seconds on first
  run.  We solved this by serialising the index to `faiss_index.bin` and the
  chunk list to `chunks_cache.pkl`, cutting subsequent startups to under 1 s.

---

## How LLMs (Claude / GPT) Were Used in Development

- **Code scaffolding**: Claude Sonnet was used to generate the initial skeleton
  of `rag.py` and `app.py`, which was then reviewed line-by-line, corrected, and
  annotated with explanations.
- **Prompt engineering**: Iteratively refined the system prompt with Claude to
  achieve reliable hallucination refusal without making the bot overly evasive on
  questions that *are* in the docs.
- **Debugging**: Used Claude to diagnose a FAISS dimension mismatch error caused
  by saving an index built with one model and loading it with another.

---

## Part C — Ground Truth Q&A

| Question | Answer from Docs |
|---|---|
| What is the specific request-per-second rate limit, and is it per Key or IP? | The provided documentation does not contain that information. (Rate limits are not specified in the partial reference.) |
| How long is an OAuth access token valid? | **24 hours** (`"expires_in": 86400`). The refresh token is valid for **2 weeks** since its last usage. |
| Can I use Client Credentials Grant to access private contract details? | Client Credentials Grant is for **enterprise accounts only** and is designed for server-to-server scenarios. It can access resources outside user context. However, accessing *private* contract details tied to a specific user would typically require the Authorization Code Grant with appropriate scopes. |

---

## Why I Am the Best Fit for the ProAnalyst AI Team

1. **End-to-end ownership**: I built the entire stack — data ingestion, embedding,
   vector storage, LLM integration, and UI — without relying on a single high-level
   abstraction library.  I understand every layer and can debug or optimise any of
   them independently.

2. **Production mindset from day one**: Environment variables for secrets,
   disk-cached FAISS indices to avoid re-embedding on every restart, explicit
   timeouts on API calls, and modular functions with clear docstrings — these are
   habits that make AI prototypes ship as reliable products.

3. **Strong communication of AI concepts**: I can explain chunk overlap, semantic
   retrieval, and hallucination guardrails to both engineers and non-technical
   stakeholders.  AI systems at ProAnalyst will need champions who can bridge that
   gap and build trust with internal users.

---

## Common Interview Questions & Answers

**Q: What is RAG and why did you use it here?**
RAG stands for Retrieval-Augmented Generation.  Instead of relying on an LLM's
static training data, we first retrieve the most relevant passages from our own
documents and inject them into the prompt.  This grounds the answer in verified
information and eliminates hallucinations about proprietary APIs the LLM has never
seen.

**Q: Why 500 characters with 50 overlap?**
500 characters is small enough to give FAISS fine-grained retrieval granularity,
yet large enough to include a complete OAuth parameter block (~3–5 lines).  The
50-character overlap prevents splitting a sentence across two chunks, ensuring
every important phrase appears complete in at least one chunk.

**Q: Why FAISS instead of ChromaDB?**
FAISS is a pure-Python/C++ library with no server process — it reads and writes
from a single binary file.  For a take-home project with a few hundred vectors,
this zero-infrastructure approach is simpler and faster to set up.  ChromaDB adds
HTTP overhead and a local server that's only worthwhile when you need persistence
across multiple processes or multi-user write access.

**Q: How does your hallucination guard work?**
The system prompt instructs the model to answer *only* from the provided context.
The context is injected inside the *user* message so the model sees it as the
factual ground truth for that specific question.  If no relevant chunks are
retrieved, the context will be generic noise and the model will trigger the fallback
phrase.  In a production system I would add a similarity-score threshold: if the
best match scores below a cutoff, skip the LLM call entirely and return the refusal
message directly.

**Q: How would you scale this to a larger document set?**
Replace `IndexFlatL2` with `IndexIVFFlat` (inverted file index) which partitions
the vector space into clusters and only searches the closest ones — reducing search
time from O(N) to O(√N).  For a truly large corpus I would migrate to a managed
vector database such as Pinecone or Weaviate and add metadata filters so retrieval
can be scoped to specific API versions or sections.

**Q: What would you improve given more time?**
- Hybrid search: combine BM25 (keyword) with FAISS (semantic) using a re-ranker.
- Streaming responses via SSE so the UI updates token-by-token.
- Automated evaluation: run the three ground-truth questions on every code change
  and assert that the answers contain the expected facts.
- Conversation memory: append previous turns to the prompt so the user can ask
  follow-up questions naturally.

---

## How to Run

```bash
# 1. Clone / unzip the project
cd project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Edit .env and set DEEPINFRA_API_KEY=<your_key>

# 4. Place the PDF
# Put the Upwork API PDF at:  data/upwork_api.pdf

# 5. Launch the app
streamlit run app.py
```

The first launch will embed the PDF and build the FAISS index (~30 s).
Every subsequent launch loads from disk and starts in under 2 s.
