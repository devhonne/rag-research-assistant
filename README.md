# Research Paper Assistant

A retrieval-augmented generation (RAG) app that answers questions over a
collection of research papers, with every claim cited back to a specific
paper and page.

**Problem:** Students and researchers spend hours searching across many PDFs
to find specific methods, datasets, and results. This project lets you ask a
plain-English question across a set of papers and get an answer with
citations, instead of manually skimming each PDF.

## Demo

Type a question like *"How does G3RAG build graphs?"* and get an answer like:

> G3RAG builds its knowledge graph entirely from the geometric properties of
> document embeddings, without using any large-language-model (LLM) inference
> during construction... [2]
>
> **Sources:** 2609.19622v1 (p.7-8, p.3-4) — scores 0.75

Ask something outside the papers' scope (*"What is the capital of France?"*)
and it correctly responds: *"I don't know based on the provided papers."*

## Architecture

```
PDFs → text extraction (page-aware) → chunking (overlapping, ~450 words)
     → embeddings (BAAI/bge-small-en-v1.5, local, free)
     → vector search (cosine similarity, top-k)
     → LLM generation with citations (Groq, openai/gpt-oss-20b)
     → FastAPI backend → browser chat UI
```

## Tech stack

| Layer | Choice |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | Custom sliding-window splitter, sentence-boundary aware |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free, 384-dim) |
| Vector search | NumPy cosine similarity (in-memory) |
| LLM | Groq API, `openai/gpt-oss-20b` |
| Backend | FastAPI |
| Frontend | Plain HTML/CSS/JS (no framework) |

## Why these choices

- **Local embeddings instead of a paid API** — free, and fast enough for a
  few hundred chunks on a laptop CPU.
- **Groq instead of OpenAI/Anthropic** — free tier with no credit card
  required, while still using a real hosted LLM rather than a toy model.
- **In-memory vector search instead of a database** — I initially built this
  with PostgreSQL + pgvector (Neon), but hit a persistent, unresolved
  connection issue in my dev environment (see Known Limitations). Rather than
  block progress, I kept the pipeline working with local NumPy search and
  documented the trade-off. The retrieval logic is identical either way — a
  database swap only changes where vectors are stored, not how they're used.

## Evaluation

Built a 40-question evaluation set spanning factual, method, numeric,
cross-paper, and deliberately unanswerable questions, with expected answers
verified against the source PDFs.

**Retrieval (does the right paper appear in the top 5?):**
- Hit rate@5: **88%**

**End-to-end answer accuracy (LLM-graded against expected answers):**
- Correct: **62%** (5/8 scored questions)
- Partial: 12.5%
- Wrong (safe refusal, not hallucination): 25%

Full methodology and results in [`findings.md`](./findings.md).

## Known limitations

1. **Retrieval misses on some questions.** Two evaluation questions failed
   because the correct chunk didn't rank in the top 5 — the model correctly
   said "I don't know" rather than guessing, but ideally retrieval would find
   it. Would be improved with hybrid search (BM25 + vectors) and reranking.
2. **Refusal threshold is a blunt instrument.** A single similarity-score
   cutoff (0.45) doesn't cleanly separate in-scope from out-of-scope
   questions, since their score ranges overlap. A more robust approach would
   add an explicit LLM check ("does this context actually answer the
   question?") before generating.
3. **Ambiguous questions can retrieve the wrong specific number** when a
   paper reports multiple related statistics under different conditions. See
   `findings.md` for a documented example and fix (asking a more specific
   question).
4. **No persistent database.** Vectors are loaded into memory from a file at
   startup rather than queried from PostgreSQL, due to an unresolved Neon
   connection issue in development. This works fine for a single-user demo
   but wouldn't scale to multiple concurrent users or large document sets.
5. **Single user, no auth.** Fine for a portfolio demo; a production version
   would need per-user document isolation.

## Running it locally

```bash
# 1. Put PDFs in ./papers, then run in order:
python extract_pdfs.py
python chunk_papers.py
python embed_and_search.py build

# 2. Add your Groq API key to .env:
#    GROQ_API_KEY=gsk_...

# 3. Start the backend
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000

# 4. Open index.html in a browser
```

Optional: run the evaluation suite with `python evaluate.py` (retrieval) and
`python evaluate_answers.py` (end-to-end answer accuracy).

## What I'd build next

- Hybrid search (BM25 + dense vectors) and a cross-encoder reranker to close
  the retrieval gap identified in evaluation
- Swap the refusal threshold for an explicit "can this context answer this
  question?" LLM check
- Revisit PostgreSQL/pgvector for multi-user persistence
- Per-claim citation verification, to catch cases like the ambiguous-question
  failure mode documented in `findings.md`
