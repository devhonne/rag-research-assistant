"""
Step 8: a tiny API so a web page can ask questions.

Setup:
    pip install fastapi uvicorn

Run:
    uvicorn api:app --reload --port 8000

Then open index.html in your browser (Step 9) and start asking questions.
This reuses the exact same retrieval + generation logic as rag_answer.py.
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GROQ_MODEL = "openai/gpt-oss-20b"
SCORE_THRESHOLD = 0.45

load_dotenv()
app = FastAPI()

# Allow index.html (opened as a local file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model and data (happens once at startup)...")
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer(MODEL_NAME)
_chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
_emb = np.load(EMB_PATH)
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(f"Ready. {len(_chunks)} chunks loaded.")


class Question(BaseModel):
    question: str
    k: int = 5


def retrieve(query, k):
    q = _model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = _emb @ q
    top = np.argsort(-scores)[:k]
    return [(_chunks[i], float(scores[i])) for i in top]


def build_prompt(query, results):
    blocks = []
    for idx, (c, score) in enumerate(results, start=1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"p.{c['page_start']}-{c['page_end']}"
        blocks.append(f"[{idx}] (paper: {c['paper']}, {pages})\n{c['text']}")
    context = "\n\n".join(blocks)
    system = (
        "You are a research paper assistant. Answer ONLY using the numbered "
        "context passages below. Cite every claim with its bracket number, "
        "like [1] or [2][3]. If the passages do not contain the answer, say "
        "exactly: \"I don't know based on the provided papers.\" Never use "
        "outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
    return system, user


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_chunks)}


@app.post("/ask")
def ask(payload: Question):
    results = retrieve(payload.question, payload.k)
    top_score = results[0][1]

    sources = [
        {
            "paper": c["paper"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "score": round(score, 3),
            "snippet": c["text"][:200],
        }
        for c, score in results
    ]

    if top_score < SCORE_THRESHOLD:
        return {
            "answer": "I don't know based on the provided papers.",
            "top_score": round(top_score, 3),
            "sources": sources,
            "skipped_llm": True,
        }

    system, user = build_prompt(payload.question, results)
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=600,
        reasoning_effort="low",
    )
    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "top_score": round(top_score, 3),
        "sources": sources,
        "skipped_llm": False,
    }
