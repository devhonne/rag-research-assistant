"""
Step 6: generate real answers with citations, using local search + Groq.

Setup already done:
    pip install groq sentence-transformers numpy
    GROQ_API_KEY in .env

Run:
    python rag_answer.py "How does reranking improve retrieval?"
    python rag_answer.py "How does G3RAG build graphs?" --k 5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GROQ_MODEL = "openai/gpt-oss-20b"  # current Groq model  # fast + free-tier friendly
SCORE_THRESHOLD = 0.45  # below this, we don't even bother calling the LLM


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def retrieve(query, k):
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    emb = np.load(EMB_PATH)
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_prompt(query, results):
    context_blocks = []
    for idx, (c, score) in enumerate(results, start=1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"p.{c['page_start']}-{c['page_end']}"
        context_blocks.append(
            f"[{idx}] (paper: {c['paper']}, {pages})\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a research paper assistant. Answer ONLY using the numbered "
        "context passages below. Cite every claim with its bracket number, "
        "like [1] or [2][3]. If the passages do not contain the answer, "
        "say exactly: \"I don't know based on the provided papers.\" "
        "Never use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
    return system, user


def ask(query, k):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

    results = retrieve(query, k)
    top_score = results[0][1]

    print(f"\nQuestion: {query}")
    print(f"Top retrieval score: {top_score:.3f}\n")

    if top_score < SCORE_THRESHOLD:
        print("Answer: I don't know based on the provided papers.")
        print("(Skipped calling the LLM: nothing relevant enough was retrieved.)")
        return

    system, user = build_prompt(query, results)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=500,
        reasoning_effort="low",
    )
    answer = response.choices[0].message.content
    print("Answer:\n" + answer)

    print("\nSources:")
    for idx, (c, score) in enumerate(results, start=1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"p.{c['page_start']}-{c['page_end']}"
        print(f"  [{idx}] {c['paper']} ({pages}) score={score:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ask(args.query, args.k)


if __name__ == "__main__":
    main()
