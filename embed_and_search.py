"""
Step 3 of the Research Paper Assistant: embeddings + first retrieval test.

Setup (one time):
    pip install sentence-transformers numpy

Build embeddings (run once, takes a few minutes on CPU):
    python embed_and_search.py build

Ask a question and see the top matching chunks:
    python embed_and_search.py ask "How does reranking improve retrieval?"
    python embed_and_search.py ask "What chunk size works best?" --k 8
"""
import argparse
import json
from pathlib import Path

import numpy as np

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # free, small, good quality
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def build():
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with {MODEL_NAME} ...")
    model = get_model()
    texts = [c["text"] for c in chunks]
    # normalize so that dot product = cosine similarity
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)
    np.save(EMB_PATH, emb)
    print(f"Saved {emb.shape} to {EMB_PATH}")


def ask(query, k):
    chunks = load_chunks()
    emb = np.load(EMB_PATH)
    model = get_model()
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    scores = emb @ q
    top = np.argsort(-scores)[:k]

    print(f"\nQuestion: {query}\n")
    for rank, i in enumerate(top, start=1):
        c = chunks[i]
        pages = (f"p.{c['page_start']}" if c["page_start"] == c["page_end"]
                 else f"p.{c['page_start']}-{c['page_end']}")
        print(f"[{rank}] score={scores[i]:.3f} | {c['paper']} | {pages}")
        print("    " + c["text"][:350].replace("\n", " ") + " ...\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("query")
    a.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    else:
        ask(args.query, args.k)


if __name__ == "__main__":
    main()
