"""
Step 4: store chunks + vectors in PostgreSQL (Neon) and search from the database.

Setup (one time):
    pip install psycopg2-binary pgvector python-dotenv

Commands:
    python db_load.py load                       # create tables and load all chunks
    python db_load.py ask "How does G3RAG build graphs?"
    python db_load.py ask "question" --paper 2609.19622v1   # search one paper only
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    paper       TEXT NOT NULL,
    page_start  INT,
    page_end    INT,
    n_words     INT,
    text        TEXT NOT NULL,
    embedding   vector({DIM})
);

CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks (paper);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect():
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url or "USERNAME" in url or "PASSWORD" in url:
        raise SystemExit("DATABASE_URL is missing or still a placeholder. "
                         "Check your .env file.")
    conn = psycopg2.connect(url)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def load():
    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
    emb = np.load(EMB_PATH)
    assert len(chunks) == len(emb), "chunks and embeddings do not match; rebuild embeddings"

    conn = connect()
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("TRUNCATE chunks RESTART IDENTITY;")  # safe to re-run
        rows = [
            (c["chunk_id"], c["paper"], c["page_start"], c["page_end"],
             c["n_words"], c["text"], emb[i])
            for i, c in enumerate(chunks)
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (chunk_id, paper, page_start, page_end, "
            "n_words, text, embedding) VALUES %s",
            rows,
        )
        cur.execute("SELECT COUNT(*) FROM chunks;")
        print(f"Loaded. Rows in database: {cur.fetchone()[0]}")
    conn.commit()
    conn.close()


def ask(query, k, paper):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    conn = connect()
    sql = """
        SELECT paper, page_start, page_end, text,
               1 - (embedding <=> %s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    params = [q]
    where = ""
    if paper:
        where = "WHERE paper = %s"
        params.append(paper)
    params += [q, k]

    with conn.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        rows = cur.fetchall()
    conn.close()

    print(f"\nQuestion: {query}\n")
    for r, (p, ps, pe, text, score) in enumerate(rows, start=1):
        pages = f"p.{ps}" if ps == pe else f"p.{ps}-{pe}"
        print(f"[{r}] score={score:.3f} | {p} | {pages}")
        print("    " + text[:300].replace("\n", " ") + " ...\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("load")
    a = sub.add_parser("ask")
    a.add_argument("query")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--paper", default=None)
    args = ap.parse_args()

    if args.cmd == "load":
        load()
    else:
        ask(args.query, args.k, args.paper)


if __name__ == "__main__":
    main()
