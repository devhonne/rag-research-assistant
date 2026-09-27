"""
Step 3.5: evaluate your search.

Reads eval_questions.json, runs each question through your vector search,
and reports how often the right paper appears in the top results.

Run:
    python evaluate.py
    python evaluate.py --label baseline     # saves results to eval_results/baseline.json
    python evaluate.py --k 5 --threshold 0.55
"""
import argparse
import json
from pathlib import Path

import numpy as np

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
QUESTIONS_PATH = Path("eval_questions.json")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def is_match(chunk, q):
    """A chunk is a hit if it comes from the right paper
    (and overlaps the right pages, when pages are given)."""
    if chunk["paper"] != q["paper"]:
        return False
    pages = q.get("pages") or []
    if not pages:
        return True
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="top score below this counts as 'I don't know'")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
    emb = np.load(EMB_PATH)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    hits, rr_list = [], []
    refusals = []
    details = []

    for q in questions:
        qv = model.encode([QUERY_PREFIX + q["question"]],
                          normalize_embeddings=True)[0]
        scores = emb @ qv
        top = np.argsort(-scores)[: args.k]
        top_score = float(scores[top[0]])

        if q["type"] == "unanswerable":
            ok = top_score < args.threshold
            refusals.append(ok)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] #{q['id']} (unanswerable) top score {top_score:.3f}")
            details.append({"id": q["id"], "type": q["type"], "pass": ok,
                            "top_score": top_score})
            continue

        if q["paper"] is None:
            print(f"[SKIP] #{q['id']} ({q['type']}) check manually")
            for i in top:
                print(f"        {chunks[i]['paper']} p.{chunks[i]['page_start']}"
                      f" score={scores[i]:.3f}")
            continue

        rank = None
        for r, i in enumerate(top, start=1):
            if is_match(chunks[i], q):
                rank = r
                break
        hit = rank is not None
        hits.append(hit)
        rr_list.append(1 / rank if rank else 0.0)
        status = f"HIT rank {rank}" if hit else "MISS"
        print(f"[{status}] #{q['id']} {q['question'][:70]}")
        if not hit:
            got = ", ".join(f"{chunks[i]['paper']}" for i in top[:3])
            print(f"        expected {q['paper']} | got {got}")
        details.append({"id": q["id"], "type": q["type"], "hit": hit,
                        "rank": rank, "top_score": top_score})

    print("\n========== RESULTS ==========")
    if hits:
        hit_rate = sum(hits) / len(hits)
        mrr = sum(rr_list) / len(rr_list)
        print(f"Hit rate@{args.k}: {hit_rate:.0%}  ({sum(hits)}/{len(hits)})")
        print(f"MRR: {mrr:.3f}")
    if refusals:
        print(f"Correct 'I don't know' (score < {args.threshold}): "
              f"{sum(refusals)}/{len(refusals)}")

    if args.label:
        Path("eval_results").mkdir(exist_ok=True)
        out = Path("eval_results") / f"{args.label}.json"
        summary = {
            "label": args.label,
            "k": args.k,
            "hit_rate": sum(hits) / len(hits) if hits else None,
            "mrr": sum(rr_list) / len(rr_list) if rr_list else None,
            "refusal_pass": sum(refusals),
            "refusal_total": len(refusals),
            "details": details,
        }
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
