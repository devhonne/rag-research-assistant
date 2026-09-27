"""
Step 7: end-to-end evaluation (retrieval + generation + correctness).

Runs every question in eval_questions.json through your real pipeline
(search -> Groq answer with citations), then asks the LLM to judge
whether the generated answer matches the expected answer.

Run:
    python evaluate_answers.py
    python evaluate_answers.py --label v1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Windows' default terminal encoding (cp1252) can't print some characters
# that LLMs generate (curly quotes, narrow spaces, etc.). Force UTF-8 so
# the script doesn't crash mid-run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNKS_PATH = Path("chunks/chunks.jsonl")
EMB_PATH = Path("chunks/embeddings.npy")
QUESTIONS_PATH = Path("eval_questions.json")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GROQ_MODEL = "openai/gpt-oss-20b"
SCORE_THRESHOLD = 0.45


def load_chunks():
    return [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]


def retrieve(model, emb, chunks, query, k=5):
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_prompt(query, results):
    blocks = []
    for idx, (c, score) in enumerate(results, start=1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"p.{c['page_start']}-{c['page_end']}"
        blocks.append(f"[{idx}] (paper: {c['paper']}, {pages})\n{c['text']}")
    context = "\n\n".join(blocks)
    system = (
        "You are a research paper assistant. Answer ONLY using the numbered "
        "context passages below. Cite every claim with its bracket number. "
        "If the passages do not contain the answer, say exactly: "
        "\"I don't know based on the provided papers.\" Never use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
    return system, user


def judge(client, question, expected, generated):
    """Ask the LLM to grade the generated answer against the expected answer."""
    prompt = (
        "You are grading a RAG system's answer.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Generated answer: {generated}\n\n"
        "Does the generated answer correctly convey the expected answer's key facts? "
        "Minor wording differences are fine. Do not explain your reasoning. "
        "Your entire reply must be a single word with no punctuation: "
        "CORRECT, PARTIAL, or WRONG."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
        reasoning_effort="low",
    )
    raw = resp.choices[0].message.content or ""
    verdict = raw.strip().upper()
    for v in ("CORRECT", "PARTIAL", "WRONG"):
        if v in verdict:
            return v
    print(f"    [judge raw output, no match found]: {raw[:200]!r}")
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    emb = np.load(EMB_PATH)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    results_log = []
    counts = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "UNKNOWN": 0,
             "refusal_pass": 0, "refusal_total": 0, "skipped": 0}

    for q in questions:
        print(f"\n--- #{q['id']} ({q['type']}) {q['question'][:70]}")

        if q["paper"] is None and q["type"] != "unanswerable":
            print("  SKIP (cross-paper, needs manual check)")
            counts["skipped"] += 1
            continue

        retrieved = retrieve(model, emb, chunks, q["question"], args.k)
        top_score = retrieved[0][1]

        if q["type"] == "unanswerable":
            counts["refusal_total"] += 1
            passed = top_score < SCORE_THRESHOLD
            if passed:
                counts["refusal_pass"] += 1
            print(f"  top_score={top_score:.3f} -> {'PASS' if passed else 'FAIL (would answer anyway)'}")
            results_log.append({"id": q["id"], "type": q["type"], "refusal_pass": passed})
            continue

        system, user = build_prompt(q["question"], retrieved)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system},
                     {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=800,
            reasoning_effort="low",
        )
        answer = resp.choices[0].message.content
        verdict = judge(client, q["question"], q["expected_answer"], answer)
        counts[verdict] = counts.get(verdict, 0) + 1

        print(f"  verdict: {verdict}")
        print(f"  expected: {q['expected_answer'][:80]}")
        print(f"  got:      {answer[:150].replace(chr(10), ' ')}")

        results_log.append({
            "id": q["id"], "type": q["type"], "verdict": verdict,
            "expected": q["expected_answer"], "generated": answer,
            "top_score": top_score,
        })

    scored = counts["CORRECT"] + counts["PARTIAL"] + counts["WRONG"] + counts["UNKNOWN"]
    print("\n========== SUMMARY ==========")
    if scored:
        acc = counts["CORRECT"] / scored
        print(f"Correct: {counts['CORRECT']}/{scored} ({acc:.0%})")
        print(f"Partial: {counts['PARTIAL']}/{scored}")
        print(f"Wrong:   {counts['WRONG']}/{scored}")
    if counts["refusal_total"]:
        print(f"Correct refusals: {counts['refusal_pass']}/{counts['refusal_total']}")
    if counts["skipped"]:
        print(f"Skipped (manual check needed): {counts['skipped']}")

    if args.label:
        Path("eval_results").mkdir(exist_ok=True)
        out = Path("eval_results") / f"answers_{args.label}.json"
        out.write_text(json.dumps({"counts": counts, "details": results_log}, indent=2),
                       encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()