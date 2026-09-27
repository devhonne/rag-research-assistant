"""
Step 2 of the Research Paper Assistant: chunking.

Reads  extracted/*.json  (from extract_pdfs.py)
Writes chunks/chunks.jsonl  (one chunk per line)

Each chunk has: chunk_id, paper, page_start, page_end, n_words, text

Run:
    python chunk_papers.py
    python chunk_papers.py --words 350 --overlap 50   # try other sizes
"""
import argparse
import json
import re
from pathlib import Path

IN_DIR = Path("extracted")
OUT_DIR = Path("chunks")

REF_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.I)


def load_words(pages):
    """Flatten a paper into a list of (word, page_number).
    Stops at the References heading so citations don't pollute retrieval."""
    words = []
    total_pages = len(pages)
    for p in pages:
        for line in p["text"].split("\n"):
            # Only treat it as the References section in the second half of the paper
            if REF_HEADING.match(line) and p["page"] > total_pages * 0.4:
                return words, True
            for w in line.split():
                words.append((w, p["page"]))
    return words, False


def chunk_words(words, size, overlap):
    """Sliding window over words. Tries to end each chunk at a sentence end."""
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + size, n)

        # Snap the end to a sentence boundary within the last 20% of the window
        if end < n:
            window_start = start + int(size * 0.8)
            for i in range(end - 1, window_start, -1):
                if words[i][0].endswith((".", "?", "!")):
                    end = i + 1
                    break

        piece = words[start:end]
        chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=450,
                    help="target words per chunk (about 600 tokens)")
    ap.add_argument("--overlap", type=int, default=60,
                    help="words shared between neighbouring chunks")
    ap.add_argument("--min-words", type=int, default=40,
                    help="drop tiny leftover chunks")
    args = ap.parse_args()

    files = sorted(IN_DIR.glob("*.json"))
    if not files:
        print("No files in ./extracted. Run extract_pdfs.py first.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "chunks.jsonl"
    total = 0
    sizes = []

    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            pages = json.loads(f.read_text(encoding="utf-8"))
            words, cut_refs = load_words(pages)
            chunks = chunk_words(words, args.words, args.overlap)

            kept = 0
            for i, piece in enumerate(chunks):
                if len(piece) < args.min_words:
                    continue
                record = {
                    "chunk_id": f"{f.stem}_{i:04d}",
                    "paper": f.stem,
                    "page_start": piece[0][1],
                    "page_end": piece[-1][1],
                    "n_words": len(piece),
                    "text": " ".join(w for w, _ in piece),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                sizes.append(len(piece))

            total += kept
            note = "References cut" if cut_refs else "no References heading found"
            print(f"{f.stem[:50]:50s} {kept:4d} chunks  ({note})")

    avg = sum(sizes) / max(len(sizes), 1)
    print(f"\nTotal chunks: {total} | average words per chunk: {avg:.0f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
