"""
Step 1 of the Research Paper Assistant:
Extract text page by page from every PDF in ./papers and check quality.

Setup:
    pip install pymupdf pdfplumber

Run:
    python extract_pdfs.py                # uses PyMuPDF (fast)
    python extract_pdfs.py --lib plumber  # uses pdfplumber (compare results)

Output:
    extracted/<paper_name>.json   -> list of {"page": n, "text": "..."}
    extracted/<paper_name>.txt    -> readable version to eyeball
    A quality report printed to the terminal.
"""
import argparse
import json
import re
from pathlib import Path

PAPERS_DIR = Path("papers")
OUT_DIR = Path("extracted")


def extract_pymupdf(path):
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            # "blocks" keeps reading order better for two-column papers
            blocks = page.get_text("blocks", sort=True)
            text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
            pages.append({"page": i, "text": text})
    return pages


def extract_plumber(path):
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({"page": i, "text": page.extract_text() or ""})
    return pages


def clean(text):
    text = text.replace("\u00ad", "")            # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)     # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_report(name, pages):
    """Flag common extraction problems."""
    total_chars = sum(len(p["text"]) for p in pages)
    empty = [p["page"] for p in pages if len(p["text"]) < 200]
    all_text = " ".join(p["text"] for p in pages)
    weird = len(re.findall(r"[\ufffd\u0000-\u0008]", all_text))
    weird_ratio = weird / max(len(all_text), 1)
    has_refs = bool(re.search(r"\n\s*references\s*\n", all_text, re.I))

    print(f"\n{name}")
    print(f"  pages: {len(pages)} | characters: {total_chars}")
    if empty:
        print(f"  WARNING near-empty pages (scanned or figure only?): {empty}")
    if weird_ratio > 0.001:
        print(f"  WARNING garbled characters: {weird} ({weird_ratio:.2%})")
    print(f"  'References' section detected: {has_refs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=["pymupdf", "plumber"], default="pymupdf")
    args = parser.parse_args()

    extractor = extract_pymupdf if args.lib == "pymupdf" else extract_plumber
    OUT_DIR.mkdir(exist_ok=True)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in ./{PAPERS_DIR}. Put your 20 papers there.")
        return

    print(f"Extracting {len(pdfs)} PDFs with {args.lib}...")
    for pdf in pdfs:
        try:
            pages = extractor(pdf)
        except Exception as e:
            print(f"\n{pdf.name}\n  ERROR: {e}")
            continue

        for p in pages:
            p["text"] = clean(p["text"])

        (OUT_DIR / f"{pdf.stem}.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        readable = "\n\n".join(f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages)
        (OUT_DIR / f"{pdf.stem}.txt").write_text(readable, encoding="utf-8")
        quality_report(pdf.name, pages)

    print(f"\nDone. Open the .txt files in ./{OUT_DIR} and read a few pages.")


if __name__ == "__main__":
    main()
