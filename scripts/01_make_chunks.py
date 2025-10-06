#!/usr/bin/env python3
import json, re, uuid
from pathlib import Path

IN_DIR = Path("./data/papers_json/")
OUT_DIR = Path("./data/parsed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_HEADERS = [
    r"^\s*abstract\s*:?$",
    r"^\s*introduction\s*:?$",
    r"^\s*background\s*:?$",
    r"^\s*methods?\s*:?$",
    r"^\s*materials\s+and\s+methods\s*:?$",
    r"^\s*results?\s*:?$",
    r"^\s*discussion\s*:?$",
    r"^\s*conclusions?\s*:?$",
    r"^\s*limitations\s*:?$",
    r"^\s*future\s+work\s*:?$",
    r"^\s*references\s*:?$",
]
SEC_RE = re.compile("|".join(SECTION_HEADERS), flags=re.I)

def normalize_whitespace(t: str) -> str:
    t = re.sub(r"\r", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    # keep paragraphs
    return t.strip()

def guess_sections(all_text: str):
    # Split into paragraphs first
    paras = [p.strip() for p in re.split(r"\n{2,}", all_text) if p.strip()]
    current = "Body"
    for p in paras:
        # If a paragraph is just a header-like line, update section
        if SEC_RE.match(p.lower()):
            current = re.sub(r":\s*$", "", p, flags=re.I).title()
            continue
        yield current, p

def robust_chunks(text: str, max_chars=1200):
    """If no blank-line paragraphs, fall back to sliding-window by sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    buf = []
    size = 0
    for s in sents:
        if size + len(s) + 1 > max_chars and buf:
            yield "Body", " ".join(buf).strip()
            buf, size = [s], len(s) + 1
        else:
            buf.append(s); size += len(s) + 1
    if buf:
        yield "Body", " ".join(buf).strip()

def process_file(pth: Path):
    with pth.open(encoding="utf-8") as f:
        doc = json.load(f)
    paper_id = pth.stem  # e.g., PMC1234567
    text = normalize_whitespace(doc.get("All_text",""))
    if not text:
        print(f"SKIP (empty): {pth.name}")
        return
    # Prefer paragraph/section heuristic, else robust chunks
    chunks = list(guess_sections(text))
    if not chunks:
        chunks = list(robust_chunks(text))

    out_path = OUT_DIR / f"{paper_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as w:
        for i, (section, para) in enumerate(chunks):
            rec = {
                "chunk_id": f"{paper_id}:{i}",
                "paper_id": paper_id,
                "section": section,
                "para_idx": i,
                "text": para
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ {pth.name} → {out_path} ({len(chunks)} chunks)")

def main():
    for p in sorted(IN_DIR.glob("*.json")):
        print(p)
        process_file(p)

if __name__ == "__main__":
    main()