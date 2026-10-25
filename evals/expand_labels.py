"""Add near-duplicate pages as relevant labels.

Many SearchUnify connector pages share boilerplate word for word (e.g. "Find and Replace",
crawl-frequency notes). A generic question generated from the Seismic page is answered just
as well by the identical passage on the Zendesk page, so that page should count as relevant.

For each golden question, every page containing a chunk whose word 5-shingle Jaccard
similarity with the source chunk is >= THRESHOLD is added to `relevant_doc_ids`.
Not applied when the question names its source page (e.g. "...a Khoros crawl..."): then the
user needs that specific page, and a copy on another connector page would be a wrong citation.
`expected_doc_ids` (the original single label) is kept for strict scoring.

    python evals/expand_labels.py [--threshold 0.7]
"""

import argparse
import json
import re
from pathlib import Path

GOLDEN = Path("evals/datasets/golden.jsonl")
CHUNKS = Path("data/processed/chunks.jsonl")


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def names_source_page(question: str, doc_id: str) -> bool:
    """True if the question mentions the page's subject, e.g. 'Khoros' for Content-Sources/Khoros.htm."""
    q = question.lower()
    name = Path(doc_id).stem.replace("-", " ").lower()
    first = name.split()[0]
    def has(word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", q) is not None

    return has(name) or (len(first) >= 4 and has(first))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()

    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    sh = {c["chunk_id"]: shingles(c["text"]) for c in chunks}
    doc_of = {c["chunk_id"]: c["doc_id"] for c in chunks}

    rows = [json.loads(line) for line in GOLDEN.open(encoding="utf-8")]
    expanded = 0
    for r in rows:
        if names_source_page(r["question"], r["expected_doc_ids"][0]):
            r["relevant_doc_ids"] = list(r["expected_doc_ids"])
            continue
        src = sh[r["source_chunk_id"]]
        dupes = {doc_of[cid] for cid, s in sh.items() if jaccard(src, s) >= args.threshold}
        relevant = sorted(set(r["expected_doc_ids"]) | dupes)
        r["relevant_doc_ids"] = relevant
        expanded += len(relevant) > len(r["expected_doc_ids"])

    GOLDEN.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    sizes = sorted(len(r["relevant_doc_ids"]) for r in rows)
    print(f"{expanded}/{len(rows)} questions gained extra relevant pages (max {sizes[-1]} pages)")


if __name__ == "__main__":
    main()
