"""Apply your review of golden_candidates.csv and write golden.jsonl.

In the CSV: set keep=n to drop a row; edit question/reference_answer text to fix it.

    python evals/finalize_golden.py
    python evals/expand_labels.py      # re-add near-duplicate relevant pages
"""

import csv
import json
from pathlib import Path

DATASETS = Path("evals/datasets")


def main() -> None:
    candidates = {}
    with (DATASETS / "golden_candidates.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            candidates[row["id"]] = row
    kept = []
    with (DATASETS / "golden_candidates.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["keep"].strip().lower() not in ("y", "yes", "1"):
                continue
            row = candidates[r["id"]]
            edited = r["question"] != row["question"] or r["reference_answer"] != row["reference_answer"]
            row.update(question=r["question"].strip(), reference_answer=r["reference_answer"].strip())
            row.update(reviewed=True, edited=edited)
            kept.append(row)
    with (DATASETS / "golden.jsonl").open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"golden.jsonl: {len(kept)} reviewed questions ({sum(r['edited'] for r in kept)} edited)")


if __name__ == "__main__":
    main()
