"""Generate candidate golden Q&A pairs from indexed chunks, for human review.

    python evals/generate_golden.py [--n 125] [--batch 5] [--seed 7]

- Stratified sample of chunks across doc sections (release notes capped, tiny chunks skipped)
- LLM writes ONE realistic support-chat question per passage, paraphrased (not copied),
  plus a short reference answer and a question type; it may skip unusable passages
- Label = the source page (page-level relevance) and source chunk

Writes evals/datasets/golden_candidates.jsonl and .csv. Review the CSV (keep / fix / drop),
then run  python evals/finalize_golden.py  to produce golden.jsonl.
"""

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from supportpilot.config import get_settings

CHUNKS = Path("data/processed/chunks.jsonl")
OUT = Path("evals/datasets/golden_candidates.jsonl")
RELEASE_NOTES_CAP = 5
MIN_TOKENS = 80

SYSTEM = """You create evaluation questions for a customer-support assistant of SearchUnify,
an enterprise search and support platform (content sources/connectors, search clients,
analytics, search tuning, AI agents).

For EACH passage, write ONE question that a real SearchUnify admin or user would type into a
support chat, and that the passage alone fully answers.

Rules:
- Sound like a real user with a task or problem ("How do I...", "Why isn't...", "Can I...").
- PARAPHRASE. Do not copy the passage's headings or distinctive phrases word for word.
- Name the product area only as a user naturally would (e.g. "Jira connector", "search client").
- Never say "the passage", "this document", "according to".
- Vary type: how_to, troubleshooting, conceptual, factual.
- answer: 1-3 sentences, using only facts from the passage.
- If a passage is only navigation, a list of links, or too vague to support a real question,
  set "skip": true.

Return JSON: {"items": [{"passage_id": "...", "skip": false, "question": "...",
"answer": "...", "type": "how_to|troubleshooting|conceptual|factual"}]}"""


def load_chunks() -> list[dict]:
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stratified_sample(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Allocate by sqrt(section size) so small sections still get questions; max 1 chunk per page first."""
    rng = random.Random(seed)
    by_section: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["tokens"] >= MIN_TOKENS:
            by_section[c["section"]].append(c)
    weights = {s: math.sqrt(len(v)) for s, v in by_section.items()}
    total = sum(weights.values())
    picked: list[dict] = []
    for section, items in by_section.items():
        quota = max(1, round(n * weights[section] / total))
        if section == "Release-Notes":
            quota = min(quota, RELEASE_NOTES_CAP)
        rng.shuffle(items)
        seen_pages: set[str] = set()
        first = [c for c in items if not (c["doc_id"] in seen_pages or seen_pages.add(c["doc_id"]))]
        rest = [c for c in items if c not in first]
        picked.extend((first + rest)[:quota])
    rng.shuffle(picked)
    return picked[:n]


def _retryable(e: BaseException) -> bool:
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503)


@retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=8, max=90), stop=stop_after_attempt(8))
def call_llm(client: httpx.Client, passages: list[dict]) -> list[dict]:
    s = get_settings()
    body = "\n\n".join(
        f'<passage id="{p["chunk_id"]}">\nPage: {p["title"]}\n{p["text"][:2200]}\n</passage>' for p in passages
    )
    resp = client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {s.groq_api_key}"},
        json={
            "model": s.llm_model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "reasoning_effort": "low",
            "max_completion_tokens": 2000,
        },
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"]).get("items", [])


def word_overlap(question: str, text: str) -> float:
    """Share of question content words that appear in the passage (lexical-shortcut check)."""
    words = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
    if not words:
        return 0.0
    passage = set(re.findall(r"[a-z]{4,}", text.lower()))
    return round(len(words & passage) / len(words), 2)


def write_review_csv(f, rows: list[dict]) -> None:
    cols = ["id", "keep", "flag", "question", "reference_answer", "type", "section", "source_url", "lexical_overlap"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        flag = "copies passage wording - consider rephrasing" if r["lexical_overlap"] >= 0.9 else ""
        w.writerow({**r, "keep": "y", "flag": flag})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=125)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sample = stratified_sample(load_chunks(), args.n, args.seed)
    by_id = {c["chunk_id"]: c for c in sample}
    rows: list[dict] = []
    used: set[str] = set()  # the LLM sometimes returns several questions for one passage
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(sample), args.batch):
            batch = sample[i : i + args.batch]
            for item in call_llm(client, batch):
                c = by_id.get(item.get("passage_id"))
                if not c or c["chunk_id"] in used or item.get("skip") or not item.get("question"):
                    continue
                used.add(c["chunk_id"])
                rows.append(
                    {
                        "id": f"q{len(rows) + 1:03d}",
                        "question": item["question"].strip(),
                        "reference_answer": item.get("answer", "").strip(),
                        "type": item.get("type", "how_to"),
                        "section": c["section"],
                        "expected_doc_ids": [c["doc_id"]],
                        "source_chunk_id": c["chunk_id"],
                        "source_url": c["url"],
                        "lexical_overlap": word_overlap(item["question"], c["text"]),
                        "generated_by": get_settings().llm_model,
                        "reviewed": False,
                    }
                )
            print(f"{min(i + args.batch, len(sample))}/{len(sample)} passages -> {len(rows)} questions")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        write_review_csv(f, rows)
    print(f"wrote {len(rows)} candidates to {OUT} (+ .csv for review)")


if __name__ == "__main__":
    main()
