"""Create a harder, real-user-style version of the golden set.

LLM-written questions reuse the docs' wording, which flatters retrieval. Real users type short,
vague, misspelled queries in their own vocabulary. Each golden question is rewritten in one of
four styles (round-robin, so the mix is balanced). Labels are kept, so the standard and hard
sets are directly comparable.

Two lessons from v1, where rewrites kept ~68% word overlap with the source passage:
- LLMs ignore "avoid the docs' wording" unless told WHICH words: each item now gets an explicit
  list of doc terms it must not use (proper nouns like Jira are allowed, users name products).
- LLMs rarely add typos when asked: typos are injected in code (seeded, reproducible).

    python evals/generate_hard_set.py   ->  evals/datasets/golden_hard.jsonl
"""

import json
import random
import re
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from supportpilot.config import get_settings

GOLDEN = Path("evals/datasets/golden.jsonl")
CHUNKS = Path("data/processed/chunks.jsonl")
OUT = Path("evals/datasets/golden_hard.jsonl")
BATCH = 15
SEED = 13

STYLES = {
    "keywords": "2-5 words like a search-box query, no full sentence",
    "symptom": "describe what the user sees or wants in plain words, as someone who doesn't know the feature's name",
    "casual_typo": "short, lowercase, casual chat tone (typos are added later, don't add them)",
    "synonyms": "a full question using everyday words instead of the docs' terminology",
}

# fmt: off
STOPWORDS = {
    "what", "when", "where", "which", "does", "with", "from", "that", "this", "have", "your",
    "there", "their", "about", "into", "will", "would", "should", "could", "able", "they",
    "them", "then", "than", "also", "only", "just", "like", "want", "need",
}
# fmt: on

SYSTEM = (
    """You rewrite support questions about SearchUnify (an enterprise search platform) into the
way real users type them. Keep the SAME information need so the same documentation page still
answers it. Do not answer the question or add new details.

Each item has a style and a list "avoid": documentation terms the user would NOT know.
Never use any word from "avoid" or an obvious variant of it (plural, -ing, etc.). Rephrase with
everyday words instead. Product and connector names that are not in "avoid" (Jira, Salesforce,
Zendesk...) may stay, since users name the tools they use.

Styles:
"""
    + "\n".join(f"- {k}: {v}" for k, v in STYLES.items())
    + """

Return JSON: {"items": [{"id": "...", "query": "..."}]}"""
)


def product_names(chunks: dict[str, dict]) -> set[str]:
    """Lowercase words of connector/product names, from 'Use X As a Content Source' titles."""
    names = {"searchunify"}
    for c in chunks.values():
        m = re.match(r"Use (.+?) [Aa]s an? Content Source", c["title"])
        if m:
            names |= {w.lower() for w in re.findall(r"[A-Za-z]{3,}", m.group(1))}
    return names


def avoid_terms(question: str, passage: str, allowed: set[str]) -> list[str]:
    """Words from the question that also appear in the passage, except product names and stopwords."""
    text = passage.lower()
    out: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", question):
        lw = w.lower()
        if lw in STOPWORDS or lw in allowed or lw in out:
            continue
        if re.search(r"\b" + re.escape(lw[:5]), text):  # stem-ish match
            out.append(lw)
    return out


def add_typos(query: str, rng: random.Random, n: int = 2) -> str:
    """Swap two adjacent letters or drop one, in up to n words of 5+ letters."""
    words = query.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    for i in rng.sample(candidates, min(n, len(candidates))):
        w = words[i]
        p = rng.randrange(1, len(w) - 2)
        words[i] = (
            w[:p] + w[p + 1] + w[p] + w[p + 2 :] if rng.random() < 0.6 else w[:p] + w[p + 1 :]
        )
    return " ".join(words)


def _retryable(e: BaseException) -> bool:
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503)


@retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(min=8, max=90),
    stop=stop_after_attempt(8),
)
def rewrite(client: httpx.Client, items: list[dict]) -> dict[str, str]:
    s = get_settings()
    body = "\n".join(
        json.dumps(
            {"id": r["id"], "style": r["style"], "question": r["question"], "avoid": r["avoid"]}
        )
        for r in items
    )
    resp = client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {s.groq_api_key}"},
        json={
            "model": s.llm_model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
            "response_format": {"type": "json_object"},
            "temperature": 0.8,
            "reasoning_effort": "medium",
            "max_completion_tokens": 4000,
        },
    )
    resp.raise_for_status()
    items_out = json.loads(resp.json()["choices"][0]["message"]["content"]).get("items", [])
    return {i["id"]: i["query"].strip() for i in items_out if i.get("query")}


def main() -> None:
    rng = random.Random(SEED)
    chunks = {c["chunk_id"]: c for c in map(json.loads, CHUNKS.open(encoding="utf-8"))}
    rows = [json.loads(line) for line in GOLDEN.open(encoding="utf-8")]
    allowed = product_names(chunks)
    styles = list(STYLES)
    for i, r in enumerate(rows):
        r["style"] = styles[i % len(styles)]
        r["avoid"] = avoid_terms(r["question"], chunks[r["source_chunk_id"]]["text"], allowed)

    out = []
    with httpx.Client(timeout=180) as client:
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            rewritten = rewrite(client, batch)
            for r in batch:
                if r["id"] not in rewritten:
                    continue
                q = rewritten[r["id"]]
                if r["style"] == "casual_typo":
                    q = add_typos(q.lower(), rng)
                out.append({**r, "original_question": r["question"], "question": q})
            print(f"{min(i + BATCH, len(rows))}/{len(rows)} rewritten")

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"wrote {len(out)} hard queries to {OUT}")


if __name__ == "__main__":
    main()
