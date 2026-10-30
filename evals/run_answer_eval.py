"""Answer quality with and without the self-check (RAGAS-style metrics, one judge call per answer).

For each question the real agent answers (gpt-oss-120b, hybrid retrieval); the self-check (gpt-oss-20b)
may revise the draft. Both the DRAFT (= no self-check) and the FINAL answer are scored by an independent
judge (Qwen 3.8 27B, a different model family from both):
  - faithfulness: share of the answer's factual claims supported by the retrieved passages (RAGAS definition)
  - correctness vs. the golden reference answer: correct / partial / incorrect / not_answered
  - for questions the docs don't cover: did the answer admit it, or assert something?
Deterministic: did the answer cite a page from the golden expected docs?
Judge check: a fabricated sentence is planted in answers the judge scored fully faithful; it must notice.

Why not the ragas package: each RAGAS faithfulness sample takes several LLM calls; on Groq's free tier
(8K tokens/min, 200K/day per model) that is too slow for 30 questions × 2 answers. Same definitions, one
structured judge call per answer.

Progress is saved after every question (evals/cache/answer_eval.jsonl), so a run that hits the daily
token limit resumes where it stopped.      python evals/run_answer_eval.py [--limit N]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from supportpilot.agent.graph import build_graph, new_turn
from supportpilot.agent.grounding import select_passages
from supportpilot.api.streaming import stream_events
from supportpilot.config import get_settings
from supportpilot.db import base

CACHE = ROOT / "evals" / "cache" / "answer_eval.jsonl"
REPORT = ROOT / "evals" / "reports" / "answer_eval.json"
JUDGE_MODEL = "qwen/qwen3.8-27b"

# Questions about SearchUnify that the public docs don't answer (non-existent or oddly specific features).
# Their closest pages are on related topics (dense cosine 0.67-0.77, all above the 0.66 doc-gap threshold),
# so retrieval returns plausible-looking passages: a harder test than off-topic questions.
UNANSWERABLE = [
    "Does SearchUnify support crawling Microsoft Teams voice messages and transcribing them?",
    "Is there a native SearchUnify app for Apple Watch?",
    "Can a search client show results in Klingon?",
    "What is SearchUnify's uptime SLA credit percentage for the Starter plan?",
    "Can SearchUnify crawl content stored on floppy disks attached to our on-prem server?",
    "How many SearchUnify employees work in the Tokyo office?",
    "Does SearchUnify integrate with the Nintendo Switch eShop?",
    "What is the exact price per seat for SearchUnify in Brazilian reais?",
    "Can I schedule a crawl to run only during full moons?",
    "Does SearchUnify support indexing handwritten notes from a reMarkable tablet?",
]

JUDGE_PROMPT = """You grade a customer-support answer. Be strict and literal.

Question: {question}
Reference answer (may be empty): {reference}

Documentation passages the assistant retrieved:
{passages}

Answer to grade:
{answer}

1. List the answer's factual claims about the product (skip greetings, ticket offers, questions back).
   For each, is it supported by the passages? Paraphrase is fine; details absent from the passages are unsupported.
2. correctness against the reference answer: "correct" (covers its key point), "partial", "incorrect"
   (contradicts it or gives wrong information), or "not_answered" (says it doesn't know / not in the docs).
   If the reference is empty, use "not_answered" when the answer declines and "incorrect" when it asserts an answer.
3. admits_unknown: true if the answer says the documentation doesn't cover the question or that it couldn't find it.

Return JSON: {{"claims": [{{"claim": "...", "supported": true}}], "correctness": "...", "admits_unknown": false}}"""

FABRICATED = " Note that this requires the Enterprise Plus licence and works for at most 50 users."


def judge_llm():
    return ChatGroq(model=JUDGE_MODEL, api_key=get_settings().groq_api_key, temperature=0, max_tokens=1500,
                    max_retries=6, model_kwargs={"response_format": {"type": "json_object"}})


def judge(llm, question: str, reference: str, passages: str, answer: str) -> dict:
    out = llm.invoke(JUDGE_PROMPT.format(question=question, reference=reference or "", passages=select_passages(passages, answer),
                                         answer=answer))
    data = json.loads(re.sub(r"^```(json)?|```$", "", str(out.content).strip()))
    claims = data.get("claims") or []
    return {"claims": len(claims), "supported": sum(bool(c.get("supported")) for c in claims),
            "correctness": data.get("correctness"), "admits_unknown": bool(data.get("admits_unknown"))}


def questions(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for name, n in (("golden.jsonl", 10), ("golden_hard.jsonl", 10)):
        rows = [json.loads(line) for line in (ROOT / "evals" / "datasets" / name).read_text(encoding="utf-8").splitlines()]
        for r in rng.sample(rows, n):
            out.append({"id": f"{name.split('.')[0]}:{r['id']}", "question": r["question"],
                        "reference": r.get("reference_answer", ""), "expected": r.get("expected_doc_ids", []),
                        "answerable": True})
    out += [{"id": f"unanswerable:{i}", "question": q, "reference": "", "expected": [], "answerable": False}
            for i, q in enumerate(UNANSWERABLE)]
    return out


def run_agent(graph, q: dict) -> dict:
    thread = f"eval-{q['id']}"
    cfg = {"configurable": {"thread_id": thread}}
    customer = {"session_id": thread, "name": "Eval", "email": "eval@example.com"}
    t0 = time.perf_counter()
    events = list(stream_events(graph, {**new_turn(q["question"]), "customer": customer}, cfg))
    latency = time.perf_counter() - t0
    state = graph.get_state(cfg).values
    draft = "".join(e["text"] for e in events if e["type"] == "token")
    g = next((e for e in events if e["type"] == "grounding"), {"verdict": "none"})
    final = g.get("revised_text") or draft
    passages = "\n\n".join(str(m.content) for m in state["messages"] if isinstance(m, ToolMessage))
    used = {int(n) for n in re.findall(r"\[(\d+)\]", final)}
    cited_urls = [c["url"] for c in state.get("citations") or [] if c["n"] in used]
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in state["messages"]
                 if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None))
    errors = [e["message"] for e in events if e["type"] == "error"]
    return {"draft": draft, "final": final, "grounding": g, "passages": passages, "cited_urls": cited_urls,
            "latency_s": round(latency, 1), "agent_tokens": tokens, "errors": errors,
            "offered_ticket": any(e["type"] == "ticket_draft" for e in events)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    base.configure("sqlite://")
    base.init_db()
    graph = build_graph(InMemorySaver())
    jl = judge_llm()
    done = {}
    if CACHE.exists():
        done = {r["id"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    qs = questions()[: args.limit] if args.limit else questions()
    for q in qs:
        if q["id"] in done:
            continue
        r = {**q, **run_agent(graph, q)}
        if r["errors"]:
            print(f"{q['id']}: agent error {r['errors'][0][:120]} (stopping; rerun later to resume)")
            break
        r["judge_draft"] = judge(jl, q["question"], q["reference"], r["passages"], r["draft"])
        r["judge_final"] = (r["judge_draft"] if r["final"] == r["draft"]
                            else judge(jl, q["question"], q["reference"], r["passages"], r["final"]))
        r["cited_expected"] = any(u.endswith(d) for u in r["cited_urls"] for d in q["expected"]) if q["expected"] else None
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done[q["id"]] = r
        print(f"{q['id']:<28} verdict={r['grounding'].get('verdict'):<9} draft={r['judge_draft']['correctness']:<12} "
              f"final={r['judge_final']['correctness']:<12} faith {r['judge_draft']['supported']}/{r['judge_draft']['claims']}"
              f" -> {r['judge_final']['supported']}/{r['judge_final']['claims']}  {r['latency_s']}s")
    summarize([done[q["id"]] for q in qs if q["id"] in done], jl)


def faith(rows, key):
    c = sum(r[key]["claims"] for r in rows)
    return round(sum(r[key]["supported"] for r in rows) / c, 3) if c else None


def summarize(rows, jl):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]

    def dist(key):
        out = {}
        for r in ans:
            out[r[key]["correctness"]] = out.get(r[key]["correctness"], 0) + 1
        return out

    # judge sensitivity: plant a fabricated claim in answers the judge rated fully faithful
    plant = [r for r in ans if r["judge_final"]["claims"] and r["judge_final"]["supported"] == r["judge_final"]["claims"]][:8]
    caught = 0
    for r in plant:
        j = judge(jl, r["question"], r["reference"], r["passages"], r["final"] + FABRICATED)
        caught += j["supported"] < j["claims"]
    summary = {
        "questions": len(rows), "answerable": len(ans), "unanswerable": len(una),
        "self_check_verdicts": {v: sum(r["grounding"].get("verdict") == v for r in rows)
                                for v in ("grounded", "revised", "no_claims", "unchecked", "none")},
        "faithfulness_draft": faith(rows, "judge_draft"), "faithfulness_final": faith(rows, "judge_final"),
        "answerable_correctness_draft": dist("judge_draft"), "answerable_correctness_final": dist("judge_final"),
        "answerable_cited_expected_doc": f"{sum(bool(r['cited_expected']) for r in ans)}/{len(ans)}",
        "unanswerable_admitted_draft": f"{sum(r['judge_draft']['admits_unknown'] for r in una)}/{len(una)}",
        "unanswerable_admitted_final": f"{sum(r['judge_final']['admits_unknown'] for r in una)}/{len(una)}",
        "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2] if rows else None,
        "judge_planted_claims": len(plant), "judge_caught": caught,
    }
    REPORT.write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
