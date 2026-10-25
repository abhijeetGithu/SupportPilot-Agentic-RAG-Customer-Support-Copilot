"""End-to-end agent scenarios with the real LLM, retriever, classifier and database.

    python evals/run_agent_eval.py [--only id1,id2]

Each scenario runs in a fresh in-memory database with its own chat session. Ticket drafts
(interrupts) are answered with the scenario's `resume` action (default: approve).

Reports: pass rate, first-tool accuracy, per-check failures, LLM calls, latency and tokens.
Writes evals/reports/agent_eval.json (with full transcripts for failure analysis).
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from supportpilot.agent.graph import build_graph, new_turn
from supportpilot.config import get_settings
from supportpilot.db import base
from supportpilot.db.models import Ticket, TicketSource
from supportpilot.tickets.schemas import TicketCreate
from supportpilot.tickets.service import TicketService, create_chat_session

SCENARIOS = Path("evals/datasets/agent_scenarios.json")
REPORT = Path("evals/reports/agent_eval.json")
SELF = {"name": "Ana", "email": "ana@example.com"}
CITATION = re.compile(r"\[\d+\]")


def run_scenario(sc: dict, graph) -> dict:
    base.configure("sqlite://")
    base.init_db()
    with base.session_scope() as db:
        for t in sc.get("seed_tickets", []):
            email = SELF["email"] if t["email"] == "SELF" else t["email"]
            TicketService(db).create(
                TicketCreate(customer_name="Seed", customer_email=email, subject=t["subject"], description=t["description"]),
                TicketSource.FORM,
            )
        session = create_chat_session(db, SELF["name"], SELF["email"])
        customer = {"session_id": session.id, **SELF}
    cfg = {"configurable": {"thread_id": customer["session_id"]}}

    tool_calls, turns, n_seen = [], [], 0
    for user_text in sc["turns"]:
        t0 = time.perf_counter()
        out = graph.invoke({**new_turn(user_text), "customer": customer}, cfg)
        interrupts = 0
        while out.get("__interrupt__") and interrupts < 3:
            interrupts += 1
            out = graph.invoke(Command(resume={"action": sc.get("resume", "approve")}), cfg)
        latency = time.perf_counter() - t0
        new = out["messages"][n_seen:]
        n_seen = len(out["messages"])
        llm_msgs = [m for m in new if isinstance(m, AIMessage) and m.usage_metadata]
        for m in new:
            if isinstance(m, AIMessage):
                tool_calls += [c["name"] for c in m.tool_calls]
        final = next((m.content for m in reversed(new) if isinstance(m, AIMessage) and not m.tool_calls), "")
        turns.append({
            "user": user_text,
            "answer": final,
            "tools": [c["name"] for m in new if isinstance(m, AIMessage) for c in m.tool_calls],
            "tool_results": [m.content[:300] for m in new if isinstance(m, ToolMessage)],
            "latency_s": round(latency, 2),
            "llm_calls": len(llm_msgs),
            "tokens_in": sum(m.usage_metadata.get("input_tokens", 0) for m in llm_msgs),
            "tokens_out": sum(m.usage_metadata.get("output_tokens", 0) for m in llm_msgs),
        })

    with base.session_scope() as db:
        tickets = db.query(Ticket).filter(Ticket.session_id == customer["session_id"]).all()
        ticket_emails = [t.customer_email for t in db.query(Ticket).all()]
    final_answer = turns[-1]["answer"]
    exp, checks = sc["expect"], {}
    if "first_tool" in exp:
        checks["first_tool"] = bool(tool_calls) and tool_calls[0] in exp["first_tool"]
    for name in exp.get("must_call", []):
        checks[f"calls_{name}"] = name in tool_calls
    if "must_call_any" in exp:
        checks["calls_any_of"] = any(n in tool_calls for n in exp["must_call_any"])
    for name in exp.get("must_not_call", []):
        checks[f"avoids_{name}"] = name not in tool_calls
    if exp.get("no_tools"):
        checks["no_tools"] = not tool_calls
    if "ticket_created" in exp:
        checks["ticket_created"] = bool(tickets) == exp["ticket_created"]
    if "handed_off" in exp:
        checks["handed_off"] = (out.get("handed_off") or False) == exp["handed_off"]
    if exp.get("answer_has_citation"):
        checks["cites_sources"] = bool(CITATION.search(final_answer))
    for s in exp.get("answer_contains", []):
        checks[f"says_{s}"] = s.lower() in final_answer.lower()
    for s in exp.get("answer_not_contains", []):
        checks[f"hides_{s}"] = s.lower() not in " ".join(t["answer"] for t in turns).lower()
    if "no_ticket_for_email" in exp:
        checks["identity_safe"] = exp["no_ticket_for_email"] not in ticket_emails
    if "min_search_calls" in exp:
        checks["searches_again"] = turns[-1]["tools"].count("search_docs") >= exp["min_search_calls"]

    return {"id": sc["id"], "passed": all(checks.values()), "checks": checks, "tool_calls": tool_calls,
            "tickets_created": len(tickets), "turns": turns}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated scenario ids")
    args = p.parse_args()
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only.split(",")]

    graph = build_graph(InMemorySaver())
    results = []
    for sc in scenarios:
        r = run_scenario(sc, graph)
        results.append(r)
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {sc['id']:24s} tools={r['tool_calls']}"
              + (f"  failed={failed}" if failed else ""))

    turns = [t for r in results for t in r["turns"]]
    first_tool = [r["checks"]["first_tool"] for r in results if "first_tool" in r["checks"]]
    all_checks = [v for r in results for v in r["checks"].values()]
    lat = sorted(t["latency_s"] for t in turns)
    summary = {
        "model": get_settings().llm_model,
        "scenarios": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "checks_passed": f"{sum(all_checks)}/{len(all_checks)}",
        "first_tool_accuracy": round(sum(first_tool) / len(first_tool), 3) if first_tool else None,
        "avg_llm_calls_per_turn": round(statistics.mean(t["llm_calls"] for t in turns), 2),
        "latency_s_p50": round(statistics.median(lat), 2),
        "latency_s_p95": round(lat[max(0, int(0.95 * len(lat)) - 1)], 2),
        "tokens_in_per_turn": round(statistics.mean(t["tokens_in"] for t in turns)),
        "tokens_out_per_turn": round(statistics.mean(t["tokens_out"] for t in turns)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
