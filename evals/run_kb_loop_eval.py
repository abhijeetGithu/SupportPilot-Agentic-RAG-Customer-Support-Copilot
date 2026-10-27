"""Does the knowledge loop make the assistant answer questions it couldn't before?

For each synthetic support case whose fix is NOT in the public docs:
  1. create the ticket + the support team's resolution, resolve it, draft a KB article with the real LLM
  2. BEFORE publishing: production retrieval and the real agent answer a paraphrased customer question
  3. publish the article (indexed into the live Qdrant collection like the docs)
  4. AFTER: same questions again
Scores: article grounded (contains the fix), anonymised, account-specific case flagged; KB retrieved in the
top 5; agent answer contains the fix and cites the KB.

Articles are numbered KB-90001+ and removed from the index at the end (also on failure).
Uses about 60K Groq tokens: run sparingly.   python evals/run_kb_loop_eval.py
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from supportpilot.agent.graph import build_graph, new_turn
from supportpilot.classifier.predict import TicketPrediction
from supportpilot.db import base
from supportpilot.db.models import AuthorType, KbArticle, KbStatus, TicketSource
from supportpilot.knowledge import publish
from supportpilot.knowledge.drafting import draft_article
from supportpilot.retrieval.retriever import search
from supportpilot.tickets.schemas import TicketCreate
from supportpilot.tickets.service import TicketService, create_chat_session

REPORT = ROOT / "evals" / "reports" / "kb_loop_eval.json"
FIRST_ID = 90001
PAUSE_S = 8

# facts: every group must match (any alternative in a group), case-insensitive
CASES = [
    {"id": "crawl_queued", "name": "Ravi Kumar", "email": "ravi@globex.io",
     "subject": "Zendesk crawl stuck in Queued",
     "description": "Hi, Ravi from Globex here. Our Zendesk Help Center content source has shown Queued for 7 hours "
                    "and never starts crawling.",
     "resolution": "Thanks Ravi. This happens when a crawl is scheduled while the previous crawl of the same source "
                   "is still being cancelled. Open Content Sources, select the source, click Stop crawl, wait until "
                   "the status shows Idle, then click Start crawl. If it is still Queued after 30 minutes, reply "
                   "with the content source ID.",
     "question": "Our Freshdesk content source has said Queued since yesterday and never crawls. What should I do?",
     "facts": [["stop crawl", "stop the crawl", "click stop"], ["idle"], ["start crawl", "start the crawl", "click start"]],
     "generalizable": True},
    {"id": "facet_zero_after_rename", "name": "Mei Lin", "email": "mei@initech.com",
     "subject": "Product facet shows 0 results after Salesforce field rename",
     "description": "We renamed the Product_Line__c field in Salesforce to Product_Family__c and now the Product "
                    "facet in our search client shows 0 for every value.",
     "resolution": "Hi Mei, facets keep pointing at the old API name after a rename. Go to Content Sources > "
                   "Salesforce > Fields, map Product_Family__c to the Product facet field again, save, and run a "
                   "full crawl (an incremental crawl won't refresh existing records).",
     "question": "We renamed a field in Salesforce and now our facet counts are all zero. How do we fix the facet?",
     "facts": [["map"], ["full crawl", "full re-crawl", "full recrawl"]],
     "generalizable": True},
    {"id": "okta_sso_loop", "name": "Jonas Weber", "email": "jonas@umbrella.de",
     "subject": "Okta SSO keeps redirecting back to login",
     "description": "After setting up Okta SAML for the SearchUnify admin console, admins are sent back to the "
                    "login page in a loop.",
     "resolution": "Hi Jonas, the loop had two causes: the ACS URL in Okta had a trailing slash (it must end exactly "
                   "with /saml/consume) and your IdP server clock was 7 minutes off. Fix the ACS URL in the Okta app "
                   "settings and sync the server time with NTP; SAML assertions are rejected beyond 5 minutes of "
                   "clock skew.",
     "question": "Admins get stuck in a redirect loop when logging in with Okta SAML. Any idea why?",
     "facts": [["/saml/consume"], ["clock", "time sync", "ntp"]],
     "generalizable": True},
    {"id": "export_timeout", "name": "Priya Nair", "email": "priya@stark.in",
     "subject": "90-day analytics export times out",
     "description": "Exporting the Searches report for the last 90 days spins and then fails with a timeout.",
     "resolution": "Hi Priya, exports of large date ranges can exceed the request time limit. Export in ranges of "
                   "30 days or less, or use Schedule Report to have the full 90-day report emailed to you as a CSV.",
     "question": "When I export a quarter of search analytics the download times out. How can I get the data?",
     "facts": [["30 days", "30-day"], ["schedule"]],
     "generalizable": True},
    {"id": "search_client_cors", "name": "Tom Hughes", "email": "tom@wayne.co.uk",
     "subject": "Embedded search client blank, CORS error",
     "description": "Our embedded search client on help.wayne.co.uk shows an empty results panel and the browser "
                    "console says blocked by CORS policy.",
     "resolution": "Hi Tom, the site's domain wasn't allowed. In Search Clients > your client > Security, add "
                   "https://help.wayne.co.uk under Allowed origins (include https:// and no trailing slash), save, "
                   "and hard-refresh the page.",
     "question": "Our embedded search shows no results and the console has a CORS blocked error. What setting fixes it?",
     "facts": [["allowed origin"], ["https://"]],
     "generalizable": True},
    {"id": "refund_one_off", "name": "Sara Costa", "email": "sara@acme.pt",
     "subject": "Charged twice for annual renewal",
     "description": "Our card was charged twice for the annual renewal this month.",
     "resolution": "Hi Sara, I've confirmed the duplicate charge on your account and issued a refund of the second "
                   "payment. It will appear on your card within 5-7 business days.",
     "question": None, "facts": [], "generalizable": False},
]


# gpt-oss writes narrow no-break spaces and non-breaking hyphens, e.g. "30\u202fdays", "90\u2011day"
UNICODE_FIX = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-"})


def normalize(text: str) -> str:
    return text.translate(UNICODE_FIX).lower()


def has_facts(text: str, facts: list[list[str]]) -> bool:
    t = normalize(text)
    return all(any(normalize(alt) in t for alt in group) for group in facts)


def leaks_in(text: str, case: dict) -> list[str]:
    """Whole-word matches of the reporting customer's name, email or company in the article."""
    company = case["email"].split("@")[1].split(".")[0]
    terms = [*case["name"].lower().split(), case["email"], company]
    return [x for x in terms if re.search(rf"\b{re.escape(x)}\b", normalize(text))]


def kb_rank(question: str, number: str) -> int | None:
    for i, h in enumerate(search(question), 1):
        if h.doc_id == f"kb/{number}":
            return i
    return None


def ask(graph, db_customer: dict, question: str) -> dict:
    cfg = {"configurable": {"thread_id": db_customer["session_id"]}}
    t0 = time.perf_counter()
    out = graph.invoke({**new_turn(question), "customer": db_customer}, cfg)
    offered_ticket = bool(out.get("__interrupt__"))
    if offered_ticket:
        out = graph.invoke(Command(resume={"action": "cancel"}), cfg)
    answer = "\n".join(str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content and not m.tool_calls)
    kb_cited = any(f"[{c['n']}]" in answer and "/support/kb/KB-9" in c["url"] for c in out.get("citations") or [])
    return {"answer": answer, "offered_ticket": offered_ticket, "kb_cited": kb_cited,
            "latency_s": round(time.perf_counter() - t0, 1)}


def classify_stub(text: str) -> TicketPrediction:
    return TicketPrediction(category="Technical Support", priority="medium", confidence=0.9,
                            priority_confidence=0.9, model="stub")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts-only", action="store_true", help="only stage 1 (about 15K tokens)")
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp())
    base.configure(f"sqlite:///{(tmp / 'kb_eval.db').as_posix()}")
    base.init_db()
    graph = build_graph(InMemorySaver())
    articles: dict[str, KbArticle] = {}
    results = []
    try:
        # 1. tickets -> resolution -> drafts (real LLM)
        with base.session_scope() as db:
            db.add(KbArticle(id=FIRST_ID - 1, title="placeholder", body="placeholder", status=KbStatus.REJECTED.value))
            db.flush()
            svc = TicketService(db, classifier=classify_stub)
            for c in CASES:
                t = svc.create(TicketCreate(customer_name=c["name"], customer_email=c["email"], subject=c["subject"],
                                            description=c["description"]), TicketSource.FORM).ticket
                svc.add_message(t.id, AuthorType.STAFF, c["resolution"])
                ticket = svc.get(t.id)
                a = draft_article(db, ticket)
                articles[c["id"]] = a
                leaks = leaks_in(f"{a.title}\n{a.body}\n{' '.join(a.keywords)}", c)
                results.append({"id": c["id"], "number": a.number, "title": a.title, "body": a.body,
                                "keywords": a.keywords, "note": a.draft_note,
                                "grounded": has_facts(a.body, c["facts"]) if c["facts"] else None,
                                "leaks": leaks, "generalizable": a.generalizable,
                                "generalizable_ok": a.generalizable == c["generalizable"],
                                "doc_coverage": a.doc_coverage})
                print(f"drafted {a.number} {c['id']}: grounded={results[-1]['grounded']} leaks={leaks} "
                      f"generalizable={a.generalizable}")
                time.sleep(PAUSE_S)

        if args.drafts_only:
            summary = {"articles_grounded": f"{sum(bool(r['grounded']) for r in results if r['grounded'] is not None)}"
                                            f"/{sum(r['grounded'] is not None for r in results)}",
                       "articles_without_customer_identifiers": f"{sum(not r['leaks'] for r in results)}/{len(results)}",
                       "reusability_flag_correct": f"{sum(r['generalizable_ok'] for r in results)}/{len(results)}"}
            (REPORT.parent / "kb_loop_drafts.json").write_text(json.dumps({"summary": summary, "results": results},
                                                                          indent=2, default=str))
            print(json.dumps(summary, indent=2))
            return

        # 2. before publishing
        for c, r in zip(CASES, results):
            if not c["question"]:
                continue
            with base.session_scope() as db:
                s = create_chat_session(db, "Eval", "eval@example.com")
                customer = {"session_id": s.id, "name": "Eval", "email": "eval@example.com"}
            r["before"] = {"kb_rank": kb_rank(c["question"], r["number"]), **ask(graph, customer, c["question"])}
            r["before"]["has_fix"] = has_facts(r["before"]["answer"], c["facts"])
            print(f"before {c['id']}: has_fix={r['before']['has_fix']} offered_ticket={r['before']['offered_ticket']}")
            time.sleep(PAUSE_S)

        # 3. publish the reusable ones (what a reviewer would do)
        for c in CASES:
            if c["generalizable"]:
                publish.index_article(articles[c["id"]])

        # 4. after publishing
        for c, r in zip(CASES, results):
            if not c["question"]:
                continue
            with base.session_scope() as db:
                s = create_chat_session(db, "Eval", "eval@example.com")
                customer = {"session_id": s.id, "name": "Eval", "email": "eval@example.com"}
            r["after"] = {"kb_rank": kb_rank(c["question"], r["number"]), **ask(graph, customer, c["question"])}
            r["after"]["has_fix"] = has_facts(r["after"]["answer"], c["facts"])
            print(f"after  {c['id']}: kb_rank={r['after']['kb_rank']} has_fix={r['after']['has_fix']} "
                  f"kb_cited={r['after']['kb_cited']}")
            time.sleep(PAUSE_S)
    finally:
        for a in articles.values():
            publish.remove_article(a)
        print("eval articles removed from the index")

    asked = [r for r in results if "after" in r]
    summary = {
        "cases": len(CASES),
        "articles_grounded": f"{sum(bool(r['grounded']) for r in asked)}/{len(asked)}",
        "articles_without_customer_identifiers": f"{sum(not r['leaks'] for r in results)}/{len(results)}",
        "reusability_flag_correct": f"{sum(r['generalizable_ok'] for r in results)}/{len(results)}",
        "kb_in_top5_after": f"{sum((r['after']['kb_rank'] or 99) <= 5 for r in asked)}/{len(asked)}",
        "answer_has_fix_before": f"{sum(r['before']['has_fix'] for r in asked)}/{len(asked)}",
        "answer_has_fix_after": f"{sum(r['after']['has_fix'] for r in asked)}/{len(asked)}",
        "kb_cited_after": f"{sum(r['after']['kb_cited'] for r in asked)}/{len(asked)}",
        "offered_ticket_before": f"{sum(r['before']['offered_ticket'] for r in asked)}/{len(asked)}",
        "offered_ticket_after": f"{sum(r['after']['offered_ticket'] for r in asked)}/{len(asked)}",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
