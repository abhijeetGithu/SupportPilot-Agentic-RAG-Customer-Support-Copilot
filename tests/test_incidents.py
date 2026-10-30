"""Incident detection and the staff incident console. Embeddings are faked with topic vectors, so the
tests check the rules (distinct customers, problem reports, burst vs. baseline, joining, notices)."""

from datetime import timedelta

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sqlalchemy import select

from supportpilot.db.models import Incident, Ticket, TicketSource
from supportpilot.incidents import detector
from supportpilot.incidents.detector import looks_like_problem
from supportpilot.tickets.schemas import TicketCreate
from supportpilot.tickets.service import TicketService, _utcnow
from tests.helpers import fake_classifier, word_overlap_similarity
from tests.test_api import client, staff_headers  # noqa: F401  (fixture)

TOPICS = {"salesforce": 0, "analytics": 1, "jira": 2}


def fake_embed(texts):
    """One axis per topic keyword, plus a little per-text noise so vectors aren't identical."""
    out = []
    for t in texts:
        v = np.zeros(8)
        v[next((i for k, i in TOPICS.items() if k in t.lower()), 7)] = 1.0
        v[3 + len(t) % 4] = 0.15
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


# captured at import, before conftest's per-test patches turn detection off
REAL_ON_TICKET_CREATED, REAL_MATCH_ACTIVE = detector.on_ticket_created, detector.match_active


@pytest.fixture
def detection(monkeypatch):
    """Turn real detection back on, with fake embeddings."""
    monkeypatch.setattr(detector, "on_ticket_created", REAL_ON_TICKET_CREATED)
    monkeypatch.setattr(detector, "match_active", REAL_MATCH_ACTIVE)
    monkeypatch.setattr(detector, "embed", fake_embed)
    return detector


def report(svc, email, text, subject=None):
    return svc.create(TicketCreate(customer_name="C", customer_email=email, subject=subject or text[:60],
                                   description=text), TicketSource.FORM).ticket


def service(db):
    return TicketService(db, fake_classifier, word_overlap_similarity)


SF = ["Salesforce crawl failing with 401 since this morning", "Our Salesforce source shows an error, crawl failed",
      "Salesforce indexing stopped, unauthorized error", "Salesforce crawl broken for us too, 401 errors"]


def test_problem_gate():
    assert looks_like_problem("Salesforce crawl failing with 401") and looks_like_problem("Search is very slow today")
    assert looks_like_problem("Scheduled crawls did not run last night")
    assert not looks_like_problem("How do I add Salesforce as a content source?")


def test_burst_from_different_customers_opens_an_incident(db, detection):
    svc = service(db)
    a = report(svc, "a@one.com", SF[0])
    b = report(svc, "b@two.com", SF[1])
    assert db.scalars(select(Incident)).all() == []  # two reports aren't enough
    c = report(svc, "c@three.com", SF[2])
    [inc] = db.scalars(select(Incident)).all()
    assert {t.id for t in inc.tickets} == {a.id, b.id, c.id}
    assert inc.detection["customers"] == 3 and inc.detection["p_value"] < 0.01
    assert all(any(e.type == "incident_opened" for e in db.get(Ticket, t.id).events) for t in (a, b, c))


def test_one_customer_repeating_is_not_an_incident(db, detection):
    svc = service(db)
    for i, text in enumerate(SF[:3]):
        report(svc, "same@one.com", f"{text} (attempt {i})")
    assert db.scalars(select(Incident)).all() == []


def test_how_to_questions_are_not_incidents(db, detection):
    svc = service(db)
    for i, text in enumerate(["How do I add Salesforce?", "Which Salesforce objects can be indexed?",
                              "Can Salesforce attachments be crawled?"]):
        report(svc, f"u{i}@x{i}.com", text)
    assert db.scalars(select(Incident)).all() == []


def test_a_usual_rate_of_similar_reports_is_not_a_burst(db, detection):
    svc = service(db)
    # Salesforce errors are normal for this team: 40 similar reports over the last week
    old = _utcnow() - timedelta(days=3)
    for i in range(40):
        db.add(Ticket(customer_name="C", customer_email=f"old{i}@x.com", subject="Salesforce error",
                      description="Salesforce crawl failed", category="Technical Support", priority="medium",
                      status="resolved", source="form", created_at=old + timedelta(hours=i),
                      embedding=fake_embed(["Salesforce crawl failed"])[0].tolist()))
    db.flush()
    for i, text in enumerate(SF[:3]):
        report(svc, f"u{i}@x{i}.com", text)
    assert db.scalars(select(Incident)).all() == []  # 3 in two hours vs ~1.4 expected: not unusual enough


def test_later_reports_join_and_get_the_public_note(db, detection):
    svc = service(db)
    for i, text in enumerate(SF[:3]):
        report(svc, f"u{i}@x{i}.com", text)
    [inc] = db.scalars(select(Incident)).all()
    inc.public_note = "Salesforce changed its login API; we are rolling out a fix, ETA 2 hours."
    db.flush()
    late = report(svc, "late@four.com", SF[3])
    other = report(svc, "z@five.com", "Analytics dashboard is empty, no data today")
    assert db.get(Ticket, late.id).incident_id == inc.id and db.get(Ticket, other.id).incident_id is None
    [msg] = db.get(Ticket, late.id).messages
    assert msg.author_type == "staff" and "ETA 2 hours" in msg.body and inc.number in msg.body


def test_detection_failure_never_blocks_ticket_creation(db, detection, monkeypatch):
    monkeypatch.setattr(detection, "embed", lambda texts: 1 / 0)
    t = report(service(db), "a@one.com", SF[0])
    assert t.number.startswith("SP-") and t.incident_id is None


# ---------- staff console ----------
@pytest.fixture
def console(client, detection):  # noqa: F811
    h = staff_headers(client)
    for i, text in enumerate(SF[:3]):
        client.post("/api/tickets", json={"customer_name": f"C{i}", "customer_email": f"u{i}@x{i}.com",
                                          "subject": text[:60], "description": text})
    [inc] = client.get("/api/admin/incidents", headers=h).json()
    return client, h, inc


def test_console_lists_and_updates_everyone(console):
    c, h, inc = console
    assert inc["status"] == "active" and inc["tickets"] == 3 and inc["customers"] == 3 and inc["open_tickets"] == 3
    detail = c.get(f"/api/admin/incidents/{inc['id']}", headers=h).json()
    tid = detail["tickets"][0]["id"]
    assert c.get(f"/api/admin/tickets/{tid}", headers=h).json()["incident"]["number"] == inc["number"]

    c.patch(f"/api/admin/incidents/{inc['id']}", json={"public_note": "We're on it."}, headers=h)
    assert c.post(f"/api/admin/incidents/{inc['id']}/updates", json={"body": "Fix is rolling out now."},
                  headers=h).json() == {"sent_to": 3}
    t = c.get(f"/api/admin/tickets/{tid}", headers=h).json()
    assert t["messages"][-1]["body"] == "Fix is rolling out now." and t["ticket"]["status"] == "in_progress"

    # the customer sees the update on their ticket page
    view = c.post("/api/tickets/lookup", json={"number": t["ticket"]["number"], "email": t["ticket"]["customer_email"]})
    assert view.json()["messages"][-1]["body"] == "Fix is rolling out now."


def test_remove_wrong_ticket_and_resolve(console):
    c, h, inc = console
    detail = c.get(f"/api/admin/incidents/{inc['id']}", headers=h).json()
    wrong = detail["tickets"][0]["id"]
    after = c.delete(f"/api/admin/incidents/{inc['id']}/tickets/{wrong}", headers=h).json()
    assert after["incident"]["tickets"] == 2
    assert c.get(f"/api/admin/tickets/{wrong}", headers=h).json()["incident"] is None

    done = c.post(f"/api/admin/incidents/{inc['id']}/resolve", json={"message": "Fixed, crawls are running again."},
                  headers=h).json()
    assert done["incident"]["status"] == "resolved" and done["incident"]["resolved_by"] == "admin"
    assert {t["status"] for t in done["tickets"]} == {"resolved"}
    assert c.get(f"/api/admin/tickets/{wrong}", headers=h).json()["ticket"]["status"] == "open"  # removed: untouched
    assert c.post(f"/api/admin/incidents/{inc['id']}/updates", json={"body": "late update"},
                  headers=h).status_code == 409


def test_ai_summary_for_staff(console, monkeypatch):
    from supportpilot.incidents import service as incidents

    c, h, inc = console
    llm = FakeListChatModel(responses=['{"title": "Salesforce crawls failing with 401", "summary": "3 customers."}'])
    monkeypatch.setattr(incidents, "get_llm", lambda **kw: llm)
    out = c.post(f"/api/admin/incidents/{inc['id']}/summarize", headers=h).json()["incident"]
    assert out["title"] == "Salesforce crawls failing with 401" and out["summary"] == "3 customers."


def test_assistant_gets_the_staff_notice(db, detection, monkeypatch):
    from supportpilot.agent import tools

    svc = service(db)
    for i, text in enumerate(SF[:3]):
        report(svc, f"u{i}@x{i}.com", text)
    [inc] = db.scalars(select(Incident)).all()
    db.commit()
    generic = "Our team is aware of reports matching this problem and is investigating."
    assert tools.known_issue("Our Salesforce crawl is failing with 401") == {"number": inc.number, "notice": generic}
    inc.public_note = "Fix rolling out, ETA 1 hour."
    db.commit()
    assert tools.known_issue("Salesforce crawl failed again")["notice"] == "Fix rolling out, ETA 1 hour."
    assert tools.known_issue("How do I add Salesforce?") is None  # questions, not problem reports
    assert tools.known_issue("Analytics dashboard is empty") is None  # a different problem


def test_known_issue_reaches_the_chat_as_its_own_event(monkeypatch):
    """The staff notice is streamed as a separate event, so the UI shows it verbatim."""
    from langchain_core.messages import AIMessage

    from supportpilot.agent import tools
    from supportpilot.api.streaming import stream_events
    from tests.test_agent_graph import call

    issue = {"number": "INC-00001", "notice": "Fix rolling out, next update by 3 PM."}
    monkeypatch.setattr(tools, "known_issue", lambda text: issue)
    monkeypatch.setattr(tools, "search", lambda q, expansions=None: [])
    from langgraph.checkpoint.memory import InMemorySaver

    from supportpilot.agent.graph import build_graph, new_turn
    from tests.test_agent_graph import ScriptedLLM

    llm = ScriptedLLM(responses=[call("search_docs", query="salesforce 401"), AIMessage("We're aware and on it.")])
    graph = build_graph(InMemorySaver(), llm=llm)
    customer = {"session_id": "s1", "name": "Ana", "email": "ana@example.com"}
    events = list(stream_events(graph, {**new_turn("Salesforce crawl failing with 401"), "customer": customer},
                                {"configurable": {"thread_id": "s1"}}))
    assert {"type": "known_issue", **issue} in events


def test_poisson_tail_matches_scipy():
    from scipy.stats import poisson

    from supportpilot.incidents.detector import poisson_tail

    for k, rate in [(1, 0.05), (3, 0.05), (3, 1.4), (5, 2.0), (7, 0.3)]:
        assert abs(poisson_tail(k, rate) - poisson.sf(k - 1, rate)) < 1e-12
