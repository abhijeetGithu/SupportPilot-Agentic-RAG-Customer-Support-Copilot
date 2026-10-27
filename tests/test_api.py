"""API tests: real FastAPI app, real graph and database, scripted LLM, parsed SSE streams."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from supportpilot.agent import tools
from supportpilot.agent.graph import build_graph
from supportpilot.api import deps
from supportpilot.api.main import app
from supportpilot.api.routes import admin as admin_routes
from supportpilot.classifier.predict import TicketPrediction
from supportpilot.config import get_settings
from supportpilot.db.models import ChatSession, Ticket, TicketSource
from supportpilot.retrieval.schemas import RetrievedChunk
from supportpilot.tickets import suggest
from supportpilot.tickets.schemas import TicketCreate
from supportpilot.tickets.service import TicketService
from tests.helpers import fake_classifier, word_overlap_similarity
from tests.test_agent_graph import ScriptedLLM, call

CHUNK = RetrievedChunk(id="Clone.htm#0", doc_id="Search-Clients/Clone.htm", text="Click the Clone icon.", score=1.0,
                       url="https://docs/Clone.htm", title="Clone a Search Client", heading=None)


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setenv("WARMUP_MODELS", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(tools, "ticket_service", lambda s: TicketService(s, fake_classifier, word_overlap_similarity))
    monkeypatch.setattr(tools, "search", lambda q, expansions=None, **kw: [CHUNK])
    app.dependency_overrides[deps.ticket_service] = _fake_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.graph = None
    get_settings.cache_clear()


def _fake_service(db: deps.DB) -> TicketService:
    return TicketService(db, fake_classifier, word_overlap_similarity)


def use_script(*messages):
    app.state.graph = build_graph(InMemorySaver(), llm=ScriptedLLM(responses=list(messages)))


def sse(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def new_session(client, email="ana@example.com") -> str:
    r = client.post("/api/sessions", json={"name": "Ana", "email": email})
    assert r.status_code == 201
    return r.json()["session_id"]


def staff_headers(client) -> dict:
    token = client.post("/api/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_and_unknown_session(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/api/chat/nope/stream", json={"message": "hi"}).status_code == 404


def test_streamed_answer_with_citations(client):
    use_script(call("search_docs", query="clone search client"), AIMessage("Click Clone【1†L1-L2】."))
    sid = new_session(client)
    events = sse(client.post(f"/api/chat/{sid}/stream", json={"message": "how to clone?"}))
    types = [e["type"] for e in events]
    assert types[0] == "tool_start" and events[0] == {"type": "tool_start", "tool": "search_docs", "query": "clone search client"}
    assert "tool_end" in types and types[-1] == "done"
    assert next(e for e in events if e["type"] == "citations")["citations"][0]["url"] == "https://docs/Clone.htm"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Click Clone[1]."


def test_ticket_draft_approve_flow(client):
    use_script(call("create_ticket", subject="Search broken", description="The search page shows no results today."),
               AIMessage("Your ticket is SP-00001."))
    sid = new_session(client)
    events = sse(client.post(f"/api/chat/{sid}/stream", json={"message": "search is broken"}))
    assert events[-2] == {"type": "ticket_draft", "draft": {"subject": "Search broken",
                                                            "description": "The search page shows no results today."}}
    assert client.get(f"/api/chat/{sid}/history").json()["pending_draft"]["subject"] == "Search broken"
    # a new message is refused while the draft is pending
    assert client.post(f"/api/chat/{sid}/stream", json={"message": "hello?"}).status_code == 409

    events = sse(client.post(f"/api/chat/{sid}/resume", json={"action": "edit", "subject": "No search results"}))
    created = next(e for e in events if e["type"] == "ticket_created")
    assert created["number"] == "SP-00001" and created["duplicate"] is False and events[-1]["ticket_number"] == "SP-00001"
    hist = client.get(f"/api/chat/{sid}/history").json()
    assert hist["pending_draft"] is None and hist["ticket_number"] == "SP-00001"
    assert [m["role"] for m in hist["messages"]] == ["customer", "ai"]
    assert client.post(f"/api/chat/{sid}/resume", json={"action": "approve"}).status_code == 409


def test_escalation_handoff_and_staff_reply_reach_the_chat(client, db):
    use_script(call("escalate_to_human", reason="customer asked for a person"))
    sid = new_session(client)
    events = sse(client.post(f"/api/chat/{sid}/stream", json={"message": "I need a human, now please"}))
    assert {"type": "handoff", "number": "SP-00001"} in events
    assert "support team" in "".join(e.get("text", "") for e in events)

    # in human mode the agent is skipped and the message is stored on the ticket
    events = sse(client.post(f"/api/chat/{sid}/stream", json={"message": "are you there?"}))
    assert events[0]["type"] == "stored"

    h = staff_headers(client)
    queue = client.get("/api/admin/escalations", headers=h).json()
    assert queue["total"] == 1
    tid = queue["items"][0]["id"]
    assert client.post(f"/api/admin/escalations/{tid}/take", headers=h).json()["ticket"]["assignee"] == "admin"
    assert client.post(f"/api/admin/escalations/{tid}/take", headers=h).status_code == 409
    client.post(f"/api/admin/tickets/{tid}/replies", json={"body": "Hi Ana, I'm on it."}, headers=h)

    polled = client.get(f"/api/chat/{sid}/messages", params={"after": 0}).json()
    assert polled["mode"] == "human" and [m["content"] for m in polled["messages"]] == ["Hi Ana, I'm on it."]
    last_id = polled["messages"][-1]["id"]
    assert client.get(f"/api/chat/{sid}/messages", params={"after": last_id}).json()["messages"] == []

    client.patch(f"/api/admin/tickets/{tid}", json={"status": "resolved"}, headers=h)
    assert client.get(f"/api/chat/{sid}/messages").json()["mode"] == "ai"
    db.expire_all()
    assert db.get(ChatSession, sid).mode == "ai"


def test_contact_form_lookup_and_customer_reply(client):
    body = {"customer_name": "Ana", "customer_email": "ana@example.com", "subject": "Invoice wrong",
            "description": "We were charged twice on the last invoice."}
    r = client.post("/api/tickets", json=body)
    assert r.status_code == 201 and r.json()["ticket"]["category"] == "Billing and Payments"
    number = r.json()["ticket"]["number"]

    assert client.post("/api/tickets/lookup", json={"number": number, "email": "mallory@x.com"}).status_code == 404
    view = client.post("/api/tickets/lookup", json={"number": number, "email": "ANA@example.com"}).json()
    assert view["status"] == "open" and "confidence" not in view

    h = staff_headers(client)
    tid = r.json()["ticket"]["id"]
    client.patch(f"/api/admin/tickets/{tid}", json={"status": "resolved"}, headers=h)
    view = client.post(f"/api/tickets/{number}/replies", json={"email": "ana@example.com", "body": "Still wrong!"}).json()
    assert view["status"] == "in_progress" and view["messages"][-1]["body"] == "Still wrong!"


def test_admin_auth_list_update_and_metrics(client, monkeypatch):
    assert client.get("/api/admin/tickets").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "x"}).status_code == 401
    h = staff_headers(client)
    for subj, desc in [("Invoice wrong", "We were charged twice this month."),
                       ("Weird behaviour", "The thing does not work the way I expected it to.")]:
        client.post("/api/tickets", json={"customer_name": "Ana", "customer_email": "ana@example.com",
                                          "subject": subj, "description": desc})
    page = client.get("/api/admin/tickets", headers=h).json()
    assert page["total"] == 2 and page["items"][0]["priority"] == "high"  # higher priority first
    assert client.get("/api/admin/tickets", params={"needs_review": True}, headers=h).json()["total"] == 1
    assert client.get("/api/admin/tickets", params={"q": "invoice"}, headers=h).json()["total"] == 1

    tid = client.get("/api/admin/tickets", params={"needs_review": True}, headers=h).json()["items"][0]["id"]
    detail = client.patch(f"/api/admin/tickets/{tid}", json={"category": "IT Support"}, headers=h).json()
    assert detail["ticket"]["needs_review"] is False
    assert detail["events"][-1]["type"] == "reclassified" and detail["events"][-1]["actor"] == "staff:admin"
    assert client.get("/api/admin/tickets/999", headers=h).status_code == 404

    m = client.get("/api/admin/metrics", headers=h).json()
    assert m["tickets_by_priority"] == {"high": 1, "medium": 1} and m["needs_review"] == 0

    monkeypatch.setattr(admin_routes, "classify_ticket", lambda text: TicketPrediction(
        category="IT Support", priority="low", confidence=0.7, priority_confidence=0.6, model="fake"))
    assert client.post("/api/admin/classify", json={"text": "vpn down"}, headers=h).json()["category"] == "IT Support"


def test_suggest_reply_is_grounded_and_cited(db):
    ticket = TicketService(db, fake_classifier, word_overlap_similarity).create(
        TicketCreate(customer_name="Ana", customer_email="ana@example.com", subject="How to clone?",
                     description="I need a copy of our search client for testing."),
        TicketSource.FORM,
    ).ticket
    t = db.get(Ticket, ticket.id)
    llm = FakeListChatModel(responses=["Hi! Use the Clone icon on the Search Clients page【1†L1-L2】."])
    out = suggest.suggest_reply(t, llm=llm, searcher=lambda q, expansions=None: [CHUNK, CHUNK])
    reply, sources = out["body"].split("\n\n**Sources**\n")
    assert reply.endswith("[1].")
    assert sources == f"1. [{CHUNK.title}]({CHUNK.url})"  # the customer gets a link for every [n]
    assert [c["n"] for c in out["citations"]] == [1]  # only sources actually cited are returned
