"""MCP server through the SDK's in-process client: tool listing, schemas, results and errors."""

from datetime import datetime

import anyio
import pytest
from mcp import Client

from supportpilot.db.models import AuthorType, Ticket, TicketSource
from supportpilot.mcp_server import server
from supportpilot.retrieval.schemas import RetrievedChunk
from supportpilot.tickets.service import TicketService
from tests.helpers import fake_classifier, word_overlap_similarity


@pytest.fixture
def mcp_env(db, monkeypatch):
    monkeypatch.setattr(server, "ticket_service", lambda s: TicketService(s, fake_classifier, word_overlap_similarity))
    searched = []

    def fake_search(query, top_n=None, **kw):
        searched.append((query, top_n))
        return [RetrievedChunk(id=f"Jira.htm#{i}", doc_id="Jira.htm", text=f"Use an API token ({i}).", score=0.9 - i / 10,
                               url="https://docs.searchunify.com/Content/Content-Sources/Jira.htm",
                               title="Jira", heading="Authentication") for i in range(top_n)]

    monkeypatch.setattr(server, "search", fake_search)
    return {"db": db, "searched": searched}


def call(name: str, args: dict | None = None, op: str = "call_tool"):
    async def go():
        async with Client(server.mcp) as client:
            if op == "list_tools":
                return await client.list_tools()
            if op == "get_prompt":
                return await client.get_prompt(name, args)
            return await client.call_tool(name, args or {})

    return anyio.run(go)


def test_lists_three_tools_with_annotations_and_output_schemas(mcp_env):
    tools = {t.name: t for t in call("", op="list_tools").tools}
    assert set(tools) == {"search_docs", "create_ticket", "get_ticket_status"}
    assert tools["search_docs"].annotations.read_only_hint is True
    assert tools["get_ticket_status"].annotations.read_only_hint is True
    assert tools["create_ticket"].annotations.read_only_hint is False
    assert "passages" in tools["search_docs"].output_schema["properties"]
    assert tools["search_docs"].input_schema["properties"]["top_k"]["maximum"] == 10


def test_search_docs_returns_numbered_passages(mcp_env):
    r = call("search_docs", {"query": "jira auth", "top_k": 3})
    assert not r.is_error
    passages = r.structured_content["passages"]
    assert [p["n"] for p in passages] == [1, 2, 3]
    assert passages[0]["url"].endswith("Jira.htm") and passages[0]["heading"] == "Authentication"
    assert mcp_env["searched"] == [("jira auth", 3)]


def test_search_docs_rejects_out_of_range_top_k(mcp_env):
    r = call("search_docs", {"query": "jira", "top_k": 50})
    assert r.is_error and mcp_env["searched"] == []


def test_create_ticket_goes_through_ticket_service(mcp_env):
    report = {"customer_name": "Ana Sharma", "customer_email": "Ana@Example.com", "subject": "Charged twice",
              "description": "I was charged twice on my invoice, card 4111 1111 1111 1111."}
    r = call("create_ticket", report)
    assert not r.is_error
    out = r.structured_content
    assert out["number"].startswith("SP-") and out["category"] == "Billing and Payments" and out["priority"] == "high"
    assert out["duplicate_of_existing"] is False

    t = mcp_env["db"].query(Ticket).one()
    assert t.source == TicketSource.MCP and t.customer_email == "ana@example.com"
    assert "4111 1111 1111 1111" not in t.description  # PII masked like every other channel
    assert [(e.type, e.actor) for e in t.events][:2] == [("created", "system"), ("classified", "ai")]

    again = call("create_ticket", report)  # the same report sent twice is merged, not duplicated
    assert again.structured_content["duplicate_of_existing"] is True
    assert again.structured_content["number"] == out["number"]


def test_create_ticket_reports_validation_errors(mcp_env):
    r = call("create_ticket", {"customer_name": "Ana", "customer_email": "not-an-email",
                               "subject": "x", "description": "short"})
    assert r.is_error
    text = r.content[0].text
    assert "customer_email" in text and "subject" in text and "description" in text
    assert mcp_env["db"].query(Ticket).count() == 0


def test_get_ticket_status_requires_the_customers_email(mcp_env):
    created = call("create_ticket", {"customer_name": "Ana", "customer_email": "ana@example.com",
                                     "subject": "Outage on search", "description": "Search outage since this morning."})
    number = created.structured_content["number"]
    t = mcp_env["db"].query(Ticket).one()
    TicketService(mcp_env["db"]).add_message(t.id, AuthorType.STAFF, "We are on it, ETA 1 hour.")
    mcp_env["db"].commit()

    ok = call("get_ticket_status", {"ticket_number": number, "customer_email": "ANA@example.com"})
    assert not ok.is_error
    assert ok.structured_content["status"] == "open" and ok.structured_content["priority"] == "urgent"
    assert ok.structured_content["latest_support_reply"] == "We are on it, ETA 1 hour."
    # RFC 3339 needs an offset; strict clients (Claude) reject results that don't match the output schema
    for field in ("created_at", "updated_at"):
        assert datetime.fromisoformat(ok.structured_content[field]).utcoffset() is not None

    other = call("get_ticket_status", {"ticket_number": number, "customer_email": "eve@example.com"})
    missing = call("get_ticket_status", {"ticket_number": "SP-99999", "customer_email": "ana@example.com"})
    assert other.is_error and missing.is_error
    assert "Outage" not in other.content[0].text  # nothing leaks to the wrong email


def test_prompt_mentions_the_tools(mcp_env):
    r = call("answer_support_question", {"question": "How do I add Jira?"}, op="get_prompt")
    text = r.messages[0].content.text
    assert "How do I add Jira?" in text and "search_docs" in text and "create_ticket" in text
