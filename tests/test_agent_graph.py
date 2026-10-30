"""Agent graph tests with a scripted LLM: real graph, tools, checkpointer and database; no API calls."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from supportpilot.agent import tools
from supportpilot.agent.graph import build_graph, new_turn
from supportpilot.db.models import ChatSession, SessionMode, Ticket
from supportpilot.retrieval.schemas import RetrievedChunk
from supportpilot.tickets.service import TicketService, create_chat_session
from tests.helpers import fake_classifier, word_overlap_similarity


class ScriptedLLM(FakeMessagesListChatModel):
    """Returns the scripted messages in order; bind_tools is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, i: int = 1, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}"}])


@pytest.fixture
def env(db, monkeypatch):
    monkeypatch.setattr(tools, "ticket_service", lambda s: TicketService(s, fake_classifier, word_overlap_similarity))
    searched = []

    def fake_search(query, expansions=None, **kw):
        searched.append((query, expansions))
        return [RetrievedChunk(id="Jira.htm#1", doc_id="Content-Sources/Jira.htm", text="Use an API token.",
                               score=1.0, url="https://docs/Jira.htm", title="Use Jira", heading="Use Jira > Auth")]

    monkeypatch.setattr(tools, "search", fake_search)
    session = create_chat_session(db, "Ana", "ana@example.com")
    db.commit()
    customer = {"session_id": session.id, "name": "Ana", "email": "ana@example.com"}
    return {"db": db, "customer": customer, "searched": searched}


def run(script, env, message, thread="t1", graph=None):
    graph = graph or build_graph(InMemorySaver(), llm=ScriptedLLM(responses=script))
    cfg = {"configurable": {"thread_id": thread}}
    out = graph.invoke({**new_turn(message), "customer": env["customer"]}, cfg)
    return graph, cfg, out


def test_answers_from_docs_with_citations(env):
    script = [call("search_docs", query="Jira content source authentication"), AIMessage("Use an API token [1].")]
    _, _, out = run(script, env, "how do i connect jira?")
    assert out["messages"][-1].content == "Use an API token [1]."
    assert out["citations"] == [{"n": 1, "title": "Use Jira", "url": "https://docs/Jira.htm", "heading": "Use Jira > Auth"}]
    # the LLM's doc-vocabulary query is fused with the customer's own wording
    assert env["searched"] == [("Jira content source authentication", ["how do i connect jira?"])]


def test_create_ticket_pauses_for_confirmation_then_creates(env):
    script = [call("create_ticket", subject="Charged twice", description="Customer was charged twice for one invoice."),
              AIMessage("Done! Your ticket is SP-00001.")]
    graph, cfg, out = run(script, env, "I was charged twice this month")
    draft = out["__interrupt__"][0].value
    assert draft == {"type": "ticket_draft", "draft": {"subject": "Charged twice",
                                                       "description": "Customer was charged twice for one invoice."}}
    assert env["db"].query(Ticket).count() == 0  # nothing created before the customer confirms

    out = graph.invoke(Command(resume={"action": "approve"}), cfg)
    ticket = env["db"].query(Ticket).one()
    assert ticket.customer_email == "ana@example.com" and ticket.source == "chat"
    assert ticket.session_id == env["customer"]["session_id"] and ticket.category == "Billing and Payments"
    assert out["ticket_number"] == "SP-00001" and out["messages"][-1].content.startswith("Done!")
    assert [m.author_type for m in ticket.messages] == ["customer"]  # transcript saved


def test_customer_can_edit_or_cancel_the_draft(env):
    script = [call("create_ticket", subject="Bug", description="Search results are empty for every query."),
              AIMessage("Created.")]
    graph, cfg, _ = run(script, env, "search is broken")
    graph.invoke(Command(resume={"action": "edit", "subject": "Search returns no results"}), cfg)
    assert env["db"].query(Ticket).one().subject == "Search returns no results"

    script = [call("create_ticket", subject="Bug", description="Something else that is broken."), AIMessage("Okay.")]
    graph, cfg, _ = run(script, env, "another issue", thread="t2")
    out = graph.invoke(Command(resume={"action": "cancel"}), cfg)
    assert env["db"].query(Ticket).count() == 1
    tool_msg = [m for m in out["messages"] if isinstance(m, ToolMessage)][-1]
    assert "cancelled" in tool_msg.content and not out.get("ticket_number")


def test_ticket_status_is_scoped_to_the_customer(env):
    svc = TicketService(env["db"], fake_classifier, word_overlap_similarity)
    from supportpilot.db.models import TicketSource
    from supportpilot.tickets.schemas import TicketCreate

    mine = svc.create(TicketCreate(customer_name="Ana", customer_email="ana@example.com", subject="Invoice",
                                   description="Invoice total is wrong this month."), TicketSource.FORM).ticket
    other = svc.create(TicketCreate(customer_name="Bob", customer_email="bob@example.com", subject="Outage",
                                    description="Complete outage on our side today."), TicketSource.FORM).ticket
    env["db"].commit()

    script = [call("get_ticket_status", ticket_number=mine.number), AIMessage("It's open."),
              call("get_ticket_status", i=2, ticket_number=other.number), AIMessage("Not found.")]
    graph, cfg, out = run(script, env, f"status of {mine.number}?")
    assert "status open" in [m for m in out["messages"] if isinstance(m, ToolMessage)][-1].content
    out = graph.invoke(new_turn(f"and {other.number}?"), cfg)
    assert "No ticket" in [m for m in out["messages"] if isinstance(m, ToolMessage)][-1].content


def test_escalation_hands_off_to_human(env):
    script = [call("escalate_to_human", reason="customer requests a refund")]
    _, _, out = run(script, env, "I want a refund now and to talk to a person!!")
    assert out["handed_off"] and out["ticket_number"] == "SP-00001"
    assert "support team" in out["messages"][-1].content
    ticket = env["db"].query(Ticket).one()
    env["db"].refresh(ticket)
    assert ticket.status == "escalated"
    session = env["db"].get(ChatSession, env["customer"]["session_id"])
    env["db"].refresh(session)
    assert session.mode == SessionMode.HUMAN.value


def test_step_limit_stops_tool_loops(env, monkeypatch):
    monkeypatch.setattr("supportpilot.agent.graph.get_settings", lambda: type("S", (), {"max_agent_steps": 2, "self_check": False, "output_guard": True})())
    script = [call("search_docs", i=i, query=f"q{i}") for i in range(5)]
    _, _, out = run(script, env, "loop forever")
    assert out["steps"] == 2 and "create a support ticket" in out["messages"][-1].content


def test_memory_across_turns(env):
    script = [AIMessage("Hi Ana!"), AIMessage("You asked about Jira earlier.")]
    graph, cfg, _ = run(script, env, "hello, I have a Jira question")
    out = graph.invoke(new_turn("what did I ask?"), cfg)
    assert [m.content for m in out["messages"]] == [
        "hello, I have a Jira question", "Hi Ana!", "what did I ask?", "You asked about Jira earlier."]


def test_native_gpt_oss_citations_are_normalized(env):
    from supportpilot.agent.graph import normalize_citations

    assert normalize_citations("Click Add【2†L1-L3】 then Save【3†L14-L19】.") == "Click Add[2] then Save[3]."
    assert normalize_citations("Only one source【1】.") == "Only one source[1]."
    script = [call("search_docs", query="jira"), AIMessage("Use a token【1†L2-L4】.")]
    _, _, out = run(script, env, "jira?")
    assert out["messages"][-1].content == "Use a token[1]."
