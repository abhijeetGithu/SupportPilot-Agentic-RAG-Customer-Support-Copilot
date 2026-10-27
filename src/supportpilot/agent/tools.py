"""Tools the agent can call. search_docs / create_ticket / get_ticket_status are also exposed via MCP.

Customer identity comes from graph state (InjectedState), so the LLM can't choose whose tickets
to create or read. Tools that change state return a Command with a ToolMessage plus state updates.
"""

import logging
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from supportpilot.db.base import session_scope
from supportpilot.db.models import AuthorType, TicketSource
from supportpilot.retrieval.retriever import search
from supportpilot.tickets.schemas import TicketCreate
from supportpilot.tickets.service import TicketService

PASSAGE_CHARS = 1200
log = logging.getLogger(__name__)


def ticket_service(db) -> TicketService:
    """Factory, so tests can swap in a TicketService with a fake classifier."""
    return TicketService(db)


def _typed(m: HumanMessage) -> str:
    """The customer's own words (the LLM additionally sees an attachment block)."""
    return str(m.additional_kwargs.get("display_text", m.content))


def _last_user_text(state: dict) -> str | None:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return _typed(m)
    return None


def _transcript(state: dict) -> list[tuple[AuthorType, str]]:
    out = []
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            files = m.additional_kwargs.get("attachments") or []
            suffix = ("\n[Attached: " + ", ".join(f["filename"] for f in files) + "]") if files else ""
            out.append((AuthorType.CUSTOMER, _typed(m) + suffix))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            out.append((AuthorType.AI, str(m.content)))
    return out


def _reply(tool_call_id: str, text: str, artifact: dict | None = None, **updates) -> Command:
    """ToolMessage for the LLM (text) plus structured data for the API (artifact), and state updates."""
    msg = ToolMessage(text, tool_call_id=tool_call_id, artifact=artifact)
    return Command(update={"messages": [msg], **updates})


@tool
def search_docs(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the SearchUnify documentation. Returns numbered passages; cite them as [n] in your answer.
    Write the query in SearchUnify's own terms."""
    # The LLM's query is already in doc vocabulary (glossary in the system prompt); fusing it with the
    # customer's own wording reproduces the Phase 2 "original + rewritten" search without an extra LLM call.
    user_text = _last_user_text(state)
    expansions = [user_text] if user_text and user_text.strip().lower() != query.strip().lower() else None
    hits = search(query, expansions=expansions)
    citations = list(state.get("citations") or [])
    blocks = []
    for h in hits:
        n = len(citations) + 1
        citations.append({"n": n, "title": h.title or h.doc_id, "url": h.url, "heading": h.heading})
        where = f"{h.title} > {h.heading}" if h.heading and h.heading != h.title else h.title
        if h.doc_type == "kb":
            where = f"Support KB article: {where}"
        blocks.append(f"[{n}] {where}\n{h.text[:PASSAGE_CHARS]}")
    text = "\n\n".join(blocks) if blocks else "No relevant documentation found."
    # A matching outage: the exact staff notice goes to the chat UI as its own card (the artifact), so its
    # times and promises can't be paraphrased; the LLM only gets told it exists.
    issue = known_issue(user_text or query)
    if issue:
        text = f"KNOWN ISSUE ({issue['number']}, being worked on): {issue['notice']}\n\n{text}"
    return _reply(tool_call_id, text, artifact={"known_issue": issue} if issue else None, citations=citations)


def known_issue(text: str) -> dict | None:
    """{"number", "notice"} if the customer describes a problem matching an active incident."""
    from supportpilot.incidents import detector
    from supportpilot.incidents.service import customer_notice

    if not detector.looks_like_problem(text):
        return None
    try:
        with session_scope() as db:
            incident = detector.match_active(db, text)
            if incident is None:
                return None
            return {"number": incident.number, "notice": customer_notice(incident)}
    except Exception:  # never let incident lookup break doc search
        log.warning("known-issue lookup failed", exc_info=True)
        return None


@tool
def create_ticket(
    subject: str,
    description: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Create a support ticket for the current customer. The customer reviews the draft before it is
    submitted. Use a short subject, and write the description in the customer's own voice (first person:
    "I tried...") with what they tried and any error messages, since they read and approve it."""
    # Pauses the graph; the API shows an editable draft and resumes with the customer's decision.
    # On resume this tool re-runs from the top and interrupt() returns that decision.
    decision = interrupt({"type": "ticket_draft", "draft": {"subject": subject, "description": description}})
    action = (decision or {}).get("action", "cancel")
    if action == "cancel":
        return _reply(tool_call_id, "The customer cancelled the ticket. Ask if there is anything else you can help with.")
    if action == "edit":
        subject = decision.get("subject") or subject
        description = decision.get("description") or description

    customer = state["customer"]
    with session_scope() as db:
        result = ticket_service(db).create(
            TicketCreate(
                customer_name=customer["name"], customer_email=customer["email"],
                subject=subject, description=description,
            ),
            TicketSource.CHAT,
            session_id=customer["session_id"],
            transcript=_transcript(state),
        )
    t = result.ticket
    if result.duplicate_of_existing:
        text = f"This matches the customer's open ticket {t.number}; the new details were added to it."
    else:
        text = f"Created ticket {t.number}: category {t.category}, priority {t.priority}."
    artifact = {"number": t.number, "category": t.category, "priority": str(t.priority),
                "duplicate": result.duplicate_of_existing}
    return _reply(tool_call_id, text, artifact=artifact, ticket_number=t.number)


@tool
def get_ticket_status(
    ticket_number: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Get the status and the latest support-team reply of one of the current customer's tickets (e.g. SP-00042)."""
    with session_scope() as db:
        svc = ticket_service(db)
        ticket = svc.get_by_number(ticket_number, email=state["customer"]["email"])
        if ticket is None:
            return _reply(tool_call_id, f"No ticket {ticket_number} was found for this customer.")
        reply = svc.latest_staff_reply(ticket)
        text = (
            f"Ticket {ticket.number}: status {ticket.status}, category {ticket.category}, priority {ticket.priority}, "
            f"subject '{ticket.subject}'. "
            + (f"Latest reply from support: {reply.body}" if reply else "No reply from the support team yet.")
        )
    return _reply(tool_call_id, text)


@tool
def escalate_to_human(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Hand the conversation to a human support agent. Creates a ticket first if none exists."""
    customer = state["customer"]
    with session_scope() as db:
        svc = ticket_service(db)
        ticket = svc.get_by_number(state["ticket_number"], customer["email"]) if state.get("ticket_number") else None
        if ticket is None:
            user_text = _last_user_text(state) or reason
            ticket_out = svc.create(
                TicketCreate(
                    customer_name=customer["name"], customer_email=customer["email"],
                    subject=f"Escalation: {reason}"[:255], description=user_text if len(user_text) >= 10 else reason,
                ),
                TicketSource.CHAT,
                session_id=customer["session_id"],
                transcript=_transcript(state),
            ).ticket
            ticket = svc.get(ticket_out.id)
        svc.escalate(ticket.id, reason)
        number = ticket.number
    return _reply(tool_call_id, f"Escalated as ticket {number}.", artifact={"number": number},
                  ticket_number=number, handed_off=True)


ALL_TOOLS = [search_docs, create_ticket, get_ticket_status, escalate_to_human]
