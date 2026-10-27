"""Working with incidents: AI summary, one update to every affected customer, resolve, auto-replies.

Customers only ever see text staff wrote (the public note and updates); the AI title and summary are for
staff, because they are generated from other customers' tickets.
"""

import json
import logging
from datetime import UTC, datetime

from supportpilot.db.models import (
    AuthorType,
    Incident,
    IncidentStatus,
    Ticket,
    TicketEvent,
    TicketMessage,
    TicketStatus,
)
from supportpilot.llm.provider import get_llm
from supportpilot.tickets.service import mask_pii

log = logging.getLogger(__name__)
OPEN = (TicketStatus.OPEN.value, TicketStatus.IN_PROGRESS.value, TicketStatus.ESCALATED.value)

SUMMARY_PROMPT = """Several customers reported what looks like the same problem with SearchUnify within a short time.
Write a JSON object for the support team:
- "title": a short incident title naming the symptom and the affected area, e.g. "Salesforce crawls failing with HTTP 401"
- "summary": 1-2 sentences: what customers see, any shared error messages, and what they have in common.
Use only the reports below. No customer names or companies.

Reports (oldest first):
{reports}"""


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def summarize(incident: Incident, llm=None) -> Incident:
    reports = "\n".join(f"- {t.subject}: {mask_pii(t.description)[:300]}" for t in incident.tickets[:15])
    llm = llm or get_llm(reasoning_effort="low", max_tokens=600)
    out = llm.bind(response_format={"type": "json_object"}).invoke(SUMMARY_PROMPT.format(reports=reports))
    data = json.loads(str(out.content))
    incident.title = str(data.get("title") or incident.title)[:255]
    incident.summary = str(data.get("summary") or "")
    return incident


def _staff_message(db, ticket: Ticket, body: str, actor: str, event: str, payload: dict) -> None:
    db.add(TicketMessage(ticket_id=ticket.id, author_type=AuthorType.STAFF.value, body=body))
    db.add(TicketEvent(ticket_id=ticket.id, type=event, actor=actor, payload=payload))


def post_update(db, incident: Incident, body: str, staff: str) -> int:
    """Send the same update to every open ticket in the incident. Returns how many customers got it."""
    open_tickets = [t for t in incident.tickets if t.status in OPEN]
    for t in open_tickets:
        _staff_message(db, t, body, f"staff:{staff}", "incident_update", {"incident": incident.number})
        if t.status == TicketStatus.OPEN.value:
            t.status = TicketStatus.IN_PROGRESS.value
    db.flush()
    return len(open_tickets)


def resolve(db, incident: Incident, staff: str, message: str | None = None, resolve_tickets: bool = True) -> int:
    """Close the incident; optionally post a final message and resolve its open tickets."""
    count = 0
    if message:
        count = post_update(db, incident, message, staff)
    if resolve_tickets:
        for t in incident.tickets:
            if t.status in OPEN:
                t.status = TicketStatus.RESOLVED.value
    incident.status = IncidentStatus.RESOLVED.value
    incident.resolved_at, incident.resolved_by = _now(), staff
    db.flush()
    return count


def auto_reply(db, incident: Incident, ticket: Ticket) -> bool:
    """A new ticket joined an incident that has a public note: tell the customer right away."""
    if not incident.public_note.strip():
        return False
    body = (f"Thanks for reporting this. We're aware of this issue and are working on it ({incident.number}):\n\n"
            f"{incident.public_note.strip()}\n\nWe'll update this ticket when it's resolved.")
    _staff_message(db, ticket, body, "system", "incident_linked", {"incident": incident.number})
    return True


def customer_notice(incident: Incident) -> str:
    """What the assistant may tell a customer about an active incident (staff text only)."""
    note = incident.public_note.strip()
    return note or "Our team is aware of reports matching this problem and is investigating."
