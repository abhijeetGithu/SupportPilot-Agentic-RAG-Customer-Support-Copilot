"""Staff dashboard endpoints (JWT required)."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import object_session

from supportpilot.api import deps
from supportpilot.api.auth import require_staff
from supportpilot.api.routes.knowledge import draft_in_background
from supportpilot.api.schemas import (
    ClassifyRequest,
    IncidentBrief,
    Metrics,
    ReplyCreate,
    SuggestedReply,
    TicketDetail,
    TicketEventOut,
    TicketPage,
)
from supportpilot.classifier.predict import TicketPrediction, classify_ticket
from supportpilot.config import get_settings
from supportpilot.db.models import AuthorType, Incident, Ticket, TicketStatus
from supportpilot.tickets.schemas import AttachmentOut, TicketMessageOut, TicketOut, TicketUpdate
from supportpilot.tickets.service import TicketService
from supportpilot.tickets.suggest import suggest_reply

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_staff)])
Staff = Annotated[str, Depends(require_staff)]


def _get(tickets: TicketService, ticket_id: int) -> Ticket:
    ticket = tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return ticket


def _detail(ticket: Ticket) -> TicketDetail:
    return TicketDetail(
        ticket=TicketOut.model_validate(ticket),
        messages=[TicketMessageOut.model_validate(m) for m in ticket.messages],
        events=[TicketEventOut.model_validate(e) for e in ticket.events],
        attachments=[AttachmentOut.model_validate(a) for a in ticket.attachments],
        incident=_incident_brief(ticket),
    )


def _incident_brief(ticket: Ticket) -> IncidentBrief | None:
    if ticket.incident_id is None:
        return None
    i = object_session(ticket).get(Incident, ticket.incident_id)
    return IncidentBrief(id=i.id, number=i.number, title=i.title, status=i.status) if i else None


@router.get("/tickets", response_model=TicketPage)
def list_tickets(
    tickets: deps.Tickets,
    status_: Annotated[TicketStatus | None, Query(alias="status")] = None,
    priority: str | None = None,
    category: str | None = None,
    needs_review: bool | None = None,
    q: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
):
    items, total = tickets.list_tickets(
        status=status_.value if status_ else None, priority=priority, category=category,
        needs_review=needs_review, q=q, page=page, page_size=page_size,
    )
    return TicketPage(items=[TicketOut.model_validate(t) for t in items], total=total, page=page, page_size=page_size)


@router.get("/tickets/{ticket_id}", response_model=TicketDetail)
def get_ticket(ticket_id: int, tickets: deps.Tickets):
    return _detail(_get(tickets, ticket_id))


@router.patch("/tickets/{ticket_id}", response_model=TicketDetail)
def update_ticket(
    ticket_id: int, body: TicketUpdate, tickets: deps.Tickets, staff: Staff, background: BackgroundTasks
):
    was_resolved = _get(tickets, ticket_id).status == TicketStatus.RESOLVED.value
    ticket = tickets.update(ticket_id, body, actor=f"staff:{staff}")
    tickets.db.refresh(ticket)
    if ticket.status == TicketStatus.RESOLVED.value and not was_resolved and get_settings().auto_draft_kb:
        # learn from the fix: draft a KB article after the response is sent, in its own DB session
        tickets.db.commit()
        background.add_task(draft_in_background, ticket.id)
    return _detail(ticket)


@router.post("/tickets/{ticket_id}/replies", response_model=TicketDetail)
def staff_reply(ticket_id: int, body: ReplyCreate, tickets: deps.Tickets, staff: Staff):
    ticket = _get(tickets, ticket_id)
    tickets.add_message(ticket_id, AuthorType.STAFF, body.body)
    if ticket.status in (TicketStatus.OPEN.value, TicketStatus.ESCALATED.value):
        tickets.update(ticket_id, TicketUpdate(status=TicketStatus.IN_PROGRESS, assignee=ticket.assignee or staff),
                       actor=f"staff:{staff}")
    tickets.db.refresh(ticket)
    return _detail(ticket)


@router.post("/tickets/{ticket_id}/suggest-reply", response_model=SuggestedReply)
def suggested_reply(ticket_id: int, tickets: deps.Tickets):
    """RAG-drafted reply with citations. Staff review and send it themselves."""
    return suggest_reply(_get(tickets, ticket_id))


@router.get("/escalations", response_model=TicketPage)
def list_escalations(tickets: deps.Tickets):
    items, total = tickets.list_tickets(status=TicketStatus.ESCALATED.value, unassigned=True, page_size=100)
    return TicketPage(items=[TicketOut.model_validate(t) for t in items], total=total, page=1, page_size=100)


@router.post("/escalations/{ticket_id}/take", response_model=TicketDetail)
def take_escalation(ticket_id: int, tickets: deps.Tickets, staff: Staff):
    ticket = _get(tickets, ticket_id)
    if ticket.status != TicketStatus.ESCALATED.value or ticket.assignee:
        raise HTTPException(status.HTTP_409_CONFLICT, "Ticket is not an open escalation")
    ticket = tickets.update(ticket_id, TicketUpdate(status=TicketStatus.IN_PROGRESS, assignee=staff),
                            actor=f"staff:{staff}")
    tickets.db.refresh(ticket)
    return _detail(ticket)


@router.get("/metrics", response_model=Metrics)
def metrics(tickets: deps.Tickets):
    return tickets.metrics()


@router.post("/classify", response_model=TicketPrediction)
def classify(body: ClassifyRequest):
    """Classifier playground."""
    return classify_ticket(body.text)
