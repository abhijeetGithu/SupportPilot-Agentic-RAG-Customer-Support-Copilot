"""Customer ticket endpoints: contact form, ticket lookup, customer replies."""

from fastapi import APIRouter, HTTPException, status

from supportpilot.api import deps
from supportpilot.api.schemas import CustomerReply, CustomerTicketView, TicketLookup
from supportpilot.attachments import UploadRejected, get_customer_attachments
from supportpilot.db.models import AuthorType, Ticket, TicketSource, TicketStatus
from supportpilot.tickets.schemas import (
    AttachmentOut,
    CreateResult,
    TicketCreate,
    TicketMessageOut,
    TicketUpdate,
)
from supportpilot.tickets.service import TicketService

router = APIRouter(prefix="/tickets", tags=["tickets"])

# Same response for "doesn't exist" and "not yours", so ticket numbers can't be probed
NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "No ticket with that number for this email")


def _customer_view(ticket: Ticket) -> CustomerTicketView:
    visible = [m for m in ticket.messages if m.author_type in (AuthorType.CUSTOMER.value, AuthorType.STAFF.value)]
    return CustomerTicketView(
        number=ticket.number, subject=ticket.subject, status=ticket.status, created_at=ticket.created_at,
        updated_at=ticket.updated_at, messages=[TicketMessageOut.model_validate(m) for m in visible],
        attachments=[AttachmentOut.model_validate(a) for a in ticket.attachments],
    )


def _owned(tickets: TicketService, number: str, email: str) -> Ticket:
    ticket = tickets.get_by_number(number, email=email)
    if ticket is None:
        raise NOT_FOUND
    return ticket


@router.post("", response_model=CreateResult, status_code=status.HTTP_201_CREATED)
def create_ticket(body: TicketCreate, tickets: deps.Tickets):
    """Contact form -> TicketService.create(source=form). Attachments were uploaded without a session."""
    try:
        files = get_customer_attachments(tickets.db, body.attachment_ids, session_id=None)
    except UploadRejected as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from None
    return tickets.create(body, TicketSource.FORM, attachments=files)


@router.post("/lookup", response_model=CustomerTicketView)
def lookup_ticket(body: TicketLookup, tickets: deps.Tickets):
    return _customer_view(_owned(tickets, body.number, body.email))


@router.post("/{number}/replies", response_model=CustomerTicketView)
def customer_reply(number: str, body: CustomerReply, tickets: deps.Tickets):
    ticket = _owned(tickets, number, body.email)
    if ticket.status == TicketStatus.CLOSED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "This ticket is closed; please open a new one")
    tickets.add_message(ticket.id, AuthorType.CUSTOMER, body.body)
    if ticket.status == TicketStatus.RESOLVED.value:  # customer says it isn't solved after all
        tickets.update(ticket.id, TicketUpdate(status=TicketStatus.IN_PROGRESS), actor="customer")
    tickets.db.refresh(ticket)
    return _customer_view(ticket)
