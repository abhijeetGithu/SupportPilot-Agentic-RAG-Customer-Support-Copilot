"""Staff incident console: detected incidents, one update to all affected customers, resolve, corrections."""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from supportpilot.api import deps
from supportpilot.api.auth import require_staff
from supportpilot.db.models import Incident, IncidentStatus, Ticket, TicketEvent
from supportpilot.incidents import service as incidents
from supportpilot.tickets.schemas import TicketOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/incidents", tags=["incidents"], dependencies=[Depends(require_staff)])
Staff = Annotated[str, Depends(require_staff)]


class IncidentOut(BaseModel):
    id: int
    number: str
    title: str
    summary: str
    status: str
    public_note: str
    tickets: int
    customers: int
    open_tickets: int
    detection: dict
    first_seen: datetime
    last_seen: datetime
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None

    @classmethod
    def of(cls, i: Incident) -> "IncidentOut":
        return cls(id=i.id, number=i.number, title=i.title, summary=i.summary or "", status=i.status,
                   public_note=i.public_note or "", tickets=len(i.tickets),
                   customers=len({t.customer_email for t in i.tickets}),
                   open_tickets=sum(t.status in incidents.OPEN for t in i.tickets), detection=i.detection or {},
                   first_seen=i.first_seen, last_seen=i.last_seen, created_at=i.created_at,
                   resolved_at=i.resolved_at, resolved_by=i.resolved_by)


class IncidentDetail(BaseModel):
    incident: IncidentOut
    tickets: list[TicketOut]


class IncidentPatch(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=255)
    summary: str | None = None
    public_note: str | None = Field(default=None, max_length=2000)


class UpdateBody(BaseModel):
    body: str = Field(min_length=5, max_length=5000)


class ResolveBody(BaseModel):
    message: str | None = Field(default=None, max_length=5000)
    resolve_tickets: bool = True


def _get(db, incident_id: int) -> Incident:
    i = db.get(Incident, incident_id)
    if i is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return i


def _detail(i: Incident) -> IncidentDetail:
    return IncidentDetail(incident=IncidentOut.of(i), tickets=[TicketOut.model_validate(t) for t in i.tickets])


@router.get("", response_model=list[IncidentOut])
def list_incidents(db: deps.DB, status_: Annotated[IncidentStatus | None, Query(alias="status")] = None):
    q = select(Incident).order_by(Incident.status.asc(), Incident.last_seen.desc())  # active first
    if status_:
        q = q.where(Incident.status == status_.value)
    return [IncidentOut.of(i) for i in db.scalars(q.limit(100))]


@router.get("/{incident_id}", response_model=IncidentDetail)
def get_incident(incident_id: int, db: deps.DB):
    return _detail(_get(db, incident_id))


@router.patch("/{incident_id}", response_model=IncidentDetail)
def update_incident(incident_id: int, body: IncidentPatch, db: deps.DB):
    i = _get(db, incident_id)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(i, field, value)
    db.flush()
    return _detail(i)


@router.post("/{incident_id}/summarize", response_model=IncidentDetail)
def summarize_incident(incident_id: int, db: deps.DB):
    """AI title and summary from the reports (for staff; never shown to customers)."""
    i = _get(db, incident_id)
    try:
        incidents.summarize(i)
    except Exception as e:  # LLM limits or a bad JSON reply: tell staff, keep the current title
        log.warning("incident summary failed", exc_info=True)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Summary failed: {e}"[:300]) from None
    db.flush()
    return _detail(i)


@router.post("/{incident_id}/updates")
def send_update(incident_id: int, body: UpdateBody, db: deps.DB, staff: Staff) -> dict:
    i = _get(db, incident_id)
    if i.status != IncidentStatus.ACTIVE.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Incident is resolved")
    return {"sent_to": incidents.post_update(db, i, body.body, staff)}


@router.post("/{incident_id}/resolve", response_model=IncidentDetail)
def resolve_incident(incident_id: int, body: ResolveBody, db: deps.DB, staff: Staff):
    i = _get(db, incident_id)
    if i.status != IncidentStatus.ACTIVE.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Already resolved")
    incidents.resolve(db, i, staff, body.message, body.resolve_tickets)
    return _detail(i)


@router.delete("/{incident_id}/tickets/{ticket_id}", response_model=IncidentDetail)
def remove_ticket(incident_id: int, ticket_id: int, db: deps.DB, staff: Staff):
    """Correct a wrong link: the ticket goes back to being handled on its own."""
    i = _get(db, incident_id)
    t = db.get(Ticket, ticket_id)
    if t is None or t.incident_id != i.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket is not part of this incident")
    t.incident_id = None
    db.add(TicketEvent(ticket_id=t.id, type="incident_unlinked", actor=f"staff:{staff}",
                       payload={"incident": i.number}))
    db.flush()
    db.refresh(i)
    return _detail(i)
