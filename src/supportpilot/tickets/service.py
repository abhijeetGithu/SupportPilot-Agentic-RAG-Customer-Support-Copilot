"""The single code path for ticket changes (chat agent, contact form, staff dashboard, MCP).

create():  mask PII -> duplicate check -> classify (+ urgent rule, review thresholds)
           -> insert ticket, transcript messages and audit events in one transaction

The classifier and the duplicate-similarity function are injectable, so tests run without
loading DistilBERT or the embedding model.
"""

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from supportpilot.attachments import link, unlinked_session_attachments
from supportpilot.classifier.predict import TicketPrediction, classify_ticket
from supportpilot.config import get_settings
from supportpilot.db.models import (
    Attachment,
    AuthorType,
    ChatSession,
    Incident,
    IncidentStatus,
    SessionMode,
    Ticket,
    TicketEvent,
    TicketMessage,
    TicketSource,
    TicketStatus,
)
from supportpilot.knowledge.related import linked_numbers
from supportpilot.tickets.schemas import CreateResult, TicketCreate, TicketOut, TicketUpdate

OPEN_STATUSES = (TicketStatus.OPEN.value, TicketStatus.IN_PROGRESS.value, TicketStatus.ESCALATED.value)
NUMBER_RE = re.compile(r"^\s*SP-?(\d{1,9})\s*$", re.IGNORECASE)

# PII that should not be stored in ticket text (the customer's own email is kept in its own column)
PII_PATTERNS = [
    (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "[card number]"),  # 13-19 digits, may contain spaces/dashes
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
    (re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"), r"\1: [redacted]"),
]


log = logging.getLogger(__name__)


def mask_pii(text: str) -> str:
    for pattern, repl in PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def parse_number(number: str) -> int | None:
    m = NUMBER_RE.match(number or "")
    return int(m.group(1)) if m else None


def embedding_similarity(a: str, b: str) -> float:
    from supportpilot.retrieval.embeddings import embed_passages

    va, vb = (np.array(v) for v in embed_passages([a, b]))
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC, matching CURRENT_TIMESTAMP defaults in SQLite/Postgres


class TicketService:
    def __init__(
        self,
        db: Session,
        classifier: Callable[[str], TicketPrediction] = classify_ticket,
        similarity: Callable[[str, str], float] = embedding_similarity,
    ):
        self.db = db
        self.classify = classifier
        self.similarity = similarity

    # ---------- create ----------
    def create(
        self,
        data: TicketCreate,
        source: TicketSource,
        session_id: str | None = None,
        transcript: list[tuple[AuthorType, str]] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> CreateResult:
        """attachments: validated files to link. Unlinked uploads from the chat session are linked too."""
        s = get_settings()
        subject, description = mask_pii(data.subject.strip()), mask_pii(data.description.strip())

        dup = self.find_duplicate(data.customer_email, subject, description)
        if dup:
            self._message(dup, AuthorType.CUSTOMER, f"[Duplicate report merged]\n{subject}\n\n{description}")
            self._event(dup, "duplicate_merged", "system", {"source": source.value, "subject": subject})
            self.attach(dup, attachments, session_id)
            self.db.flush()
            return CreateResult(ticket=TicketOut.model_validate(dup), duplicate_of_existing=True)

        # what screenshots / PDFs show (e.g. the exact error) also informs routing; the description is unchanged
        notes = [a.summary_line() for a in self._candidates(attachments, session_id) if a.analysis]
        pred = self.classify("\n\n".join([subject, description, *notes]))
        needs_review = pred.confidence < s.review_confidence_threshold or (
            not pred.urgent_rule and pred.priority_confidence < s.review_priority_threshold
        )
        ticket = Ticket(
            session_id=session_id,
            customer_name=data.customer_name,
            customer_email=data.customer_email.lower(),
            subject=subject,
            description=description,
            category=pred.category,
            priority=pred.priority,
            confidence=pred.confidence,
            priority_confidence=pred.priority_confidence,
            classifier_model=pred.model,
            needs_review=needs_review,
            status=TicketStatus.OPEN.value,
            source=source.value,
        )
        self.db.add(ticket)
        self.db.flush()  # assigns ticket.id

        for author, body in transcript or []:
            self._message(ticket, author, mask_pii(body))
        self._event(ticket, "created", "customer" if source != TicketSource.MCP else "system", {"source": source.value})
        self._event(ticket, "classified", "ai", pred.model_dump() | {"needs_review": needs_review})
        self.attach(ticket, attachments, session_id)
        self.db.flush()
        self.db.refresh(ticket)
        if s.detect_incidents:
            self.detect_incident(ticket)
        return CreateResult(ticket=TicketOut.model_validate(ticket))

    def detect_incident(self, ticket: Ticket) -> None:
        """Link the ticket to an incident (a burst of similar reports). Never blocks ticket creation."""
        from supportpilot.incidents import detector
        from supportpilot.incidents import service as incidents

        try:
            active_before = set(self.db.scalars(
                select(Incident.id).where(Incident.status == IncidentStatus.ACTIVE.value)).all())
            incident = detector.on_ticket_created(self.db, ticket)
            if incident is None:
                return
            if incident.id in active_before:
                self._event(ticket, "incident_linked", "system", {"incident": incident.number})
                incidents.auto_reply(self.db, incident, ticket)
            else:
                for t in incident.tickets:
                    self._event(t, "incident_opened", "system", {"incident": incident.number})
            self.db.flush()
        except Exception:
            log.warning("incident detection failed for ticket %s", ticket.id, exc_info=True)

    def find_duplicate(self, email: str, subject: str, description: str) -> Ticket | None:
        """Open ticket from the same customer within the window whose text is semantically near-identical."""
        s = get_settings()
        cutoff = _utcnow() - timedelta(hours=s.duplicate_window_hours)
        candidates = self.db.scalars(
            select(Ticket)
            .where(Ticket.customer_email == email.lower(), Ticket.status.in_(OPEN_STATUSES), Ticket.created_at >= cutoff)
            .order_by(Ticket.created_at.desc())
            .limit(5)
        ).all()
        new_text = f"{subject}\n\n{description}"
        for t in candidates:
            if self.similarity(new_text, f"{t.subject}\n\n{t.description}") >= s.duplicate_similarity:
                return t
        return None

    # ---------- read ----------
    def get(self, ticket_id: int) -> Ticket | None:
        return self.db.get(Ticket, ticket_id)

    def get_by_number(self, number: str, email: str | None = None) -> Ticket | None:
        """If email is given, only return the ticket when it belongs to that customer."""
        tid = parse_number(number)
        ticket = self.db.get(Ticket, tid) if tid else None
        if ticket and email and ticket.customer_email != email.lower():
            return None
        return ticket

    def latest_staff_reply(self, ticket: Ticket) -> TicketMessage | None:
        replies = [m for m in ticket.messages if m.author_type == AuthorType.STAFF.value]
        return replies[-1] if replies else None

    def session_ticket(self, session_id: str) -> Ticket | None:
        """Most recent ticket created from a chat session."""
        return self.db.scalars(
            select(Ticket).where(Ticket.session_id == session_id).order_by(Ticket.id.desc()).limit(1)
        ).first()

    def list_tickets(
        self,
        status: str | None = None,
        priority: str | None = None,
        category: str | None = None,
        needs_review: bool | None = None,
        q: str | None = None,
        unassigned: bool = False,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[list[Ticket], int]:
        stmt = select(Ticket)
        if status:
            stmt = stmt.where(Ticket.status == status)
        if priority:
            stmt = stmt.where(Ticket.priority == priority)
        if category:
            stmt = stmt.where(Ticket.category == category)
        if needs_review is not None:
            stmt = stmt.where(Ticket.needs_review == needs_review)
        if unassigned:
            stmt = stmt.where(Ticket.assignee.is_(None))
        if q:
            like = f"%{q.lower()}%"
            stmt = stmt.where(
                func.lower(Ticket.subject).like(like) | func.lower(Ticket.description).like(like)
                | func.lower(Ticket.customer_email).like(like)
            )
        total = self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        # urgent/high first, then newest
        rank = case({"urgent": 0, "high": 1, "medium": 2, "low": 3}, value=Ticket.priority, else_=4)
        items = self.db.scalars(
            stmt.order_by(rank, Ticket.id.desc()).offset((page - 1) * page_size).limit(page_size)
        ).all()
        return list(items), total

    def metrics(self) -> dict:
        def counts(column) -> dict[str, int]:
            rows = self.db.execute(select(column, func.count()).group_by(column)).all()
            return {str(k): v for k, v in rows if k is not None}

        sessions = self.db.scalar(select(func.count()).select_from(ChatSession)) or 0
        with_ticket = self.db.scalar(
            select(func.count(func.distinct(Ticket.session_id))).where(Ticket.session_id.is_not(None))
        ) or 0
        return {
            "tickets_by_status": counts(Ticket.status),
            "tickets_by_category": counts(Ticket.category),
            "tickets_by_priority": counts(Ticket.priority),
            "needs_review": self.db.scalar(select(func.count()).where(Ticket.needs_review.is_(True))) or 0,
            "chat_sessions": sessions,
            # share of chats the assistant handled without a ticket
            "ai_resolution_rate": round(1 - with_ticket / sessions, 3) if sessions else 0.0,
        }

    # ---------- update ----------
    def update(self, ticket_id: int, changes: TicketUpdate, actor: str) -> Ticket:
        ticket = self._require(ticket_id)
        for field, value in changes.model_dump(exclude_none=True).items():
            value = getattr(value, "value", value)
            old = getattr(ticket, field)
            if old == value:
                continue
            setattr(ticket, field, value)
            kind = {"status": "status_changed", "assignee": "assigned"}.get(field, "reclassified")
            # 'reclassified' events (category/priority corrections) double as classifier training data
            self._event(ticket, kind, actor, {"field": field, "old": old, "new": value})
        if changes.category or changes.priority:
            ticket.needs_review = False
        if ticket.status in (TicketStatus.RESOLVED.value, TicketStatus.CLOSED.value) and ticket.session_id:
            session = self.db.get(ChatSession, ticket.session_id)
            if session and session.mode == SessionMode.HUMAN.value:
                session.mode = SessionMode.AI.value  # hand the chat back to the assistant
        self.db.flush()
        return ticket

    def add_message(
        self, ticket_id: int, author: AuthorType, body: str, citations: list | None = None
    ) -> TicketMessage:
        ticket = self._require(ticket_id)
        msg = self._message(ticket, author, mask_pii(body), citations)
        self._event(ticket, "reply", author.value, {"message_chars": len(body)})
        if author == AuthorType.STAFF and (articles := linked_numbers(body)):
            self._event(ticket, "kb_linked", author.value, {"articles": articles})  # article usage stats
        self.db.flush()
        return msg

    def escalate(self, ticket_id: int, reason: str) -> Ticket:
        """Set status to escalated and switch the linked chat session to human mode."""
        ticket = self._require(ticket_id)
        ticket.status = TicketStatus.ESCALATED.value
        self._event(ticket, "escalated", "ai", {"reason": reason})
        if ticket.session_id:
            session = self.db.get(ChatSession, ticket.session_id)
            if session:
                session.mode = SessionMode.HUMAN.value
        self.db.flush()
        return ticket

    def attach(self, ticket: Ticket, attachments: list[Attachment] | None = None, session_id: str | None = None) -> None:
        """Link files (explicit ones plus unlinked uploads from the chat session) to the ticket and log it."""
        new = link(self._candidates(attachments, session_id), ticket.id)
        if new:
            self._event(ticket, "attachments_added", "customer", {"files": [a.filename for a in new]})

    # ---------- helpers ----------
    def _candidates(self, attachments: list[Attachment] | None, session_id: str | None) -> list[Attachment]:
        candidates = list(attachments or [])
        if session_id:
            candidates += [a for a in unlinked_session_attachments(self.db, session_id) if a not in candidates]
        return candidates

    def _require(self, ticket_id: int) -> Ticket:
        ticket = self.db.get(Ticket, ticket_id)
        if ticket is None:
            raise LookupError(f"ticket {ticket_id} not found")
        return ticket

    def _message(self, ticket: Ticket, author: AuthorType, body: str, citations: list | None = None) -> TicketMessage:
        msg = TicketMessage(ticket_id=ticket.id, author_type=author.value, body=body, citations=citations)
        self.db.add(msg)
        return msg

    def _event(self, ticket: Ticket, kind: str, actor: str, payload: dict | None = None) -> None:
        self.db.add(TicketEvent(ticket_id=ticket.id, type=kind, actor=actor, payload=payload))


# ---------- chat sessions ----------
def create_chat_session(db: Session, name: str, email: str) -> ChatSession:
    session = ChatSession(customer_name=name, customer_email=email.lower())
    db.add(session)
    db.flush()
    return session


def get_chat_session(db: Session, session_id: str) -> ChatSession | None:
    return db.get(ChatSession, session_id)
