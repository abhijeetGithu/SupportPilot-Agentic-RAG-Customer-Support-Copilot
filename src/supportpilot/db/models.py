"""Tables: chat_sessions, tickets, ticket_messages, ticket_events, attachments, kb_articles, question_logs,
incidents.
See docs/ARCHITECTURE.md §3."""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from supportpilot.db.base import Base


class SessionMode(StrEnum):
    AI = "ai"
    HUMAN = "human"


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketSource(StrEnum):
    CHAT = "chat"
    FORM = "form"
    MCP = "mcp"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AuthorType(StrEnum):
    CUSTOMER = "customer"
    AI = "ai"
    STAFF = "staff"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_name: Mapped[str] = mapped_column(String(120))
    customer_email: Mapped[str] = mapped_column(String(255), index=True)
    mode: Mapped[str] = mapped_column(String(10), default=SessionMode.AI.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("chat_sessions.id"))
    customer_name: Mapped[str] = mapped_column(String(120))
    customer_email: Mapped[str] = mapped_column(String(255), index=True)
    subject: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64), index=True)
    priority: Mapped[str] = mapped_column(String(10), default=Priority.MEDIUM.value, index=True)
    confidence: Mapped[float | None] = mapped_column(Float)  # calibrated category confidence
    priority_confidence: Mapped[float | None] = mapped_column(Float)
    classifier_model: Mapped[str | None] = mapped_column(String(20))  # distilbert | tfidf-logreg
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default=TicketStatus.OPEN.value, index=True)
    source: Mapped[str] = mapped_column(String(10))
    assignee: Mapped[str | None] = mapped_column(String(120))
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    embedding: Mapped[list | None] = mapped_column(JSON)  # normalized bge vector, for incident detection
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["TicketMessage"]] = relationship(
        back_populates="ticket", order_by="TicketMessage.id"
    )
    events: Mapped[list["TicketEvent"]] = relationship(
        back_populates="ticket", order_by="TicketEvent.id"
    )
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="ticket", order_by="Attachment.created_at"
    )

    @property
    def number(self) -> str:
        """Human-facing ticket number, e.g. SP-01042."""
        return f"SP-{self.id:05d}"


class TicketMessage(Base):
    __tablename__ = "ticket_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    author_type: Mapped[str] = mapped_column(String(10))
    body: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="messages")


class TicketEvent(Base):
    """Audit log. 'reclassified' events double as classifier feedback data."""

    __tablename__ = "ticket_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    type: Mapped[str] = mapped_column(String(20))
    actor: Mapped[str] = mapped_column(String(10))
    payload: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="events")


class Attachment(Base):
    """An uploaded file. Starts unlinked (uploaded in the chat or form) and is linked to a ticket
    when the ticket is created, or immediately if the chat session already has one."""

    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    filename: Mapped[str] = mapped_column(String(160))  # sanitized original name, for display only
    content_type: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(10))  # image | pdf | text
    size_bytes: Mapped[int]
    uploaded_by: Mapped[str] = mapped_column(String(10), default="customer")
    # what the vision model / PDF reader found (summary, visible_text, error_messages, ...); None if not analyzed
    analysis: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    ticket: Mapped[Ticket | None] = relationship(back_populates="attachments")

    def summary_line(self) -> str:
        """One line about what the file shows (from the analysis), e.g. for ticket classification."""
        a = self.analysis or {}
        errors = "; ".join(a.get("error_messages") or [])
        return " ".join(x for x in (a.get("summary", ""), a.get("product_area", ""), errors) if x)


class KbStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    REJECTED = "rejected"


class KbArticle(Base):
    """Knowledge-base article drafted by the LLM from a resolved ticket, reviewed by staff.
    Published articles are indexed in Qdrant next to the docs, so the assistant can answer with them."""

    __tablename__ = "kb_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_ticket_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)  # markdown: problem, symptoms, cause, resolution
    product_area: Mapped[str] = mapped_column(String(120), default="")
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(12), default=KbStatus.DRAFT.value, index=True)
    # closest existing doc when drafted: {"title", "url", "score"}; helps staff spot duplicates of the docs
    doc_coverage: Mapped[dict | None] = mapped_column(JSON)
    generalizable: Mapped[bool] = mapped_column(Boolean, default=True)  # LLM's view: useful beyond this customer?
    draft_note: Mapped[str] = mapped_column(Text, default="")  # why (not) generalizable
    model: Mapped[str | None] = mapped_column(String(80))
    reviewed_by: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    published_at: Mapped[datetime | None] = mapped_column(DateTime)

    source_ticket: Mapped[Ticket | None] = relationship()

    @property
    def number(self) -> str:
        return f"KB-{self.id:05d}"


class QuestionLog(Base):
    """One customer question in the chat, with how well the docs covered it. Feeds the Doc gaps page."""

    __tablename__ = "question_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    question: Mapped[str] = mapped_column(Text)  # PII-masked
    coverage: Mapped[float | None] = mapped_column(Float)  # cosine similarity of the best doc/KB chunk
    top_title: Mapped[str | None] = mapped_column(String(255))
    top_url: Mapped[str | None] = mapped_column(String(500))
    outcome: Mapped[str] = mapped_column(String(12))  # answered | ticket | handoff | error
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class IncidentStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"


class Incident(Base):
    """A burst of similar tickets from different customers (likely an outage), detected automatically."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(10), default=IncidentStatus.ACTIVE.value, index=True)
    # staff-written status message; the assistant and auto-replies only ever show this text to customers
    public_note: Mapped[str] = mapped_column(Text, default="")
    centroid: Mapped[list] = mapped_column(JSON)  # mean of member embeddings (normalized)
    detection: Mapped[dict] = mapped_column(JSON, default=dict)  # window count, baseline rate, p-value
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    resolved_by: Mapped[str | None] = mapped_column(String(80))

    tickets: Mapped[list[Ticket]] = relationship(order_by="Ticket.created_at")

    @property
    def number(self) -> str:
        return f"INC-{self.id:05d}"
