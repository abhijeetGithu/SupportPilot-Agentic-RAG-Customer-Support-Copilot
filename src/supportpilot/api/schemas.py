from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from supportpilot.tickets.schemas import AttachmentOut, TicketMessageOut, TicketOut


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr


class SessionOut(BaseModel):
    session_id: str
    mode: Literal["ai", "human"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)  # from POST /api/uploads


class ResumeRequest(BaseModel):
    """Customer's answer to a ticket_draft interrupt."""

    action: Literal["approve", "edit", "cancel"]
    subject: str | None = Field(default=None, max_length=255)
    description: str | None = None


class Citation(BaseModel):
    n: int
    title: str
    url: str
    heading: str | None = None


class ChatMessage(BaseModel):
    role: Literal["customer", "ai", "staff"]
    content: str
    id: int | None = None  # ticket message id (staff replies), used for polling
    attachments: list[dict] = []  # {id, filename, kind, size_bytes} for customer messages


class ChatHistory(BaseModel):
    session_id: str
    mode: Literal["ai", "human"]
    ticket_number: str | None
    messages: list[ChatMessage]
    pending_draft: dict | None = None


class StaffMessages(BaseModel):
    mode: Literal["ai", "human"]
    ticket_number: str | None
    messages: list[ChatMessage]


class TicketLookup(BaseModel):
    number: str
    email: EmailStr


class ReplyCreate(BaseModel):
    body: str = Field(min_length=1, max_length=8000)


class CustomerReply(ReplyCreate):
    email: EmailStr


class TicketEventOut(BaseModel):
    id: int
    type: str
    actor: str
    payload: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CustomerTicketView(BaseModel):
    """What a customer may see: no internal events, no classifier internals."""

    number: str
    subject: str
    status: str
    created_at: datetime
    updated_at: datetime
    messages: list[TicketMessageOut]
    attachments: list[AttachmentOut] = []


class TicketDetail(BaseModel):
    ticket: TicketOut
    messages: list[TicketMessageOut]
    events: list[TicketEventOut]
    attachments: list[AttachmentOut] = []


class TicketPage(BaseModel):
    items: list[TicketOut]
    total: int
    page: int
    page_size: int


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SuggestedReply(BaseModel):
    body: str
    citations: list[Citation] = []


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=3, max_length=8000)


class Metrics(BaseModel):
    tickets_by_status: dict[str, int]
    tickets_by_category: dict[str, int]
    tickets_by_priority: dict[str, int]
    needs_review: int
    chat_sessions: int
    ai_resolution_rate: float  # share of chat sessions that ended without a ticket
