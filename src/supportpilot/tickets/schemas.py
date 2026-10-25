from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from supportpilot.db.models import Priority, TicketSource, TicketStatus


class TicketCreate(BaseModel):
    customer_name: str = Field(max_length=120)
    customer_email: EmailStr
    subject: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=10)
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)  # from POST /api/uploads


class TicketUpdate(BaseModel):
    """Staff edits. Changing category/priority logs a 'reclassified' event."""

    status: TicketStatus | None = None
    priority: Priority | None = None
    category: str | None = None
    assignee: str | None = None


class TicketOut(BaseModel):
    id: int
    number: str
    subject: str
    description: str
    customer_name: str
    customer_email: str
    category: str | None
    priority: Priority
    confidence: float | None
    priority_confidence: float | None = None
    classifier_model: str | None = None
    needs_review: bool
    status: TicketStatus
    source: TicketSource
    assignee: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TicketMessageOut(BaseModel):
    id: int
    author_type: str
    body: str
    citations: list | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AttachmentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    kind: str  # image | pdf | text
    size_bytes: int
    uploaded_by: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateResult(BaseModel):
    ticket: TicketOut
    duplicate_of_existing: bool = False
