from datetime import datetime

from pydantic import BaseModel, Field

from supportpilot.db.models import KbArticle
from supportpilot.knowledge.publish import article_url


class KbArticleOut(BaseModel):
    id: int
    number: str
    source_ticket_id: int | None
    source_ticket_number: str | None
    title: str
    body: str
    product_area: str
    keywords: list[str]
    status: str
    doc_coverage: dict | None
    generalizable: bool
    draft_note: str
    model: str | None
    reviewed_by: str | None
    url: str
    linked_in_replies: int = 0  # staff replies that linked this article
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None

    @classmethod
    def of(cls, a: KbArticle, linked: int = 0) -> "KbArticleOut":
        return cls(
            id=a.id, number=a.number, source_ticket_id=a.source_ticket_id,
            source_ticket_number=a.source_ticket.number if a.source_ticket else None,
            title=a.title, body=a.body, product_area=a.product_area or "", keywords=a.keywords or [],
            status=a.status, doc_coverage=a.doc_coverage, generalizable=a.generalizable,
            draft_note=a.draft_note or "", model=a.model, reviewed_by=a.reviewed_by, url=article_url(a),
            linked_in_replies=linked,
            created_at=a.created_at, updated_at=a.updated_at, published_at=a.published_at,
        )


class KbArticleUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=5, max_length=255)
    body: str | None = Field(default=None, min_length=20)
    product_area: str | None = Field(default=None, max_length=120)
    keywords: list[str] | None = Field(default=None, max_length=12)


class KbSuggestion(BaseModel):
    """A published article an agent can link in a reply."""

    id: int
    number: str
    title: str
    product_area: str
    url: str
    score: float  # cosine similarity to the ticket / search text


class KbPublicArticle(BaseModel):
    number: str
    title: str
    body: str
    product_area: str
    published_at: datetime | None


class GapTopic(BaseModel):
    label: str
    count: int
    sessions: int
    outcomes: dict[str, int]
    avg_coverage: float | None
    closest_doc: dict | None
    last_seen: str
    examples: list[str]


class GapReport(BaseModel):
    days: int
    threshold: float
    questions: int
    gap_questions: int
    ended_in_ticket_or_handoff: int
    topics: list[GapTopic]
