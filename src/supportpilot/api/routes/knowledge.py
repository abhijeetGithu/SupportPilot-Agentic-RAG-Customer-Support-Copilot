"""Knowledge loop: KB drafts from resolved tickets, staff review and publishing, doc-gap report,
and the public page for published articles (the target of KB citations in the chat)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from supportpilot.api import deps
from supportpilot.api.auth import require_staff
from supportpilot.db.base import session_scope
from supportpilot.db.models import KbArticle, KbStatus, Ticket
from supportpilot.knowledge import drafting, gaps, publish, related
from supportpilot.knowledge.publish import article_url
from supportpilot.knowledge.schemas import (
    GapReport,
    KbArticleOut,
    KbArticleUpdate,
    KbPublicArticle,
    KbSuggestion,
)

log = logging.getLogger(__name__)

admin = APIRouter(prefix="/admin", tags=["knowledge"], dependencies=[Depends(require_staff)])
public = APIRouter(tags=["knowledge"])
Staff = Annotated[str, Depends(require_staff)]


def _article(db, article_id: int) -> KbArticle:
    a = db.get(KbArticle, article_id)
    if a is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Article not found")
    return a


def existing_article(db, ticket_id: int) -> KbArticle | None:
    return db.scalars(select(KbArticle).where(KbArticle.source_ticket_id == ticket_id,
                                              KbArticle.status != KbStatus.REJECTED.value)).first()


def draft_for_ticket(db, ticket: Ticket) -> KbArticle:
    if not drafting.has_resolution(ticket):
        raise HTTPException(status.HTTP_409_CONFLICT, "The ticket has no reply from the support team to learn from")
    if existing := existing_article(db, ticket.id):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Already drafted as {existing.number}")
    return drafting.draft_article(db, ticket)


def draft_in_background(ticket_id: int) -> None:
    """Runs after a ticket is resolved. Failures (LLM limits, network) only skip the draft."""
    try:
        with session_scope() as db:
            ticket = db.get(Ticket, ticket_id)
            if ticket and drafting.has_resolution(ticket) and not existing_article(db, ticket_id):
                a = drafting.draft_article(db, ticket)
                log.info("drafted %s from %s", a.number, ticket.number)
    except Exception:
        log.warning("KB draft for ticket %s failed", ticket_id, exc_info=True)


# ---------- staff ----------
@admin.post("/tickets/{ticket_id}/kb-draft", response_model=KbArticleOut, status_code=status.HTTP_201_CREATED)
def create_draft(ticket_id: int, db: deps.DB):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return KbArticleOut.of(draft_for_ticket(db, ticket))


@admin.get("/kb", response_model=list[KbArticleOut])
def list_articles(db: deps.DB, status_: Annotated[KbStatus | None, Query(alias="status")] = None,
                  ticket_id: int | None = None):
    q = select(KbArticle).order_by(KbArticle.created_at.desc(), KbArticle.id.desc())
    if status_:
        q = q.where(KbArticle.status == status_.value)
    if ticket_id is not None:
        q = q.where(KbArticle.source_ticket_id == ticket_id)
    linked = related.link_counts(db)
    return [KbArticleOut.of(a, linked.get(a.number, 0)) for a in db.scalars(q.limit(200))]


# declared before /kb/{article_id} so "search" isn't parsed as an id
@admin.get("/kb/search", response_model=list[KbSuggestion])
def search_articles(db: deps.DB, q: Annotated[str | None, Query(max_length=500)] = None,
                    ticket_id: int | None = None, limit: Annotated[int, Query(ge=1, le=20)] = 5):
    """Published articles for the reply box: typed search (q), or suggestions for a ticket (weak matches hidden)."""
    if q and q.strip():
        text, min_score = q.strip(), 0.0
    elif ticket_id is not None:
        ticket = db.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
        text, min_score = f"{ticket.subject}\n{ticket.description[:500]}", related.SUGGEST_MIN_SCORE
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Give q or ticket_id")
    return [KbSuggestion(id=a.id, number=a.number, title=a.title, product_area=a.product_area or "",
                         url=article_url(a), score=round(score, 4))
            for a, score in related.related_articles(db, text, limit=limit, min_score=min_score)]


@admin.get("/kb/{article_id}", response_model=KbArticleOut)
def get_article(article_id: int, db: deps.DB):
    return KbArticleOut.of(_article(db, article_id))


@admin.patch("/kb/{article_id}", response_model=KbArticleOut)
def update_article(article_id: int, body: KbArticleUpdate, db: deps.DB, staff: Staff):
    a = _article(db, article_id)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(a, field, value)
    a.reviewed_by = staff
    if a.status == KbStatus.PUBLISHED.value:
        publish.index_article(a)  # keep the search index in sync with the edit
    db.flush()
    return KbArticleOut.of(a)


@admin.post("/kb/{article_id}/publish", response_model=KbArticleOut)
def publish_article(article_id: int, db: deps.DB, staff: Staff):
    a = _article(db, article_id)
    if a.status == KbStatus.PUBLISHED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Already published")
    publish.publish(a, staff)
    db.flush()
    return KbArticleOut.of(a)


@admin.post("/kb/{article_id}/unpublish", response_model=KbArticleOut)
def unpublish_article(article_id: int, db: deps.DB, staff: Staff):
    a = _article(db, article_id)
    if a.status != KbStatus.PUBLISHED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Not published")
    publish.unpublish(a, staff)
    db.flush()
    return KbArticleOut.of(a)


@admin.post("/kb/{article_id}/reject", response_model=KbArticleOut)
def reject_article(article_id: int, db: deps.DB, staff: Staff):
    a = _article(db, article_id)
    if a.status == KbStatus.PUBLISHED.value:
        publish.remove_article(a)
    a.status, a.reviewed_by, a.published_at = KbStatus.REJECTED.value, staff, None
    db.flush()
    return KbArticleOut.of(a)


@admin.get("/gaps", response_model=GapReport)
def gap_report(db: deps.DB, days: Annotated[int, Query(ge=1, le=365)] = 30):
    return gaps.gap_report(db, days=days)


# ---------- public ----------
@public.get("/kb/{number}", response_model=KbPublicArticle)
def public_article(number: str, db: deps.DB):
    try:
        article_id = int(number.upper().removeprefix("KB-"))
    except ValueError:
        article_id = 0
    a = db.get(KbArticle, article_id) if article_id else None
    if a is None or a.status != KbStatus.PUBLISHED.value:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Article not found")
    return KbPublicArticle(number=a.number, title=a.title, body=a.body, product_area=a.product_area or "",
                           published_at=a.published_at)
