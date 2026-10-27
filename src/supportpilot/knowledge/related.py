"""Find published KB articles for a ticket or a search box, so agents can link them in replies.

Dense (bge) search restricted to KB chunks: its cosine score is interpretable ("78% match"), unlike the
rank-based hybrid score, so the dashboard can show it and hide weak suggestions.
"""

import re

from qdrant_client import models
from sqlalchemy import select

from supportpilot.db.models import KbArticle, KbStatus, TicketEvent

KB_LINK = re.compile(r"/support/kb/(KB-\d{5,})", re.IGNORECASE)
SUGGEST_MIN_SCORE = 0.60  # auto-suggestions for a ticket; typed searches show everything


def related_articles(db, text: str, limit: int = 5, min_score: float = 0.0, searcher=None) -> list[tuple[KbArticle, float]]:
    from supportpilot.retrieval.retriever import dense_search

    kb_only = models.Filter(must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="kb"))])
    hits = (searcher or dense_search)(text, top_k=limit * 4, query_filter=kb_only)
    best: dict[int, float] = {}
    for h in hits:  # several chunks per article: keep its best score
        try:
            article_id = int(h.doc_id.removeprefix("kb/KB-"))
        except ValueError:
            continue
        best[article_id] = max(best.get(article_id, 0.0), float(h.score))
    if not best:
        return []
    published = {a.id: a for a in db.scalars(select(KbArticle).where(
        KbArticle.id.in_(best), KbArticle.status == KbStatus.PUBLISHED.value))}
    ranked = sorted(((published[i], s) for i, s in best.items() if i in published and s >= min_score),
                    key=lambda pair: pair[1], reverse=True)
    return ranked[:limit]


def linked_numbers(body: str) -> list[str]:
    """KB article numbers linked in a reply, e.g. ['KB-00003']."""
    return sorted({m.upper() for m in KB_LINK.findall(body)})


def link_counts(db) -> dict[str, int]:
    """How many staff replies linked each article (from 'kb_linked' ticket events)."""
    counts: dict[str, int] = {}
    for payload in db.scalars(select(TicketEvent.payload).where(TicketEvent.type == "kb_linked")):
        for number in (payload or {}).get("articles", []):
            counts[number] = counts.get(number, 0) + 1
    return counts
