"""Doc gaps: which customer questions the docs (and KB) don't cover, grouped into topics.

Every chat question is logged with its coverage (cosine similarity between the question and the
best-matching doc or KB chunk, from the dense bge index) and its outcome (answered / ticket / handoff).
A question is a gap if coverage is below the threshold or the conversation ended with a ticket or a
handoff. Gaps are grouped with agglomerative clustering on question embeddings (cosine, average
linkage, distance threshold): deterministic and needs no cluster count, which suits the tens to
low thousands of questions a support team reviews.
"""

import logging
import re
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select

from supportpilot.config import get_settings
from supportpilot.db.models import QuestionLog
from supportpilot.tickets.service import mask_pii

log = logging.getLogger(__name__)

CLUSTER_DISTANCE = 0.35  # cosine distance; 0.32-0.35 recovered 6/6 paraphrase topics (ARI 1.0, 25 questions);
# 0.35 also merges a real pair at 0.336 that 0.32 split. Unrelated topics sit around 0.5. See phase7_5b doc.
MIN_WORDS = 4  # "thanks", "ok great" aren't questions
SKIP = re.compile(r"^(i've attached (a file|some files)\.|thanks?( you)?|ok(ay)?|hi|hello)\W*$", re.IGNORECASE)


def doc_coverage(text: str) -> dict | None:
    """Best matching doc/KB chunk by dense cosine similarity: {"title", "url", "score"}."""
    from supportpilot.retrieval.retriever import dense_search

    try:
        hits = dense_search(text, top_k=1)
    except Exception:  # Qdrant down must not break chat or drafting
        log.warning("coverage lookup failed", exc_info=True)
        return None
    if not hits:
        return None
    h = hits[0]
    return {"title": h.title or h.doc_id, "url": h.url, "score": round(float(h.score), 4)}


def is_question(text: str) -> bool:
    t = text.strip()
    return len(t.split()) >= MIN_WORDS and not SKIP.match(t)


def log_question(db, session_id: str | None, text: str, outcome: str, coverage_fn=None) -> QuestionLog | None:
    if not is_question(text):
        return None
    question = mask_pii(text.strip())[:2000]
    cov = (coverage_fn or doc_coverage)(question)
    entry = QuestionLog(session_id=session_id, question=question, outcome=outcome,
                        coverage=cov["score"] if cov else None,
                        top_title=cov["title"][:255] if cov else None, top_url=cov["url"][:500] if cov else None)
    db.add(entry)
    db.flush()
    return entry


def is_gap(q: QuestionLog, threshold: float) -> bool:
    return q.outcome in ("ticket", "handoff") or (q.coverage is not None and q.coverage < threshold)


def cluster(texts: list[str], distance: float = CLUSTER_DISTANCE, embed=None) -> list[int]:
    """Cluster label per text (singletons get their own label)."""
    if len(texts) < 2:
        return [0] * len(texts)
    if embed is None:
        from supportpilot.retrieval.embeddings import embed_passages as embed
    return average_linkage(np.asarray(embed(texts), dtype=float), distance)


def average_linkage(vectors: np.ndarray, threshold: float) -> list[int]:
    """Agglomerative clustering, cosine distance, average linkage: merge the closest pair of clusters
    while their distance is below the threshold. Same result as scikit-learn's AgglomerativeClustering
    (tested), without the dependency in the slim runtime image. O(n^3): fine for the few hundred
    gap questions a report covers."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    d = 1.0 - v @ v.T
    n = len(v)
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    members = {i: [i] for i in range(n)}
    alive = np.ones(n, dtype=bool)
    while alive.sum() > 1:
        masked = np.where(np.outer(alive, alive), d, np.inf)
        a, b = divmod(int(np.argmin(masked)), n)
        if masked[a, b] >= threshold:
            break
        # Lance-Williams update for average linkage, then fold b into a
        d[a, :] = d[:, a] = (size[a] * d[a, :] + size[b] * d[b, :]) / (size[a] + size[b])
        d[a, a] = np.inf
        size[a] += size[b]
        members[a] += members.pop(b)
        alive[b] = False
        d[b, :] = d[:, b] = np.inf
    labels = [0] * n
    for label, idx in enumerate(members.values()):
        for i in idx:
            labels[i] = label
    return labels


def gap_report(db, days: int = 30, embed=None) -> dict:
    s = get_settings()
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    logs = db.scalars(select(QuestionLog).where(QuestionLog.created_at >= since)
                      .order_by(QuestionLog.created_at)).all()
    gaps = [q for q in logs if is_gap(q, s.gap_coverage_threshold)]
    labels = cluster([q.question for q in gaps], embed=embed)

    groups: dict[int, list[QuestionLog]] = {}
    for q, label in zip(gaps, labels):
        groups.setdefault(label, []).append(q)
    topics = []
    for members in groups.values():
        covs = [q.coverage for q in members if q.coverage is not None]
        best = max(members, key=lambda q: q.coverage or 0)
        topics.append({
            "label": min((q.question for q in members), key=len)[:200],  # shortest phrasing reads best
            "count": len(members),
            "sessions": len({q.session_id for q in members}),
            "outcomes": dict(Counter(q.outcome for q in members)),
            "avg_coverage": round(sum(covs) / len(covs), 3) if covs else None,
            "closest_doc": {"title": best.top_title, "url": best.top_url} if best.top_title else None,
            "last_seen": max(q.created_at for q in members).isoformat(),
            "examples": [q.question[:300] for q in members[-5:]][::-1],
        })
    topics.sort(key=lambda t: (t["count"], t["sessions"]), reverse=True)
    return {
        "days": days,
        "threshold": s.gap_coverage_threshold,
        "questions": len(logs),
        "gap_questions": len(gaps),
        "ended_in_ticket_or_handoff": sum(q.outcome in ("ticket", "handoff") for q in logs),
        "topics": topics,
    }
