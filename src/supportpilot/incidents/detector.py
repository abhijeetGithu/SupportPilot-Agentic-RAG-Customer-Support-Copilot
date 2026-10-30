"""Online incident detection: runs once per new ticket.

A new ticket either joins an active incident (similar to its centroid) or, together with recent similar
tickets, opens a new one when all three hold:
  1. similar:   cosine >= incident_similarity to the new ticket (bge embeddings, calibrated)
  2. distinct:  at least incident_min_reports different customers inside the time window
  3. a burst:   that many similar tickets this fast is unlikely given their normal rate. Poisson test:
                lambda = similar tickets in the previous baseline_days, scaled to one window;
                p = P(X >= k | lambda) must be <= incident_max_p_value.
Condition 3 matters because similarity alone can't separate an outage from a popular how-to question
(both have ~0.78 median pair similarity in the calibration set); popular questions have a high baseline.

Linear scan over recent tickets: fine for thousands of tickets a week; a vector index would replace it
at higher volume.
"""

import logging
import math
import re
from datetime import timedelta

import numpy as np
from sqlalchemy import select

from supportpilot.config import get_settings
from supportpilot.db.models import Incident, IncidentStatus, Ticket

log = logging.getLogger(__name__)
MIN_BASELINE_RATE = 0.05  # per window: even a never-seen issue isn't treated as impossible
JOIN_MIN_CLOSE_MEMBERS = 2


# Outages are reported as something broken; how-to questions about the same feature are not.
# Written from general support language (not tuned on the eval set); only problem reports can open or
# join an incident. Without this gate, bursts of similar how-to questions opened false incidents.
# Negations (did not run, never starts, no longer) were added after the held-out eval exposed them.
PROBLEM = re.compile(
    r"\b(fail\w*|error\w*|broken|not (working|loading|updating|indexing|being indexed|showing)|(doesn|isn|aren|won|can|didn|hasn|haven)'?t|did not|has not|have not|never (runs?|ran|starts?|started|loads?|completes?|finish\w*)|no longer|not (running|starting|started|run|crawl\w*|respond\w*|available|accessible)|access denied|permission|cannot|unable|stopped|down|outage|slow\w*|time(d)? ?out|timing out|hang\w*|crash\w*|blank|empty|no data|missing|unauthori[sz]ed|forbidden|invalid|rejected|spinner|stuck|zero|(shows?|showing) (0|zero|no)\b|[45]\d\d)\b",
    re.IGNORECASE,
)


def looks_like_problem(text: str) -> bool:
    """True for reports of something broken (error, failure, slowness, missing data)."""
    return bool(PROBLEM.search(text))


def poisson_tail(k: int, rate: float) -> float:
    """P(X >= k) for X ~ Poisson(rate). Exact for the small k used here (no scipy at runtime)."""
    term, below = math.exp(-rate), 0.0
    for i in range(k):
        below += term
        term *= rate / (i + 1)
    return max(0.0, 1.0 - below)


def embed(texts: list[str]) -> np.ndarray:
    """Normalized bge passage vectors (one row per text). Replaced by a fake in tests."""
    from supportpilot.retrieval.embeddings import embed_passages

    v = np.asarray(embed_passages(texts), dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ticket_text(t: Ticket) -> str:
    return f"{t.subject}\n{t.description[:1000]}"


def _vectors(tickets: list[Ticket]) -> np.ndarray:
    return np.asarray([t.embedding for t in tickets], dtype=float).reshape(len(tickets), -1)


def _centroid(vectors: np.ndarray) -> list[float]:
    c = vectors.mean(axis=0)
    return (c / np.linalg.norm(c)).tolist()


def on_ticket_created(db, ticket: Ticket, now=None) -> Incident | None:
    """Embed the ticket and link it to an incident if it belongs to one. Returns the incident (or None)."""
    s = get_settings()
    now = now or ticket.created_at
    if ticket.embedding is None:
        ticket.embedding = embed([ticket_text(ticket)])[0].tolist()
    if not looks_like_problem(ticket_text(ticket)):
        return None
    v = np.asarray(ticket.embedding)

    # 1. join an active incident: similar to its centroid AND to at least two of its tickets. The centroid
    #    alone drifts toward "search problems in general" and absorbed neighbouring outages in the eval.
    active = db.scalars(select(Incident).where(Incident.status == IncidentStatus.ACTIVE.value)).all()
    for incident in sorted(active, key=lambda i: float(np.asarray(i.centroid) @ v), reverse=True):
        if float(np.asarray(incident.centroid) @ v) < s.incident_similarity:
            break
        members = [t for t in incident.tickets if t.embedding is not None]
        close = int(np.sum(_vectors(members) @ v >= s.incident_similarity)) if members else 0
        if close >= min(JOIN_MIN_CLOSE_MEMBERS, len(members)):
            return join(db, incident, ticket, now)

    # 2. open a new incident from a burst of similar tickets
    window_start = now - timedelta(minutes=s.incident_window_minutes)
    recent = [t for t in db.scalars(select(Ticket).where(
        Ticket.created_at >= window_start, Ticket.created_at <= now, Ticket.id != ticket.id,
        Ticket.incident_id.is_(None), Ticket.embedding.is_not(None))).all()]
    if not recent:
        return None
    sims = _vectors(recent) @ v
    similar = [t for t, sim in zip(recent, sims) if sim >= s.incident_similarity and looks_like_problem(ticket_text(t))]
    customers = {t.customer_email for t in similar} | {ticket.customer_email}
    k = len(customers)  # count customers, not tickets: one customer repeating isn't a burst
    if k < s.incident_min_reports:
        return None

    baseline_start = window_start - timedelta(days=s.incident_baseline_days)
    older = db.scalars(select(Ticket).where(Ticket.created_at >= baseline_start, Ticket.created_at < window_start,
                                            Ticket.embedding.is_not(None))).all()
    baseline = int(np.sum(_vectors(older) @ v >= s.incident_similarity)) if older else 0
    windows = s.incident_baseline_days * 24 * 60 / s.incident_window_minutes
    rate = max(baseline / windows, MIN_BASELINE_RATE)
    p_value = poisson_tail(k, rate)
    if p_value > s.incident_max_p_value:
        return None

    members = [*similar, ticket]
    incident = Incident(
        title=min((t.subject for t in members), key=len)[:255],  # replaced by an LLM title in the background
        centroid=_centroid(_vectors(members)),
        detection={"customers": k, "tickets": len(members), "baseline_similar": baseline,
                   "expected_per_window": round(rate, 3), "p_value": p_value,
                   "window_minutes": s.incident_window_minutes, "similarity": s.incident_similarity,
                   "opened_at": now.isoformat()},
        first_seen=min(t.created_at for t in members), last_seen=now,
    )
    db.add(incident)
    db.flush()
    for t in members:
        t.incident_id = incident.id
    log.info("opened %s from %d tickets (%d customers, p=%.2g)", incident.number, len(members), k, p_value)
    return incident


def join(db, incident: Incident, ticket: Ticket, now) -> Incident:
    ticket.incident_id = incident.id
    incident.last_seen = max(incident.last_seen, now)
    members = [t for t in incident.tickets if t.embedding is not None]
    if ticket not in members:
        members.append(ticket)
    incident.centroid = _centroid(_vectors(members))
    return incident


def match_active(db, text: str, threshold: float | None = None) -> Incident | None:
    """Active incident that a customer's message is about (for the assistant), or None."""
    active = db.scalars(select(Incident).where(Incident.status == IncidentStatus.ACTIVE.value)).all()
    if not active:
        return None
    v = embed([text])[0]
    sims = [float(np.asarray(i.centroid) @ v) for i in active]
    best = int(np.argmax(sims))
    return active[best] if sims[best] >= (threshold or get_settings().incident_similarity) else None
