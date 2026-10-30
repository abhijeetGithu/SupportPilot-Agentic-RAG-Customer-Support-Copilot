"""Rate limits for the public demo: per-IP sliding windows, plus a daily cap on assistant turns.

The demo runs on free tiers (Groq: 200K tokens/day per model), so one visitor or bot must not be able to
use up the day's quota. In-memory: the service runs as a single instance (Render free tier).
"""

import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from threading import Lock

from fastapi import HTTPException, Request, status

from supportpilot.config import get_settings

_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_daily = {"day": "", "count": 0}
_lock = Lock()


def client_ip(request: Request) -> str:
    """The last X-Forwarded-For entry: added by Render's proxy, so the client can't forge it. (The first
    entry is whatever the client sent; trusting it would let anyone dodge the limits.)"""
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    return forwarded[-1] if forwarded else (request.client.host if request.client else "unknown")


def limit(name: str, max_calls: int, per_seconds: int, daily_cap: bool = False):
    """FastAPI dependency: at most max_calls per per_seconds per IP (and the daily cap if asked)."""

    def check(request: Request) -> None:
        s = get_settings()
        if not s.rate_limits:
            return
        now = time.monotonic()
        key = (name, client_ip(request))
        with _lock:
            q = _hits[key]
            while q and now - q[0] > per_seconds:
                q.popleft()
            if len(q) >= max_calls:
                wait = int(per_seconds - (now - q[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                    f"Too many requests. Please try again in {wait} seconds.",
                                    headers={"Retry-After": str(wait)})
            if daily_cap:
                today = datetime.now(UTC).date().isoformat()
                if _daily["day"] != today:
                    _daily.update(day=today, count=0)
                if _daily["count"] >= s.daily_chat_limit:
                    raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                        "The demo has reached today's limit for the AI assistant (free tier). "
                                        "Please come back tomorrow, or use the contact form.")
                _daily["count"] += 1
            q.append(now)

    return check


def reset() -> None:
    with _lock:
        _hits.clear()
        _daily.update(day="", count=0)
