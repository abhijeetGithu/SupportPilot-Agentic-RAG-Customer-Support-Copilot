"""Staff auth: an admin account and an optional read-only demo account (both from settings), JWT bearer.

The demo account lets visitors (recruiters) explore the dashboard on the public deployment without
being able to change anything: every non-GET staff request is refused, except a few read-only POSTs.
"""

import hmac
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supportpilot.api.ratelimit import limit
from supportpilot.api.schemas import LoginRequest, TokenOut
from supportpilot.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)
VIEWER_ALLOWED_POSTS = ("/suggest-reply", "/classify")  # compute only, nothing is stored


def _matches(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())  # constant time


@router.post("/login", response_model=TokenOut, dependencies=[Depends(limit("login", 10, 600))])
def login(body: LoginRequest) -> TokenOut:
    s = get_settings()
    if _matches(body.username, s.staff_username) and _matches(body.password, s.staff_password):
        role = "admin"
    elif s.demo_password and _matches(body.username, s.demo_username) and _matches(body.password, s.demo_password):
        role = "viewer"
    else:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    exp = datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes)
    token = jwt.encode({"sub": body.username, "role": role, "exp": exp}, s.jwt_secret, algorithm="HS256")
    return TokenOut(access_token=token, role=role)


def require_staff(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        payload = jwt.decode(creds.credentials, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from None
    if (payload.get("role") == "viewer" and request.method not in ("GET", "HEAD")
            and not request.url.path.endswith(VIEWER_ALLOWED_POSTS)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The demo account is read-only.")
    return payload["sub"]
