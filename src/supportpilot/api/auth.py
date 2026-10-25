"""Staff auth: one demo staff account from settings, JWT bearer tokens."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supportpilot.api.schemas import LoginRequest, TokenOut
from supportpilot.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


@router.post("/login", response_model=TokenOut)
def login(body: LoginRequest) -> TokenOut:
    s = get_settings()
    if body.username != s.staff_username or body.password != s.staff_password:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    exp = datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes)
    token = jwt.encode({"sub": body.username, "exp": exp}, s.jwt_secret, algorithm="HS256")
    return TokenOut(access_token=token)


def require_staff(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        payload = jwt.decode(creds.credentials, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return payload["sub"]
