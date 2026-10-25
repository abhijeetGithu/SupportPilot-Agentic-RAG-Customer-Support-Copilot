"""Shared FastAPI dependencies. Tests override `ticket_service` and set `app.state.graph`."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from supportpilot.db.base import get_db
from supportpilot.db.models import ChatSession
from supportpilot.tickets.service import TicketService, get_chat_session

DB = Annotated[Session, Depends(get_db)]


def ticket_service(db: DB) -> TicketService:
    return TicketService(db)


Tickets = Annotated[TicketService, Depends(ticket_service)]


def graph(request: Request):
    return request.app.state.graph


def chat_session(session_id: str, db: DB) -> ChatSession:
    """The session id is an unguessable UUID and acts as the customer's bearer credential."""
    session = get_chat_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")
    return session
