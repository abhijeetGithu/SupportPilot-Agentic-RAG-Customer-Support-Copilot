"""Customer chat: sessions, streamed agent turns (SSE), ticket-draft resume, history, handoff polling."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from supportpilot.agent.graph import new_turn
from supportpilot.api import deps
from supportpilot.api.schemas import (
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ResumeRequest,
    SessionCreate,
    SessionOut,
    StaffMessages,
)
from supportpilot.api.streaming import stream_events, to_sse
from supportpilot.db.models import AuthorType, ChatSession, SessionMode, TicketStatus
from supportpilot.tickets.service import create_chat_session

router = APIRouter(tags=["chat"])
Session = Annotated[ChatSession, Depends(deps.chat_session)]
Graph = Annotated[object, Depends(deps.graph)]


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _customer(session: ChatSession) -> dict:
    return {"session_id": session.id, "name": session.customer_name, "email": session.customer_email}


def _sse(events) -> EventSourceResponse:
    return EventSourceResponse(iterate_in_threadpool(to_sse(events)), ping=15)


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(body: SessionCreate, db: deps.DB):
    session = create_chat_session(db, body.name, body.email)
    return SessionOut(session_id=session.id, mode=session.mode)


@router.post("/chat/{session_id}/stream")
def stream_chat(body: ChatRequest, session: Session, graph: Graph, tickets: deps.Tickets):
    """AI mode: run the agent and stream events. Human mode: store the message for the support team."""
    if session.mode == SessionMode.HUMAN.value:
        ticket = tickets.session_ticket(session.id)
        if ticket is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Session is in human mode but has no ticket")
        msg = tickets.add_message(ticket.id, AuthorType.CUSTOMER, body.message)
        tickets.db.commit()  # the stream may outlive this request's session
        return _sse(iter([{"type": "stored", "message_id": msg.id}, {"type": "done", "ticket_number": ticket.number}]))

    if graph.get_state(_cfg(session.id)).interrupts:
        raise HTTPException(status.HTTP_409_CONFLICT, "A ticket draft is waiting for approve, edit or cancel")
    graph_input = {**new_turn(body.message), "customer": _customer(session)}
    return _sse(stream_events(graph, graph_input, _cfg(session.id)))


@router.post("/chat/{session_id}/resume")
def resume_chat(body: ResumeRequest, session: Session, graph: Graph):
    """Answer the pending ticket draft (approve / edit / cancel) and stream the rest of the turn."""
    if not graph.get_state(_cfg(session.id)).interrupts:
        raise HTTPException(status.HTTP_409_CONFLICT, "No ticket draft is waiting for a decision")
    decision = body.model_dump(exclude_none=True)
    return _sse(stream_events(graph, Command(resume=decision), _cfg(session.id)))


@router.get("/chat/{session_id}/history", response_model=ChatHistory)
def history(session: Session, graph: Graph, tickets: deps.Tickets):
    """Conversation so far (for page reloads): customer and assistant messages plus staff replies."""
    state = graph.get_state(_cfg(session.id))
    messages = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            messages.append(ChatMessage(role="customer", content=str(m.content)))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            messages.append(ChatMessage(role="ai", content=str(m.content)))
    ticket = tickets.session_ticket(session.id)
    if ticket:
        messages += [ChatMessage(role="staff", content=m.body, id=m.id)
                     for m in ticket.messages if m.author_type == AuthorType.STAFF.value]
    draft = next((i.value["draft"] for i in state.interrupts if isinstance(i.value, dict)), None)
    return ChatHistory(session_id=session.id, mode=session.mode, ticket_number=state.values.get("ticket_number"),
                       messages=messages, pending_draft=draft)


@router.get("/chat/{session_id}/messages", response_model=StaffMessages)
def poll_staff_messages(session: Session, tickets: deps.Tickets, after: int = 0):
    """Handoff mode: staff replies newer than message id `after`. The widget polls this every few seconds."""
    ticket = tickets.session_ticket(session.id)
    if ticket is None:
        return StaffMessages(mode=session.mode, ticket_number=None, messages=[])
    new = [ChatMessage(role="staff", content=m.body, id=m.id)
           for m in ticket.messages if m.author_type == AuthorType.STAFF.value and m.id > after]
    mode = session.mode
    if ticket.status in (TicketStatus.RESOLVED.value, TicketStatus.CLOSED.value):
        mode = SessionMode.AI.value
    return StaffMessages(mode=mode, ticket_number=ticket.number, messages=new)
