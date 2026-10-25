"""LangGraph state for one chat session (thread_id = chat session id)."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Customer(TypedDict):
    """Set by the API from the chat session, never by the LLM."""

    session_id: str
    name: str
    email: str


class Citation(TypedDict):
    n: int  # the [n] the answer refers to
    title: str
    url: str
    heading: str | None


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer: Customer
    steps: int  # LLM calls in the current turn (reset by the caller on each user message)
    citations: list[Citation]  # sources retrieved in the current turn (reset by the caller)
    ticket_number: str | None  # set once a ticket exists for this session
    handed_off: bool
