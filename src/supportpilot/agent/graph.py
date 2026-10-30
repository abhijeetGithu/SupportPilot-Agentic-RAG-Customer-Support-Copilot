"""LangGraph agent.

    START -> agent --(tool calls)--> tools --> agent -> ... -> verify -> guard -> END
                   \\--(no tool calls)--> verify -> guard -> END
             tools --(escalated)--> handoff -> END

- One tool call per LLM turn (parallel_tool_calls=False): create_ticket pauses the graph with interrupt(),
  and LangGraph re-runs the whole tool step on resume, so parallel calls would repeat their side effects.
- Step limit: after MAX_AGENT_STEPS LLM calls in one turn the agent stops and offers a ticket.
- Memory: the checkpointer stores the full state per thread_id (= chat session id).
- verify (Phase 8): if the turn searched the docs, a separate model checks the answer's claims against the
  passages; unsupported claims trigger a rewrite from the passages (or "not in the docs" + ticket offer).
- guard (Phase 8): output guard (prompt leaks via canary + verbatim overlap, other people's emails,
  external markdown images). The stream is checked for leaks as it goes (api/streaming.py).
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from supportpilot.agent import grounding
from supportpilot.agent.prompts import MAX_STEPS_REPLY, handoff_reply, rules_text, system_prompt
from supportpilot.agent.state import AgentState
from supportpilot.agent.tools import ALL_TOOLS
from supportpilot.config import get_settings
from supportpilot.guardrails import output_guard
from supportpilot.llm.provider import get_llm

PROMPT_NGRAMS = output_guard.prompt_ngrams(rules_text())  # verbatim-leak detection

# gpt-oss was trained to cite as 【2†L1-L3】 / 【2】; prompts don't reliably override that, so normalise to [2]
_NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")  # 【2†L1-L3】 or 【2】


def normalize_citations(text: str) -> str:
    return _NATIVE_CITATION.sub(r"[\1]", text)


def build_graph(checkpointer: BaseCheckpointSaver | None = None, llm: BaseChatModel | None = None):
    base_llm = llm or get_llm(reasoning_effort="low")
    model = base_llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    max_steps = get_settings().max_agent_steps

    def agent(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"messages": [AIMessage(MAX_STEPS_REPLY)]}
        customer = state["customer"]
        prompt = SystemMessage(system_prompt(customer["name"], state.get("ticket_number")))
        response = model.invoke([prompt, *state["messages"]])
        if isinstance(response.content, str) and response.content:
            response.content = normalize_citations(response.content)
        return {"messages": [response], "steps": steps + 1}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    def verify(state: AgentState) -> dict:
        if not get_settings().self_check:
            return {}
        return grounding.verify_turn(state["messages"], base_llm) or {}

    def guard(state: AgentState) -> dict:
        """Last check before the answer stands: prompt leaks, other people's emails, external images."""
        last = state["messages"][-1]
        if not get_settings().output_guard or not isinstance(last, AIMessage) or not str(last.content):
            return {}
        result = output_guard.check_output(str(last.content), state["customer"]["email"], PROMPT_NGRAMS)
        if not result.changed:
            return {}
        info = {"reasons": result.reasons, "text": result.text}
        return {"messages": [AIMessage(result.text, id=last.id)], "output_guard": info}

    def after_tools(state: AgentState) -> str:
        return "handoff" if state.get("handed_off") else "agent"

    def handoff(state: AgentState) -> dict:
        return {"messages": [AIMessage(handoff_reply(state["ticket_number"]))]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("handoff", handoff)
    g.add_node("verify", verify)
    g.add_node("guard", guard)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "verify"])
    g.add_edge("verify", "guard")
    g.add_edge("guard", END)
    g.add_conditional_edges("tools", after_tools, ["agent", "handoff"])
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)


def new_turn(message: str, attachments: list | None = None, flagged: bool = False) -> dict:
    """Graph input for a new customer message: resets the per-turn counters.

    With attachments, the LLM sees a description of each file (and the text of logs) appended to the
    message, while `display_text` / `attachments` keep the original words and file names for the UI.
    """
    from langchain_core.messages import HumanMessage

    from supportpilot.attachments import describe_for_llm, meta
    from supportpilot.guardrails.input_guard import AGENT_NOTE

    content = message
    extra: dict = {}
    if attachments:
        content = f"{message}\n\n{describe_for_llm(attachments)}"
        extra = {"display_text": message, "attachments": [meta(a) for a in attachments]}
    if flagged:  # the input guard's note is for the LLM only; the UI keeps the customer's words
        content = f"{content}\n\n{AGENT_NOTE}"
        extra = {"display_text": message, **extra, "guard_flagged": True}
    return {"messages": [HumanMessage(content, additional_kwargs=extra)], "steps": 0, "citations": []}


def display_text(message) -> str:
    """What the customer actually typed (without the attachment block the LLM sees)."""
    return str(message.additional_kwargs.get("display_text", message.content))
