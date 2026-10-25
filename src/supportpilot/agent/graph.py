"""LangGraph agent.

    START -> agent --(tool calls)--> tools --> agent -> ... -> END
                   \\--(no tool calls)--> END
             tools --(escalated)--> handoff -> END

- One tool call per LLM turn (parallel_tool_calls=False): create_ticket pauses the graph with interrupt(),
  and LangGraph re-runs the whole tool step on resume, so parallel calls would repeat their side effects.
- Step limit: after MAX_AGENT_STEPS LLM calls in one turn the agent stops and offers a ticket.
- Memory: the checkpointer stores the full state per thread_id (= chat session id).
- Input/output guardrails are added in Phase 8.
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from supportpilot.agent.prompts import MAX_STEPS_REPLY, handoff_reply, system_prompt
from supportpilot.agent.state import AgentState
from supportpilot.agent.tools import ALL_TOOLS
from supportpilot.config import get_settings
from supportpilot.llm.provider import get_llm

# gpt-oss was trained to cite as 【2†L1-L3】 / 【2】; prompts don't reliably override that, so normalise to [2]
_NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")  # 【2†L1-L3】 or 【2】


def normalize_citations(text: str) -> str:
    return _NATIVE_CITATION.sub(r"[\1]", text)


def build_graph(checkpointer: BaseCheckpointSaver | None = None, llm: BaseChatModel | None = None):
    model = (llm or get_llm(reasoning_effort="low")).bind_tools(ALL_TOOLS, parallel_tool_calls=False)
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
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def after_tools(state: AgentState) -> str:
        return "handoff" if state.get("handed_off") else "agent"

    def handoff(state: AgentState) -> dict:
        return {"messages": [AIMessage(handoff_reply(state["ticket_number"]))]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("handoff", handoff)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["agent", "handoff"])
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)


def new_turn(message: str) -> dict:
    """Graph input for a new customer message: resets the per-turn counters."""
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(message)], "steps": 0, "citations": []}
