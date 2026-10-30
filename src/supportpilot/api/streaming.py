"""Turn a LangGraph run into chat events for Server-Sent Events.

Event types (mirrored in frontend/lib/types.ts ChatEvent):
  token {text} · tool_start {tool, query?} · tool_end {tool} · citations {citations}
  ticket_draft {draft} · ticket_created {number, category, priority, duplicate}
  handoff {number} · known_issue {number, notice}
  grounding {verdict, claims, supported, unsupported, revised_text?} · output_guard {reasons, text}
  done {ticket_number} · error {message}

LangGraph is run with stream_mode=["messages", "updates"]: "messages" yields LLM tokens as they are
generated, "updates" yields each node's state changes (tool calls, tool results, interrupts).
"""

import json
import logging
from collections.abc import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

log = logging.getLogger(__name__)

STREAMED_NODES = {"agent"}  # only stream tokens produced by the agent's LLM


class CitationStreamFixer:
    """Rewrites gpt-oss citations (【2†L1-L3】, 【2】) to [2] on a token stream.

    A citation can be split across chunks, so text from an open 【 is held back until its 】 arrives.
    """

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, text: str) -> str:
        from supportpilot.agent.graph import normalize_citations

        self.buf += text
        cut = self.buf.rfind("【")
        if cut != -1 and "】" not in self.buf[cut:]:
            ready, self.buf = self.buf[:cut], self.buf[cut:]
        else:
            ready, self.buf = self.buf, ""
        return normalize_citations(ready)

    def flush(self) -> str:
        from supportpilot.agent.graph import normalize_citations

        out, self.buf = normalize_citations(self.buf), ""
        return out


def stream_events(graph, graph_input, config: dict) -> Iterator[dict]:
    """Run the graph and yield event dicts. Never raises: failures become an `error` event."""
    from supportpilot.agent.graph import PROMPT_NGRAMS
    from supportpilot.guardrails.output_guard import leaks

    fixer = CitationStreamFixer()
    streamed_ids: set[str] = set()
    shown, cut_off = "", False  # text sent this turn; stop streaming once it would leak the system prompt

    def safe(text: str) -> str:
        nonlocal shown, cut_off
        if cut_off or not text:
            return ""
        if leaks(shown + text, PROMPT_NGRAMS):
            cut_off = True  # the guard node replaces the answer; the leak never reaches the browser
            return ""
        shown += text
        return text

    try:
        for mode, chunk in graph.stream(graph_input, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") not in STREAMED_NODES or not isinstance(msg, AIMessageChunk | AIMessage):
                    continue
                if msg.id:
                    streamed_ids.add(msg.id)
                text = msg.content if isinstance(msg.content, str) else ""
                if text and not getattr(msg, "tool_call_chunks", None):
                    out = safe(fixer.feed(text))
                    if out:
                        yield {"type": "token", "text": out}
                continue

            # mode == "updates": {node_name: state_update} or {"__interrupt__": (Interrupt, ...)}
            for node, update in chunk.items():
                if node == "__interrupt__":
                    for intr in update:
                        if isinstance(intr.value, dict) and intr.value.get("type") == "ticket_draft":
                            yield {"type": "ticket_draft", "draft": intr.value["draft"]}
                    continue
                if not isinstance(update, dict):
                    continue
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        tail = safe(fixer.flush())
                        if tail:
                            yield {"type": "token", "text": tail}
                        for call in m.tool_calls:
                            event = {"type": "tool_start", "tool": call["name"]}
                            if call["name"] == "search_docs":
                                event["query"] = call["args"].get("query")
                            yield event
                    elif isinstance(m, AIMessage) and m.content and (not m.id or m.id not in streamed_ids):
                        # messages written by nodes without an LLM call (handoff, step-limit reply)
                        yield {"type": "token", "text": str(m.content)}
                    elif isinstance(m, ToolMessage):
                        yield {"type": "tool_end", "tool": m.name}
                        art = m.artifact or {}
                        if m.name == "search_docs" and art.get("known_issue"):
                            yield {"type": "known_issue", **art["known_issue"]}
                        if m.name == "create_ticket" and art.get("number"):
                            yield {"type": "ticket_created", **art}
                        if m.name == "escalate_to_human" and art.get("number"):
                            yield {"type": "handoff", "number": art["number"]}
                if update.get("output_guard"):
                    yield {"type": "output_guard", **update["output_guard"]}
                if update.get("grounding"):
                    yield {"type": "grounding", **update["grounding"]}
                if update.get("citations"):
                    yield {"type": "citations", "citations": update["citations"]}

        tail = safe(fixer.flush())
        if tail:
            yield {"type": "token", "text": tail}
        state = graph.get_state(config)
        yield {"type": "done", "ticket_number": state.values.get("ticket_number")}
    except Exception as e:  # surface to the client instead of breaking the stream
        log.exception("chat stream failed")
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"[:300]}


def to_sse(events: Iterator[dict]) -> Iterator[dict]:
    """sse-starlette format: the event name plus the JSON payload (which also carries `type`)."""
    for e in events:
        yield {"event": e["type"], "data": json.dumps(e, ensure_ascii=False)}
