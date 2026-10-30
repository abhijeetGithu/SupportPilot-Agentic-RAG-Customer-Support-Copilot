"""Self-check: are the answer's factual claims supported by the passages the agent retrieved?

One call to a separate, smaller model (gpt-oss-20b, its own rate-limit quota) extracts the claims and
checks each against the passages. If any claim is unsupported, the main model rewrites the answer from
the passages only, or says it couldn't find it in the docs and offers a ticket. The customer sees the
original stream replaced by the revision, marked as revised (not silently).
"""

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from supportpilot.config import get_settings

log = logging.getLogger(__name__)
PASSAGES_CHARS = 20000  # ~5K tokens. Was 8000: the eval found answers whose cited passage was cut off
BLOCK = re.compile(r"(?m)^(?=\[\d+\] )")  # search_docs blocks start with "[n] Title"


def select_passages(passages: str, answer: str, limit: int = PASSAGES_CHARS) -> str:
    """Passages the answer cites first, then the rest, within the limit (so cited evidence is never cut)."""
    blocks = [b for b in BLOCK.split(passages) if b.strip()]
    cited = set(re.findall(r"\[(\d+)\]", answer))
    first = [b for b in blocks if re.match(r"\[(\d+)\]", b) and re.match(r"\[(\d+)\]", b).group(1) in cited]
    ordered = first + [b for b in blocks if b not in first]
    out, size = [], 0
    for b in ordered:
        if size + len(b) > limit:
            break
        out.append(b)
        size += len(b)
    return "".join(out) if out else passages[:limit]

CHECK_PROMPT = """You check a customer-support answer against the documentation passages it was based on.

Customer question:
{question}

Documentation passages:
{passages}

Answer to check:
{answer}

List every factual claim the answer makes about SearchUnify: what it does, where settings and buttons are,
steps, limits, option names, causes, timelines. Skip greetings, offers to help or open a ticket, questions
to the customer, and statements that something isn't covered by the documentation.
For each claim decide whether the passages state it or directly imply it. Paraphrase is fine. Details that
appear nowhere in the passages are NOT supported, even if they sound plausible.
Return JSON: {{"claims": [{{"claim": "...", "supported": true, "sources": [1]}}]}}"""

REVISE_PROMPT = """Your draft answer to the customer contained statements the documentation passages don't support:
{unsupported}

Rewrite the answer using only information the passages support, keeping the [n] citations for what you
keep. Drop the unsupported statements. If what remains doesn't answer the question, say briefly that you
couldn't find this in the documentation and offer to open a support ticket. Don't mention this review.

Customer question:
{question}

Documentation passages:
{passages}

Draft answer:
{answer}"""


class Claim(BaseModel):
    claim: str
    supported: bool
    sources: list[int] = Field(default_factory=list)


class CheckResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]


def checker_llm():
    from langchain_groq import ChatGroq

    s = get_settings()
    return ChatGroq(model=s.self_check_model, api_key=s.groq_api_key, temperature=0, max_tokens=2000,
                    max_retries=3, reasoning_effort="low", model_kwargs={"response_format": {"type": "json_object"}})


def check(question: str, passages: str, answer: str, llm=None) -> CheckResult:
    llm = llm or checker_llm()
    out = llm.invoke(CHECK_PROMPT.format(question=question, passages=select_passages(passages, answer), answer=answer))
    return CheckResult.model_validate(json.loads(str(out.content)))


def revise(question: str, passages: str, answer: str, unsupported: list[Claim], llm) -> str:
    bullets = "\n".join(f"- {c.claim}" for c in unsupported)
    out = llm.invoke(REVISE_PROMPT.format(unsupported=bullets, question=question,
                                          passages=select_passages(passages, answer), answer=answer))
    return str(out.content).strip()


def turn_context(messages: list) -> tuple[str, str] | None:
    """(question, passages) for the current turn, or None if the agent didn't search the docs."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    if start < 0:
        return None
    turn = messages[start + 1:]
    passages = [str(m.content) for m in turn if isinstance(m, ToolMessage) and m.name == "search_docs"]
    if not passages:
        return None
    question = str(messages[start].additional_kwargs.get("display_text", messages[start].content))
    return question, "\n\n".join(passages)


def verify_turn(messages: list, reviser_llm, checker=None) -> dict | None:
    """State update for the graph's verify node: grounding info, and a revised answer if needed."""
    answer = messages[-1] if messages else None
    ctx = turn_context(messages)
    if ctx is None or not isinstance(answer, AIMessage) or answer.tool_calls or not str(answer.content).strip():
        return None
    question, passages = ctx
    try:
        result = (checker or check)(question, passages, str(answer.content))
    except Exception:  # the check is a safety net: if it fails, the original answer stands
        log.warning("self-check failed", exc_info=True)
        return {"grounding": {"verdict": "unchecked"}}
    total, bad = len(result.claims), result.unsupported
    info = {"verdict": "no_claims" if total == 0 else "grounded" if not bad else "revised",
            "claims": total, "supported": total - len(bad), "unsupported": [c.claim for c in bad]}
    if not bad:
        return {"grounding": info}
    from supportpilot.agent.graph import normalize_citations

    text = normalize_citations(revise(question, passages, str(answer.content), bad, reviser_llm))
    info["revised_text"] = text
    # same id: add_messages replaces the streamed draft, so history shows the revised answer
    return {"messages": [AIMessage(text, id=answer.id)], "grounding": info}
