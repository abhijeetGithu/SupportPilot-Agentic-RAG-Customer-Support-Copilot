"""Draft a knowledge-base article from a resolved ticket (staff review it before it is published).

Grounding: the LLM sees only the ticket conversation and may only use facts stated there. It also
says whether the fix is useful beyond this customer (account-specific refunds aren't).
Privacy: input is already PII-masked; the output is checked again and the customer's name, email and
email domain are removed, so a published article can't identify who reported the problem.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from supportpilot.db.models import AuthorType, KbArticle, KbStatus, Ticket
from supportpilot.llm.provider import get_llm
from supportpilot.tickets.service import mask_pii

PROMPT = """You turn a resolved SearchUnify support ticket into a short knowledge-base article that will
help other customers with the same problem.

Rules:
- Use ONLY facts stated in the ticket conversation below. Do not add steps, settings or causes that
  the support team did not state. If the cause wasn't stated, write "Not stated."
- Write for any customer: no names, emails, company names, ticket numbers or dates.
- Steps as a numbered list, using the exact UI labels and error messages from the conversation.

Return a JSON object with these keys:
- "title": a search-friendly title naming the symptom, e.g. "Jira crawl fails with HTTP 401 after token rotation"
- "problem": 1-2 sentences describing what the customer sees
- "symptoms": list of exact error messages or observable symptoms (may be empty)
- "cause": the cause the support team identified, or "Not stated."
- "resolution": markdown numbered steps that fixed it
- "product_area": e.g. "Content Sources > Jira", or ""
- "keywords": 3-8 search keywords and synonyms a customer might use
- "fixed_by": who performed the fix. "customer" if the customer applied it by following instructions
  (changing a setting, re-running a crawl, editing a configuration). "support_team" if the support team
  did it on their side for this account (issued a refund or credit, corrected billing, changed a
  contract or licence, fixed data in the backend).
- "note": one sentence explaining the fixed_by decision

Ticket subject: {subject}
Category: {category}

Conversation (oldest first):
{conversation}"""


class Draft(BaseModel):
    title: str = Field(min_length=5, max_length=255)
    problem: str
    symptoms: list[str] = Field(default_factory=list)
    cause: str = "Not stated."
    resolution: str
    product_area: str = ""
    keywords: list[str] = Field(default_factory=list)
    fixed_by: Literal["customer", "support_team"] = "customer"
    note: str = ""

    @property
    def generalizable(self) -> bool:
        """Only fixes customers can apply themselves make useful articles. Asking who did the fix (a fact
        in the conversation) proved more reliable than asking the LLM whether the case 'generalizes'."""
        return self.fixed_by == "customer"


def conversation_text(ticket: Ticket, max_chars: int = 6000) -> str:
    lines = [f"customer (description): {ticket.description}"]
    for m in ticket.messages:
        if m.author_type == AuthorType.AI.value and len(lines) > 1:
            continue  # the chat transcript's AI turns add little; staff replies carry the fix
        who = {"staff": "support team", "customer": "customer", "ai": "assistant"}.get(m.author_type, m.author_type)
        lines.append(f"{who}: {m.body}")
    text = "\n\n".join(lines)
    return text[-max_chars:]  # keep the end: that's where the resolution is


def has_resolution(ticket: Ticket) -> bool:
    return any(m.author_type == AuthorType.STAFF.value for m in ticket.messages)


def scrub(text: str, ticket: Ticket) -> str:
    """Remove anything identifying the reporting customer, after the usual PII masking."""
    text = mask_pii(text)
    email = ticket.customer_email.lower()
    domain = email.split("@")[-1]
    company = domain.split(".")[0]
    names = [n for n in ticket.customer_name.split() if len(n) > 2]
    for term in [email, domain, *names] + ([company] if len(company) > 3 else []):
        text = re.sub(rf"\b{re.escape(term)}\b", "[customer]" if term in names else "[redacted]", text,
                      flags=re.IGNORECASE)
    return re.sub(r"\bSP-?\d{3,9}\b", "[ticket]", text, flags=re.IGNORECASE)


def to_markdown(d: Draft) -> str:
    parts = [f"## Problem\n{d.problem}"]
    if d.symptoms:
        parts.append("## Symptoms\n" + "\n".join(f"- `{s}`" if len(s) < 160 else f"- {s}" for s in d.symptoms))
    parts.append(f"## Cause\n{d.cause}")
    parts.append(f"## Resolution\n{d.resolution}")
    return "\n\n".join(parts)


def generate(ticket: Ticket, llm=None) -> tuple[Draft, str]:
    llm = llm or get_llm(reasoning_effort="low", max_tokens=1500)
    out = llm.bind(response_format={"type": "json_object"}).invoke(
        PROMPT.format(subject=ticket.subject, category=ticket.category or "unknown",
                      conversation=conversation_text(ticket))
    )
    return Draft.model_validate(json.loads(str(out.content))), getattr(llm, "model_name", None) or ""


def draft_article(db, ticket: Ticket, llm=None, coverage_fn=None) -> KbArticle:
    """Create a draft KbArticle for a resolved ticket (not committed)."""
    from supportpilot.knowledge.gaps import doc_coverage

    draft, model = generate(ticket, llm)
    for field in ("title", "problem", "cause", "resolution", "product_area"):
        setattr(draft, field, scrub(getattr(draft, field), ticket))
    draft.symptoms = [scrub(s, ticket) for s in draft.symptoms]
    draft.keywords = [scrub(k, ticket) for k in draft.keywords][:8]

    coverage = (coverage_fn or doc_coverage)(f"{draft.title}\n{draft.problem}")
    article = KbArticle(
        source_ticket_id=ticket.id, title=draft.title[:255], body=to_markdown(draft),
        product_area=draft.product_area[:120], keywords=draft.keywords, status=KbStatus.DRAFT.value,
        doc_coverage=coverage, generalizable=draft.generalizable, draft_note=draft.note, model=model,
    )
    db.add(article)
    db.flush()
    return article
