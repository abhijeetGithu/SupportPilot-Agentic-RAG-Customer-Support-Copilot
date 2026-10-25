"""AI-drafted reply for staff (human in the loop: staff edit and send it; nothing is sent automatically)."""

from supportpilot.agent.graph import normalize_citations
from supportpilot.db.models import Ticket
from supportpilot.llm.provider import get_llm
from supportpilot.retrieval.retriever import search

PROMPT = """You are drafting a reply for a SearchUnify support engineer to send to a customer.

Customer: {name}
Ticket {number}: {subject}
Customer's description:
{description}

Latest messages:
{conversation}

Documentation passages:
{passages}

Write the reply the engineer will send:
- Use only the passages for product facts and cite them as [n]. If they don't cover the issue,
  ask for the specific details needed (error text, screenshots, affected content source) instead of guessing.
- Friendly and concise: a short acknowledgement, numbered steps if relevant, no sign-off placeholder text.
- Greet the customer by first name and write in the second person. Don't mention that you are an AI
  or that passages were provided."""


def suggest_reply(ticket: Ticket, llm=None, searcher=search) -> dict:
    hits = searcher(ticket.subject, expansions=[ticket.description[:500]])
    citations = [
        {"n": i, "title": h.title or h.doc_id, "url": h.url, "heading": h.heading} for i, h in enumerate(hits, 1)
    ]
    passages = "\n\n".join(f"[{i}] {h.title}\n{h.text[:1000]}" for i, h in enumerate(hits, 1)) or "(none found)"
    conversation = "\n".join(f"{m.author_type}: {m.body[:400]}" for m in ticket.messages[-6:]) or "(none)"
    first_name = ticket.customer_name.split()[0] if ticket.customer_name.strip() else "there"
    llm = llm or get_llm(reasoning_effort="low", max_tokens=1200)
    out = llm.invoke(
        PROMPT.format(name=first_name, number=ticket.number, subject=ticket.subject,
                      description=ticket.description[:2000], conversation=conversation, passages=passages)
    )
    body = normalize_citations(str(out.content)).strip()
    cited = [c for c in citations if f"[{c['n']}]" in body]
    return {"body": body, "citations": cited}
