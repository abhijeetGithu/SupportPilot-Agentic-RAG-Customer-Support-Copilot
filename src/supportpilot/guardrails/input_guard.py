"""Phase 6: input checks before the LLM sees user text."""

from pydantic import BaseModel


class GuardResult(BaseModel):
    allowed: bool
    text: str  # possibly PII-masked
    reasons: list[str] = []


def check_input(text: str) -> GuardResult:
    """Prompt-injection heuristics + LLM classifier; mask emails/phones/card numbers."""
    raise NotImplementedError("Phase 6")
