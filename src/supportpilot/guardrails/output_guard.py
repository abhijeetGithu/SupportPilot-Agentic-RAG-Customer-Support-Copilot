"""Phase 6: output checks before returning to the user."""

from supportpilot.guardrails.input_guard import GuardResult


def check_output(answer: str, context: list[str]) -> GuardResult:
    """Redact PII; flag doc-based answers that cite nothing."""
    raise NotImplementedError("Phase 6")
