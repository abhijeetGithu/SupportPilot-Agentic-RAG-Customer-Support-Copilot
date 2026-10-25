"""Phase 7: Langfuse tracing. Returns a LangChain callback handler, or None if not configured."""

from supportpilot.config import get_settings


def get_langfuse_handler():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse.callback import CallbackHandler

    return CallbackHandler(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
    )
