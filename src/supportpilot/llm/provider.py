"""LLM factory: switch provider with the LLM_PROVIDER env var."""

from langchain_core.language_models.chat_models import BaseChatModel

from supportpilot.config import get_settings


def get_llm(max_tokens: int | None = None, reasoning_effort: str | None = None) -> BaseChatModel:
    """reasoning_effort ("low" | "medium" | "high") only applies to reasoning models on Groq."""
    s = get_settings()
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        return ChatGroq(
            model=s.llm_model,
            temperature=s.llm_temperature,
            api_key=s.groq_api_key,
            max_tokens=max_tokens,
            max_retries=6,
            **extra,
        )
    if s.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=s.llm_model,
            temperature=s.llm_temperature,
            google_api_key=s.google_api_key,
            max_output_tokens=max_tokens,
        )
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.llm_model,
            temperature=s.llm_temperature,
            base_url=s.ollama_base_url,
            num_predict=max_tokens,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
