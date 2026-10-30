"""Central settings loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    groq_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None  # Qdrant Cloud
    embedding_cache_dir: str | None = None  # where fastembed keeps its ONNX models (baked into the Docker image)
    qdrant_collection: str = "support_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    hybrid_sparse_weight: float = 1.0  # BM25 weight in RRF (dense = 1.0)
    rrf_k: int = 2  # small k favours items ranked at the very top of either list (swept 2/10/20/60)
    use_reranker: bool = False  # evaluated in Phase 2: no Recall@5 gain, ~2 s/query on CPU
    query_rewrite: bool = True  # Phase 2: +13 pts Recall@5 on vocabulary-mismatch queries, ~1 s LLM call
    release_notes_weight: float = 1.0  # <1 down-weights release notes unless the query asks about releases
    # tiktoken tokens; bge-small truncates at 512 wordpieces incl. the contextual header
    chunk_size: int = 384
    chunk_overlap: int = 48

    # Database: Postgres in Docker/cloud, SQLite for quick local runs and tests
    database_url: str = "sqlite:///data/processed/supportpilot.db"
    sqlite_checkpoints_path: str = "data/processed/checkpoints.db"  # agent memory when not on Postgres

    # Agent (checkpointer reuses database_url: Postgres saver or SQLite saver)
    max_agent_steps: int = 8

    # Tickets
    duplicate_window_hours: int = 24
    duplicate_similarity: float = 0.85
    # calibrated confidence below these -> needs_review (chosen in Phase 3: ~18% of tickets flagged)
    review_confidence_threshold: float = 0.40  # category
    review_priority_threshold: float = 0.50

    # Attachments (screenshots, logs) on chat messages and tickets
    uploads_dir: str = "data/uploads"
    max_upload_mb: int = 10
    max_attachments: int = 5  # per message / ticket
    # Screenshots and PDFs are read by a vision model at upload time (Phase 7.5)
    analyze_attachments: bool = True
    vision_model: str = "qwen/qwen3.8-27b"  # Groq, image input; ~1.3K tokens and ~1 s per screenshot
    vision_max_side: int = 1600  # px; larger images are downscaled before sending
    pdf_max_pages: int = 5  # text layer read from the first pages; scanned PDFs: first 2 pages as images

    # Knowledge loop (Phase 7.5): resolved tickets -> KB drafts -> staff review -> indexed with the docs
    public_app_url: str = "http://localhost:3000"  # KB citations link to {public_app_url}/support/kb/KB-00001
    auto_draft_kb: bool = True  # draft an article in the background when a ticket is resolved
    # best-chunk cosine below this = likely not covered. Calibrated: flags 1% of standard and 10% of paraphrased
    # answerable golden questions, catches 16/20 off-topic ones (docs/results/phase7_5b_knowledge_loop.md)
    gap_coverage_threshold: float = 0.66

    # Guardrails (Phase 8). Prompt Guard flags (never blocks): it scored normal messages like "ignore my
    # last message, I meant Confluence" at 0.997. See docs/results/phase8_trust.md
    input_guard: bool = True
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    injection_threshold: float = 0.9
    output_guard: bool = True

    # Self-check (Phase 8): claims in doc-based answers are verified by a second model before they stand
    self_check: bool = True
    self_check_model: str = "openai/gpt-oss-20b"  # separate Groq quota from the chat model

    # Incident detection (Phase 7.5c): a burst of similar tickets from different customers
    detect_incidents: bool = True
    # pairs: 0.74 links 81% of same-outage pairs, 3% of near-misses; replaying traffic, 0.74-0.78 all work and
    # 0.76 is mid-range (19/20 outages, 0 false, 0.95 member precision). See docs/results/phase7_5c_incidents.md
    incident_similarity: float = 0.76
    incident_window_minutes: int = 120
    incident_min_reports: int = 3  # from distinct customers
    incident_baseline_days: int = 7  # normal rate of similar tickets, for the burst test
    incident_max_p_value: float = 0.01  # Poisson: this many similar tickets this fast would be very unusual

    # Classifier
    classifier_keep_loaded: bool = True  # false on 512 MB hosts: load per prediction, then release
    classifier_onnx_dir: str = "models/onnx"  # int8 ONNX export (production): scripts/export_onnx.py
    classifier_model_dir: str = "models/distilbert"  # from the Colab notebook; TF-IDF fallback
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"  # artifacts go to ./mlartifacts

    # API / auth
    environment: Literal["development", "production"] = "development"  # production refuses default secrets
    cors_origins: list[str] = ["http://localhost:3000"]
    jwt_secret: str = "change-me"
    jwt_expire_minutes: int = 480
    staff_username: str = "admin"
    staff_password: str = "admin"
    # optional read-only account for the public demo (recruiters can explore the dashboard, not change it)
    demo_username: str = "demo"
    demo_password: str = ""
    # public demo protection: per-IP limits and a daily cap on assistant turns (Groq free tier)
    rate_limits: bool = True
    daily_chat_limit: int = 300
    warmup_models: bool = True  # load DistilBERT + embeddings at API startup (first call otherwise ~30 s)

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


def check_production(s: "Settings") -> list[str]:
    """Problems that must be fixed before running with ENVIRONMENT=production."""
    problems = []
    if s.jwt_secret in ("change-me", "") or len(s.jwt_secret) < 32:
        problems.append("JWT_SECRET must be a random string of at least 32 characters")
    if s.staff_password in ("admin", "") or len(s.staff_password) < 12:
        problems.append("STAFF_PASSWORD must be set (at least 12 characters, not 'admin')")
    if not s.groq_api_key:
        problems.append("GROQ_API_KEY is missing")
    if s.database_url.startswith("sqlite"):
        problems.append("DATABASE_URL should point to Postgres (the host's disk is wiped on restart)")
    return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
