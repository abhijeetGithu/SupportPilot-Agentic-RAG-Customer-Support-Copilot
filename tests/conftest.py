import pytest

from supportpilot.db import base


@pytest.fixture(autouse=True)
def no_vision_calls(monkeypatch):
    """Tests never call the vision API; uploads behave as if analysis failed unless a test fakes it."""
    from supportpilot import file_insights

    def offline(*a, **kw):
        raise RuntimeError("vision API disabled in tests")

    monkeypatch.setattr(file_insights, "call_vision", offline)


@pytest.fixture(autouse=True)
def no_knowledge_calls(monkeypatch):
    """No LLM drafting or Qdrant coverage lookups in tests unless a test fakes them."""
    from supportpilot.knowledge import drafting, gaps

    def offline(*a, **kw):
        raise RuntimeError("KB drafting LLM disabled in tests")

    monkeypatch.setattr(drafting, "generate", offline)
    monkeypatch.setattr(gaps, "doc_coverage", lambda text: None)


@pytest.fixture
def db():
    """Fresh in-memory SQLite database per test."""
    base.configure("sqlite://")
    base.init_db()
    session = base.SessionLocal()
    yield session
    session.close()
    base.Base.metadata.drop_all(base.get_engine())



@pytest.fixture(autouse=True)
def no_incident_detection(monkeypatch):
    """Incident detection embeds every ticket with bge; off unless a test turns it on (tests/test_incidents.py)."""
    from supportpilot.incidents import detector

    monkeypatch.setattr(detector, "on_ticket_created", lambda db, ticket, now=None: None)
    monkeypatch.setattr(detector, "match_active", lambda db, text, threshold=None: None)
