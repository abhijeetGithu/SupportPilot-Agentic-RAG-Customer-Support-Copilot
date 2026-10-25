import pytest

from supportpilot.db import base


@pytest.fixture
def db():
    """Fresh in-memory SQLite database per test."""
    base.configure("sqlite://")
    base.init_db()
    session = base.SessionLocal()
    yield session
    session.close()
    base.Base.metadata.drop_all(base.get_engine())

