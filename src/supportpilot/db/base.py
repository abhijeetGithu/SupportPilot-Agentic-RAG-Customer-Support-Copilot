"""Engine and session factory. Postgres in Docker/cloud, SQLite locally and in tests.

The engine is created lazily, so tests can point it at their own database with configure().
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from supportpilot.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_sessionmaker: sessionmaker | None = None


def configure(url: str | None = None) -> Engine:
    """(Re)create the engine. url defaults to DATABASE_URL; 'sqlite://' gives a shared in-memory DB."""
    global _engine, _sessionmaker
    url = url or get_settings().database_url
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one connection, so every session sees the same in-memory DB
    _engine = create_engine(url, **kwargs)
    _sessionmaker = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    return _engine or configure()


def SessionLocal() -> Session:  # factory-style name, mirrors sqlalchemy convention
    if _sessionmaker is None:
        configure()
    return _sessionmaker()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success, roll back on error. For code outside FastAPI (agent tools, scripts)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, committed if the request succeeds."""
    with session_scope() as db:
        yield db


def init_db() -> None:
    from supportpilot.db import models  # noqa: F401  (register tables)

    engine = get_engine()
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """create_all() doesn't alter existing tables, so add new nullable columns to them.
    Small built-in migration for this project's additive changes (no renames or drops)."""
    db_inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not db_inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in db_inspector.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    col_type = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col_type}'))
