"""Persistent agent memory, chosen from DATABASE_URL: Postgres in Docker/cloud, SQLite locally.

Conversation state (messages, pending ticket drafts, ticket number) survives API restarts, so a
customer can approve a ticket draft even after a redeploy.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from supportpilot.config import get_settings

SQLITE_CHECKPOINTS = Path("data/processed/checkpoints.db")


def make_checkpointer(database_url: str | None = None) -> BaseCheckpointSaver:
    url = database_url or get_settings().database_url
    if url.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        conninfo = url.replace("postgresql+psycopg://", "postgresql://")
        pool = ConnectionPool(conninfo, max_size=10, kwargs={"autocommit": True, "row_factory": dict_row})
        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables if missing
        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    SQLITE_CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(SQLITE_CHECKPOINTS, check_same_thread=False))
