"""FastAPI app. All business routes live under /api (proxied by Next.js)."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from supportpilot import __version__
from supportpilot.api import auth
from supportpilot.api.routes import admin, chat, incidents, knowledge, tickets, uploads
from supportpilot.config import get_settings
from supportpilot.db.base import init_db

log = logging.getLogger("supportpilot")


def warm_up() -> None:
    """Load DistilBERT and the embedding models now, instead of on the first customer request (~30 s)."""
    from supportpilot.classifier.predict import classify_ticket
    from supportpilot.retrieval import embeddings

    t0 = time.perf_counter()
    classify_ticket("warm-up ticket: the search page shows no results")
    embeddings.embed_query("warm up")
    embeddings.sparse_query("warm up")
    log.info("models warmed up in %.1fs", time.perf_counter() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if getattr(app.state, "graph", None) is None:  # tests inject their own graph
        from supportpilot.agent.checkpointer import make_checkpointer
        from supportpilot.agent.graph import build_graph

        app.state.graph = build_graph(make_checkpointer())
    if get_settings().warmup_models:
        warm_up()
    yield


app = FastAPI(title="SupportPilot", version=__version__, lifespan=lifespan)
app.state.graph = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUTERS = (chat.router, tickets.router, admin.router, auth.router, uploads.router, knowledge.admin, knowledge.public,
           incidents.router)
for r in ROUTERS:
    app.include_router(r, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}
