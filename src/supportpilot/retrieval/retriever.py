"""Retrieval strategies. Each step can be switched on/off so its effect is measurable.

- dense_search:  bge-small cosine similarity (Phase 1 baseline)
- sparse_search: BM25 keyword match
- hybrid_search: dense + BM25 candidates fused with Reciprocal Rank Fusion in Qdrant
- search:        production path = hybrid -> collapse duplicate boilerplate -> cross-encoder rerank
"""

import re
from typing import Literal

from qdrant_client import models

from supportpilot.config import get_settings
from supportpilot.retrieval import embeddings
from supportpilot.retrieval.reranker import rerank
from supportpilot.retrieval.schemas import RetrievedChunk
from supportpilot.retrieval.vector_store import DENSE, SPARSE, get_client

PREFETCH_K = 40  # candidates per retriever before fusion
# release notes mention every feature briefly, so they outrank the real guide on vague queries
RELEASE_QUERY = re.compile(r"\b(release|version|new in|what'?s new|changelog|q[1-4]|20\d\d)\b", re.IGNORECASE)


def _to_chunks(points: list[models.ScoredPoint]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=p.payload["chunk_id"],
            doc_id=p.payload["doc_id"],
            text=p.payload["text"],
            score=p.score,
            url=p.payload["url"],
            title=" ".join((p.payload.get("title") or "").split()) or None,
            section=p.payload.get("section"),
            doc_type=p.payload.get("doc_type"),
            heading=p.payload.get("heading"),
            dup_group=p.payload.get("dup_group"),
        )
        for p in points
    ]


def _sparse_vector(query: str) -> models.SparseVector:
    s = embeddings.sparse_query(query)
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def dense_search(
    query: str, top_k: int, collection: str | None = None, query_filter: models.Filter | None = None
) -> list[RetrievedChunk]:
    result = get_client().query_points(
        collection or get_settings().qdrant_collection,
        query=embeddings.embed_query(query),
        using=DENSE,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return _to_chunks(result.points)


def sparse_search(
    query: str, top_k: int, collection: str | None = None, query_filter: models.Filter | None = None
) -> list[RetrievedChunk]:
    result = get_client().query_points(
        collection or get_settings().qdrant_collection,
        query=_sparse_vector(query),
        using=SPARSE,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return _to_chunks(result.points)


def rrf_fuse(rankings: list[tuple[list[RetrievedChunk], float]], k: int) -> list[RetrievedChunk]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum over lists of weight / (k + rank)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for hits, weight in rankings:
        for rank, h in enumerate(hits, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + weight / (k + rank)
            first_seen.setdefault(h.id, h)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    return [first_seen[i].model_copy(update={"score": scores[i]}) for i in order]


def hybrid_search(
    query: str,
    top_k: int,
    collection: str | None = None,
    query_filter: models.Filter | None = None,
    sparse_weight: float | None = None,
    expansions: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Dense (+ BM25 if sparse_weight > 0) for the query and each expansion, fused with weighted RRF.

    All rankings come from one batched Qdrant request.
    """
    s = get_settings()
    w_sparse = s.hybrid_sparse_weight if sparse_weight is None else sparse_weight
    requests, weights = [], []
    for q in [query, *(expansions or [])]:
        requests.append(
            models.QueryRequest(
                query=embeddings.embed_query(q), using=DENSE, limit=PREFETCH_K, filter=query_filter, with_payload=True
            )
        )
        weights.append(1.0)
        if w_sparse > 0:
            requests.append(
                models.QueryRequest(
                    query=_sparse_vector(q), using=SPARSE, limit=PREFETCH_K, filter=query_filter, with_payload=True
                )
            )
            weights.append(w_sparse)
    responses = get_client().query_batch_points(collection or s.qdrant_collection, requests=requests)
    rankings = [(_to_chunks(r.points), w) for r, w in zip(responses, weights)]
    fused = rrf_fuse(rankings, s.rrf_k)
    if s.release_notes_weight != 1.0 and not RELEASE_QUERY.search(query):
        fused = sorted(
            (c.model_copy(update={"score": c.score * s.release_notes_weight}) if c.doc_type == "release_notes" else c
             for c in fused),
            key=lambda c: c.score,
            reverse=True,
        )
    return fused[:top_k]


def collapse_duplicates(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only the best-ranked copy of each near-duplicate group."""
    seen: set[str] = set()
    out = []
    for c in chunks:
        key = c.dup_group or c.id
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def search(
    query: str,
    top_n: int | None = None,
    collection: str | None = None,
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    dedupe: bool = True,
    use_reranker: bool | None = None,
    reranker_model: str | None = None,
    query_filter: models.Filter | None = None,
    sparse_weight: float | None = None,
    expansions: list[str] | None = None,
) -> list[RetrievedChunk]:
    """expansions: extra phrasings of the query (e.g. from query_rewrite), fused with the original."""
    s = get_settings()
    top_n = top_n or s.rerank_top_n
    use_reranker = s.use_reranker if use_reranker is None else use_reranker
    # fetch extra so there are enough left after collapsing duplicates
    k = max(s.retrieval_top_k, top_n) + 10
    if mode == "sparse":
        candidates = sparse_search(query, k, collection, query_filter)
    elif mode == "dense" and not expansions:
        candidates = dense_search(query, k, collection, query_filter)
    else:
        w = 0.0 if mode == "dense" else sparse_weight
        candidates = hybrid_search(query, k, collection, query_filter, sparse_weight=w, expansions=expansions)
    if dedupe:
        candidates = collapse_duplicates(candidates)
    if use_reranker:
        return rerank(query, candidates[: s.retrieval_top_k], top_n, reranker_model)
    return candidates[:top_n]
