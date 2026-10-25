"""Cross-encoder reranking (fastembed TextCrossEncoder, ONNX on CPU).

A cross-encoder reads the query and passage together, so it judges relevance much better
than comparing two separate embeddings, but it is too slow to run over the whole index.
It is only used on the ~20 candidates that hybrid search returns.
"""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from supportpilot.config import get_settings
from supportpilot.ingestion.chunking import embedding_text
from supportpilot.retrieval.schemas import RetrievedChunk


@lru_cache
def get_reranker(model_name: str) -> TextCrossEncoder:
    return TextCrossEncoder(model_name)


def rerank(
    query: str, chunks: list[RetrievedChunk], top_n: int, model_name: str | None = None
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    model = get_reranker(model_name or get_settings().reranker_model)
    docs = [
        embedding_text({"title": c.title or "", "section": c.section or "", "heading": c.heading}, c.text)
        for c in chunks
    ]
    scores = list(model.rerank(query, docs))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [c.model_copy(update={"score": float(s)}) for c, s in ranked]
