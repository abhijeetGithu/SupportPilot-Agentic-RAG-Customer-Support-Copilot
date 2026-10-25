"""Dense (bge-small) and sparse (BM25) encoders via fastembed. CPU-only, ONNX, no torch."""

from functools import lru_cache

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

from supportpilot.config import get_settings


@lru_cache
def dense_model() -> TextEmbedding:
    return TextEmbedding(get_settings().embedding_model)


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(get_settings().sparse_model)


def dense_dim() -> int:
    return len(next(dense_model().query_embed("dim")))


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return [v.tolist() for v in dense_model().passage_embed(texts, batch_size=batch_size)]


def embed_query(query: str) -> list[float]:
    return next(dense_model().query_embed(query)).tolist()


def sparse_passages(texts: list[str], batch_size: int = 64) -> list[SparseEmbedding]:
    return list(sparse_model().passage_embed(texts, batch_size=batch_size))


def sparse_query(query: str) -> SparseEmbedding:
    return next(sparse_model().query_embed(query))
