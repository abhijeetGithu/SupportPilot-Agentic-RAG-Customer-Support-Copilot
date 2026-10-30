"""Qdrant collection with a dense (bge-small) and a sparse (BM25) vector per chunk."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models

from supportpilot.config import get_settings
from supportpilot.retrieval import embeddings
from supportpilot.retrieval.text import embedding_text

if TYPE_CHECKING:  # llama-index is only imported when chunks are written (ingestion, KB publishing)
    from llama_index.core.schema import TextNode

DENSE = "dense"
SPARSE = "bm25"


def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False) -> None:
    if client.collection_exists(name):
        if not recreate:
            return
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=embeddings.dense_dim(), distance=models.Distance.COSINE)},
        # BM25 term weights from fastembed; Qdrant applies IDF at query time
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("section", "doc_type", "doc_id", "dup_group"):
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client: QdrantClient, name: str, chunks: list[TextNode], batch_size: int = 64) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [embedding_text(c.metadata, c.text) for c in batch]
        dense = embeddings.embed_passages(texts)
        sparse = embeddings.sparse_passages(texts)
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=point_id(c.id_),
                    vector={
                        DENSE: d,
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload={**c.metadata, "text": c.text},
                )
                for c, d, s in zip(batch, dense, sparse)
            ],
        )
