"""Qdrant collection with a dense (bge-small) and a sparse (BM25) vector per chunk."""

import uuid

from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient, models

from supportpilot.config import get_settings
from supportpilot.ingestion.chunking import embedding_text
from supportpilot.retrieval import embeddings

DENSE = "dense"
SPARSE = "bm25"


def get_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=60)


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
