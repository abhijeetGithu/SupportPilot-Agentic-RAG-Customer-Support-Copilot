"""load -> clean -> chunk -> embed (dense + BM25) -> upsert into Qdrant."""

import json
from pathlib import Path

from supportpilot.config import get_settings
from supportpilot.ingestion.chunking import chunk_documents
from supportpilot.ingestion.dedup import assign_dup_groups
from supportpilot.ingestion.loaders import load_documents
from supportpilot.retrieval.vector_store import ensure_collection, get_client, upsert_chunks

CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def run_ingestion(
    source_dir: Path,
    recreate: bool = False,
    collection: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunks_file: Path | None = CHUNKS_FILE,
) -> dict:
    """Index all docs. Returns stats. Writes the chunks to chunks_file for eval/debugging."""
    s = get_settings()
    collection = collection or s.qdrant_collection
    chunk_size = chunk_size or s.chunk_size
    chunk_overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap

    docs, stats = load_documents(source_dir)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    stats["chunks_in_dup_groups"] = assign_dup_groups(chunks)

    if chunks_file:
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with chunks_file.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({**c.metadata, "text": c.text}, ensure_ascii=False) + "\n")

    client = get_client()
    ensure_collection(client, collection, recreate=recreate)
    upsert_chunks(client, collection, chunks)

    tokens = [c.metadata["tokens"] for c in chunks]
    return {
        **stats,
        "chunks": len(chunks),
        "avg_chunk_tokens": round(sum(tokens) / len(tokens)),
        "max_chunk_tokens": max(tokens),
        "collection": collection,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
