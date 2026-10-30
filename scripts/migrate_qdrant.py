"""Copy the search index (vectors + payloads) from one Qdrant to another, e.g. local -> Qdrant Cloud.
No re-embedding: points are copied as they are. KB articles are skipped by default (they belong to the
database they were published from; publish them again on the new deployment).

    python scripts/migrate_qdrant.py --dst https://xyz.cloud.qdrant.io:6333 --dst-key <api key>
"""

import argparse

from qdrant_client import QdrantClient, models

from supportpilot.retrieval.vector_store import DENSE, SPARSE

PAYLOAD_INDEXES = ("section", "doc_type", "doc_id", "dup_group")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="http://localhost:6333")
    ap.add_argument("--src-key", default=None)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dst-key", required=True)
    ap.add_argument("--collection", default="support_docs")
    ap.add_argument("--dst-collection", default=None, help="defaults to --collection")
    ap.add_argument("--include-kb", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    target = a.dst_collection or a.collection
    if a.src.rstrip("/") == a.dst.rstrip("/") and target == a.collection:
        raise SystemExit("source and destination are the same collection: refusing to delete it")
    src = QdrantClient(url=a.src, api_key=a.src_key, timeout=120)
    dst = QdrantClient(url=a.dst, api_key=a.dst_key, timeout=120)
    info = src.get_collection(a.collection)
    if dst.collection_exists(target):
        dst.delete_collection(target)
    dst.create_collection(
        target,
        vectors_config={DENSE: models.VectorParams(size=info.config.params.vectors[DENSE].size,
                                                   distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in PAYLOAD_INDEXES:
        dst.create_payload_index(target, field, models.PayloadSchemaType.KEYWORD)

    skip_kb = None if a.include_kb else models.Filter(must_not=[
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="kb"))])
    copied, offset = 0, None
    while True:
        points, offset = src.scroll(a.collection, scroll_filter=skip_kb, limit=a.batch, offset=offset,
                                    with_payload=True, with_vectors=True)
        if not points:
            break
        dst.upsert(target, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                                         for p in points])
        copied += len(points)
        print(f"copied {copied}")
        if offset is None:
            break
    print(f"done: {copied} points; destination has {dst.count(target).count}")


if __name__ == "__main__":
    main()
