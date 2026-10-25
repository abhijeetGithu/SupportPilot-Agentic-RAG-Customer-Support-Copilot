"""python scripts/ingest.py [--source data/raw/docs] [--recreate] [--collection NAME] [--chunk-size N]"""

import argparse
import json
from pathlib import Path

from supportpilot.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/docs"))
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--collection", help="defaults to QDRANT_COLLECTION")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    args = parser.parse_args()
    stats = run_ingestion(
        args.source,
        recreate=args.recreate,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        # experiment collections don't overwrite the main chunks file
        chunks_file=None if args.collection else Path("data/processed/chunks.jsonl"),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
