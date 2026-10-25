"""Retrieval metrics (Recall@K, MRR) per strategy, with per-section breakdown and failures.

    python evals/run_retrieval_eval.py --strategy hybrid_dedup_rerank [--golden PATH]
        [--collection NAME] [--reranker MODEL] [--tag NAME] [--fail-under-recall 0.8]

Strategies: dense, bm25, hybrid, hybrid_dedup, dense_dedup_rerank, hybrid_dedup_rerank

Golden set rows (evals/datasets/golden.jsonl):
  {"id", "question", "expected_doc_ids": [...], "source_chunk_id", "section", "type", ...}

Writes evals/reports/retrieval_<tag>.json and a failures file for error analysis.
Exits 1 if Recall@5 is below --fail-under-recall (used as the CI gate).
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from supportpilot.config import get_settings
from supportpilot.evaluation.metrics import first_hit_rank, mean, recall_at_k, reciprocal_rank
from supportpilot.retrieval import retriever

K_VALUES = (1, 3, 5, 10)
REPORTS = Path("evals/reports")


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# name -> retriever.search() options; each adds one component, so gains are attributable
STRATEGIES = {
    "dense": {"mode": "dense", "dedupe": False, "use_reranker": False},
    "bm25": {"mode": "sparse", "dedupe": False, "use_reranker": False},
    "hybrid": {"mode": "hybrid", "dedupe": False, "use_reranker": False},
    "hybrid_dedup": {"mode": "hybrid", "dedupe": True, "use_reranker": False},
    "dense_dedup_rerank": {"mode": "dense", "dedupe": True, "use_reranker": True},
    "hybrid_dedup_rerank": {"mode": "hybrid", "dedupe": True, "use_reranker": True},
}


REWRITE_CACHE = Path("evals/cache/rewrites.json")


def cached_rewrites(questions: list[str]) -> dict[str, str]:
    """LLM rewrites, cached on disk so repeated evals don't spend API quota."""
    from supportpilot.retrieval.query_rewrite import rewrite_query

    cache = json.loads(REWRITE_CACHE.read_text(encoding="utf-8")) if REWRITE_CACHE.exists() else {}
    todo = [q for q in questions if q not in cache]
    for i, q in enumerate(todo, 1):
        cache[q] = rewrite_query(q)
        if i % 10 == 0 or i == len(todo):
            print(f"  rewrote {i}/{len(todo)}")
            REWRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            REWRITE_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def run(
    strategy: str,
    golden: list[dict],
    collection: str | None,
    reranker_model: str | None = None,
    sparse_weight: float | None = None,
    rewrite: bool = False,
) -> tuple[dict, list[dict]]:
    opts = STRATEGIES[strategy]
    rewrites = cached_rewrites([q["question"] for q in golden]) if rewrite else {}
    per_q, latencies = [], []
    for q in golden:
        t0 = time.perf_counter()
        hits = retriever.search(
            q["question"],
            top_n=max(K_VALUES),
            collection=collection,
            reranker_model=reranker_model,
            sparse_weight=sparse_weight,
            expansions=[rewrites[q["question"]]] if rewrite else None,
            **opts,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        doc_ids = [h.doc_id for h in hits]
        chunk_ids = [h.id for h in hits]
        # relevant_doc_ids adds near-duplicate pages (evals/expand_labels.py); strict = original label
        relevant = set(q.get("relevant_doc_ids") or q["expected_doc_ids"])
        per_q.append(
            {
                **{k: q.get(k) for k in ("id", "question", "section", "type", "source_chunk_id")},
                "expected_doc_ids": q["expected_doc_ids"],
                "rank": first_hit_rank(doc_ids, relevant),
                "chunk_rank": first_hit_rank(chunk_ids, {q["source_chunk_id"]}),
                **{f"recall@{k}": recall_at_k(doc_ids, relevant, k) for k in K_VALUES},
                "strict_recall@5": recall_at_k(doc_ids, set(q["expected_doc_ids"]), 5),
                "rr": reciprocal_rank(doc_ids, relevant),
                "top5": [{"doc_id": h.doc_id, "heading": h.heading, "score": round(h.score, 4)} for h in hits[:5]],
            }
        )

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            **{f"recall@{k}": round(mean([r[f"recall@{k}"] for r in rows]), 3) for k in K_VALUES},
            "mrr@10": round(mean([r["rr"] for r in rows]), 3),
            "strict_recall@5": round(mean([r["strict_recall@5"] for r in rows]), 3),
            "chunk_recall@5": round(mean([float(bool(r["chunk_rank"] and r["chunk_rank"] <= 5)) for r in rows]), 3),
        }

    by_section, by_type = defaultdict(list), defaultdict(list)
    for r in per_q:
        by_section[r["section"]].append(r)
        by_type[r["type"]].append(r)
    report = {
        "strategy": strategy,
        "collection": collection,
        "overall": summarize(per_q),
        "latency_ms": {"p50": round(statistics.median(latencies), 1), "p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 1)},
        "by_section": {s: summarize(v) for s, v in sorted(by_section.items(), key=lambda x: -len(x[1]))},
        "by_type": {t: summarize(v) for t, v in by_type.items()},
    }
    return report, per_q


def print_report(report: dict) -> None:
    o = report["overall"]
    print(f"\n== {report['strategy']} ({report['collection'] or 'default'}) on {o['n']} questions")
    print("  " + "  ".join(f"R@{k}={o[f'recall@{k}']:.3f}" for k in K_VALUES) + f"  MRR@10={o['mrr@10']:.3f}")
    print(f"  strict R@5={o['strict_recall@5']:.3f} (original single label)  chunk R@5={o['chunk_recall@5']:.3f}")
    print(f"  latency p50={report['latency_ms']['p50']}ms p95={report['latency_ms']['p95']}ms")
    print("  by section (n, R@5):")
    for s, m in report["by_section"].items():
        print(f"    {s:26s} n={m['n']:3d}  R@5={m['recall@5']:.2f}  MRR={m['mrr@10']:.2f}")
    print("  by type (n, R@5): " + ", ".join(f"{t} {m['n']}/{m['recall@5']:.2f}" for t, m in report["by_type"].items()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=list(STRATEGIES), default="hybrid_dedup_rerank")
    p.add_argument("--golden", type=Path, default=Path("evals/datasets/golden.jsonl"))
    p.add_argument("--collection")
    p.add_argument("--reranker", help="cross-encoder model, defaults to RERANKER_MODEL")
    p.add_argument("--sparse-weight", type=float, help="BM25 weight in RRF, defaults to HYBRID_SPARSE_WEIGHT")
    p.add_argument("--rewrite", action="store_true", help="also search with an LLM-rewritten query (cached)")
    p.add_argument("--tag", help="report name suffix, defaults to strategy")
    p.add_argument("--fail-under-recall", type=float)
    args = p.parse_args()

    golden = load_golden(args.golden)
    report, per_q = run(
        args.strategy, golden, args.collection, args.reranker, args.sparse_weight, args.rewrite
    )
    report["golden"] = str(args.golden)
    report["sparse_weight"] = args.sparse_weight if args.sparse_weight is not None else get_settings().hybrid_sparse_weight
    report["query_rewrite"] = args.rewrite  # latency excludes the LLM call (cached); measured separately
    report["reranker"] = (args.reranker or get_settings().reranker_model) if STRATEGIES[args.strategy]["use_reranker"] else None
    print_report(report)

    tag = args.tag or args.strategy
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"retrieval_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (REPORTS / f"retrieval_{tag}_failures.jsonl").open("w", encoding="utf-8") as f:
        for r in per_q:
            if not r["recall@5"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  report -> {REPORTS / f'retrieval_{tag}.json'}")

    if args.fail_under_recall is not None and report["overall"]["recall@5"] < args.fail_under_recall:
        print(f"FAIL: Recall@5 {report['overall']['recall@5']} < {args.fail_under_recall}")
        sys.exit(1)


if __name__ == "__main__":
    main()
