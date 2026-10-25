"""Retrieval metrics.

Relevance is page-level by default: a question is answered if any retrieved chunk comes
from its source page. Chunk-level hit (exact source chunk retrieved) is reported too.
"""


def first_hit_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant item, or None."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return rank
    return None


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Hit rate @k (one relevant page per question, so recall@k == hit@k)."""
    return float(any(item in relevant for item in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    rank = first_hit_rank(retrieved[:k] if k else retrieved, relevant)
    return 1.0 / rank if rank else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
