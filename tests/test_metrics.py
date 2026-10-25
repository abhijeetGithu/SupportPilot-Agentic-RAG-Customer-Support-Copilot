from supportpilot.evaluation.metrics import first_hit_rank, recall_at_k, reciprocal_rank


def test_metrics():
    retrieved = ["a", "b", "c", "b"]
    assert first_hit_rank(retrieved, {"c"}) == 3
    assert recall_at_k(retrieved, {"c"}, 2) == 0.0
    assert recall_at_k(retrieved, {"c"}, 3) == 1.0
    assert reciprocal_rank(retrieved, {"b"}) == 0.5
    assert reciprocal_rank(retrieved, {"z"}) == 0.0
    assert reciprocal_rank(retrieved, {"c"}, k=2) == 0.0
