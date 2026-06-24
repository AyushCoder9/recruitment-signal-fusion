"""Phase 9 — metric correctness on toy data with hand-checkable values."""
import math

from src.eval.metrics import (average_precision, composite, ndcg_at_k,
                              precision_at_k)

REL = {"a": 4, "b": 3, "c": 0, "d": 3, "e": 0}


def test_ndcg_perfect_is_one():
    ideal = ["a", "b", "d", "c", "e"]
    assert abs(ndcg_at_k(ideal, REL, 5) - 1.0) < 1e-9


def test_ndcg_worst_below_perfect():
    worst = ["c", "e", "b", "d", "a"]
    assert ndcg_at_k(worst, REL, 5) < ndcg_at_k(["a", "b", "d", "c", "e"], REL, 5)


def test_ndcg_known_value():
    # single relevant item (gain 1) at rank 2 -> dcg=1/log2(3); idcg=1/log2(2)=1
    rel = {"x": 1}
    val = ndcg_at_k(["z", "x"], rel, 5)
    assert abs(val - (1 / math.log2(3)) / 1.0) < 1e-9


def test_precision_at_k():
    order = ["a", "c", "b", "e", "d"]   # relevant(tier>=3): a,b,d
    assert precision_at_k(order, REL, 5) == 3 / 5
    assert precision_at_k(order, REL, 1) == 1.0


def test_average_precision():
    order = ["a", "c", "b", "e", "d"]   # rel at positions 1,3,5
    # AP = mean(1/1, 2/3, 3/5) over 3 relevants
    expect = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert abs(average_precision(order, REL) - expect) < 1e-9


def test_composite_weights():
    order = ["a", "b", "d", "c", "e"]
    m = composite(order, REL)
    manual = 0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"] + 0.15 * m["map"] + 0.05 * m["p@10"]
    assert abs(m["composite"] - manual) < 1e-9
