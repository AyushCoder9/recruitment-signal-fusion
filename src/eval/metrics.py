"""Ranking metrics — the exact scoring the challenge uses, so this is our only steering
wheel offline. composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10.

All functions take `ranked_ids` (best-first) and `relevance` (candidate_id -> tier 0..4).
'Relevant' for MAP / P@k means tier >= REL_THRESHOLD (3), per the spec ("tier 3+").
"""
from __future__ import annotations

import math

REL_THRESHOLD = 3


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: dict, k: int) -> float:
    gains = [(2 ** relevance.get(cid, 0) - 1) for cid in ranked_ids[:k]]
    ideal = sorted((2 ** r - 1) for r in relevance.values())[::-1][:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], relevance: dict, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= REL_THRESHOLD)
    return hits / k


def average_precision(ranked_ids: list[str], relevance: dict, k: int | None = None) -> float:
    n_rel_total = sum(1 for r in relevance.values() if r >= REL_THRESHOLD)
    if n_rel_total == 0:
        return 0.0
    ids = ranked_ids[:k] if k else ranked_ids
    hits = 0
    ap = 0.0
    for i, cid in enumerate(ids, 1):
        if relevance.get(cid, 0) >= REL_THRESHOLD:
            hits += 1
            ap += hits / i
    return ap / min(n_rel_total, len(ids)) if hits else 0.0


def composite(ranked_ids: list[str], relevance: dict) -> dict:
    m = {
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
        "ndcg@50": ndcg_at_k(ranked_ids, relevance, 50),
        "map": average_precision(ranked_ids, relevance),
        "p@5": precision_at_k(ranked_ids, relevance, 5),
        "p@10": precision_at_k(ranked_ids, relevance, 10),
    }
    m["composite"] = (0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"]
                      + 0.15 * m["map"] + 0.05 * m["p@10"])
    return m
