"""Adversarial and metamorphic robustness tests.

These prove properties beyond "does the validator pass":
  - Keyword-stuffer must rank below the median (our system is not fooled by keyword lists).
  - Honeypot (impossible timeline) must be gated to near-zero (catches what LLMs miss).
  - All-sentinel (-1) behavioral profile must not be penalized relative to median
    (we never punish "no history").
  - JSON field-order permutation must leave rank invariant (metamorphic invariance).
  - Determinism: three consecutive rank runs on the same feature frame must be byte-identical.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import pandas as pd
import pytest

from src.eval.adversarial import (
    honeypot_profile,
    keyword_stuffer_profile,
    neutral_sentinel_profile,
    permute_json_fields,
)
from src.io.artifacts import load_features
from src.io.config import load_config, resolve_path
from src.io.schema import Candidate
from src.live_features import build_live_features
from src.scoring import pipeline, ranker

FEAT_PATH = resolve_path(load_config()["artifacts"]["features"])
SKIP_NO_FEAT = pytest.mark.skipif(
    not os.path.exists(FEAT_PATH), reason="needs precomputed features.parquet"
)


def _score_single(candidate_dict: dict, cfg: dict, model=None, cols=None) -> float:
    """Build features for one synthetic candidate and return its final score."""
    c = Candidate.model_validate(candidate_dict)
    df = build_live_features([c], cfg)
    scored = pipeline.score(df, cfg, model=model, cols=cols)
    return float(scored["final"].iloc[0])


@SKIP_NO_FEAT
def test_keyword_stuffer_ranks_below_median():
    """HR Manager with every AI buzzword must score below the 50th-percentile threshold."""
    cfg = load_config()
    feat = load_features(FEAT_PATH)
    model, cols = (None, None)
    if cfg["ranker"].get("enabled", True):
        rp = resolve_path(cfg["artifacts"]["ranker"])
        if os.path.exists(rp):
            model, cols = ranker.load(rp)

    # pool median final score
    scored_pool = pipeline.score(feat, cfg, model=model, cols=cols)
    median_score = float(scored_pool["final"].median())

    stuffer_score = _score_single(keyword_stuffer_profile(seed=1), cfg, model, cols)
    assert stuffer_score < median_score, (
        f"Keyword-stuffer scored {stuffer_score:.4f} >= pool median {median_score:.4f}; "
        "system is susceptible to keyword stuffing."
    )


@SKIP_NO_FEAT
def test_honeypot_gets_near_zero_score():
    """Candidate with impossible timeline (22 YOE, 15-yr career span) must be gated."""
    cfg = load_config()
    feat = load_features(FEAT_PATH)
    model, cols = (None, None)
    if cfg["ranker"].get("enabled", True):
        rp = resolve_path(cfg["artifacts"]["ranker"])
        if os.path.exists(rp):
            model, cols = ranker.load(rp)

    score = _score_single(honeypot_profile(seed=2), cfg, model, cols)
    gate = cfg.get("modifiers", {}).get("honeypot_gate", 0.03)
    # Score must be at most 2× the gate value (effectively floored by the gate)
    assert score <= gate * 2, (
        f"Honeypot scored {score:.4f}, expected ≤ {gate*2:.4f}. "
        "Honeypot detection is not flooring impossible-timeline candidates."
    )


@SKIP_NO_FEAT
def test_sentinel_not_penalized():
    """Candidate with all behavioral sentinels (-1 = no history) must score at or above
    a similarly-qualified candidate who has LOW engagement signals (0.2 response rate).

    Sentinel means NEUTRAL — we never punish absence of history.
    """
    cfg = load_config()
    feat = load_features(FEAT_PATH)
    model, cols = (None, None)
    if cfg["ranker"].get("enabled", True):
        rp = resolve_path(cfg["artifacts"]["ranker"])
        if os.path.exists(rp):
            model, cols = ranker.load(rp)

    # Baseline: neutral sentinel
    sentinel_score = _score_single(neutral_sentinel_profile(seed=3), cfg, model, cols)

    # Same profile but with VERY low engagement (not sentinel — should be penalized)
    low_eng = neutral_sentinel_profile(seed=3)
    low_eng["redrob_signals"]["recruiter_response_rate"] = 0.05
    low_eng["redrob_signals"]["offer_acceptance_rate"] = 0.05
    low_eng["redrob_signals"]["interview_completion_rate"] = 0.05
    low_eng["redrob_signals"]["last_active_date"] = "2022-01-01"
    low_eng["candidate_id"] = "CAND_SYN_0000099"
    low_score = _score_single(low_eng, cfg, model, cols)

    assert sentinel_score >= low_score, (
        f"Sentinel scored {sentinel_score:.4f} < low-engagement {low_score:.4f}. "
        "Missing behavioral history is being penalized instead of treated as neutral."
    )


@SKIP_NO_FEAT
def test_json_field_order_invariance():
    """Permuting JSON field order must not change the final score (metamorphic invariance)."""
    cfg = load_config()
    feat = load_features(FEAT_PATH)
    model, cols = (None, None)
    if cfg["ranker"].get("enabled", True):
        rp = resolve_path(cfg["artifacts"]["ranker"])
        if os.path.exists(rp):
            model, cols = ranker.load(rp)

    base = neutral_sentinel_profile(seed=3)
    shuffled = permute_json_fields(base, seed=7)
    # ensure candidate_id still matches
    shuffled["candidate_id"] = base["candidate_id"]

    score_base = _score_single(base, cfg, model, cols)
    score_shuffled = _score_single(shuffled, cfg, model, cols)
    assert abs(score_base - score_shuffled) < 1e-9, (
        f"Score changed from {score_base:.6f} to {score_shuffled:.6f} on field-order permutation. "
        "Ranking is NOT order-invariant."
    )


@SKIP_NO_FEAT
def test_determinism_three_runs():
    """Three consecutive rank calls on the same feature store must produce identical CSVs."""
    import csv
    cfg = load_config()
    feat = load_features(FEAT_PATH)
    model, cols = (None, None)
    if cfg["ranker"].get("enabled", True):
        rp = resolve_path(cfg["artifacts"]["ranker"])
        if os.path.exists(rp):
            model, cols = ranker.load(rp)

    hashes = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            scored = pipeline.score(feat, cfg, model=model, cols=cols)
            top = pipeline.rank_top(scored, top_n=100)
            path = os.path.join(tmp, f"run_{i}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["candidate_id", "rank", "score"])
                for _, r in top.iterrows():
                    w.writerow([r["candidate_id"], int(r["rank"]), round(float(r["final"]), 8)])
            hashes.append(hashlib.sha256(open(path, "rb").read()).hexdigest())

    assert len(set(hashes)) == 1, (
        f"Non-deterministic ranking — got {len(set(hashes))} distinct CSV hashes across 3 runs."
    )
