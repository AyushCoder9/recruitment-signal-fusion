"""Evaluation harness — the only steering wheel (no live leaderboard). Scores any ranking
variant against the LLM-judge tier labels (composite/NDCG/MAP) and reports honeypot /
off-target / foreign rates in the top-100 of the FULL ranking. Drives the ablation table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring import modifiers, pipeline, rubric
from .metrics import composite


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1)


def metrics_on_labeled(final_score: pd.Series, labels: pd.DataFrame) -> dict:
    """final_score indexed by candidate_id; rank the labeled candidates, score vs tiers."""
    common = final_score.index.intersection(labels.index)
    s = final_score.loc[common]
    rel = {cid: int(labels.loc[cid, "tier"]) for cid in common}
    order = list(s.sort_values(ascending=False).index)
    return composite(order, rel)


def top100_health(final_score: pd.Series, feat: pd.DataFrame, top_n: int = 100) -> dict:
    top = final_score.sort_values(ascending=False).head(top_n).index
    t = feat.loc[top]
    pure_off = ((t["is_offtarget_current"] > 0) & (t["relevant_history_flag"] == 0)).sum()
    return {
        "honeypot": int((t["honeypot_flag"] > 0).sum()),
        "offtarget": int(pure_off),
        "foreign": int((t["in_india_flag"] == 0).sum()),
    }


def variants(feat: pd.DataFrame, cfg: dict, model=None, cols=None) -> dict:
    """Return {name: final_score Series} for progressive ablation."""
    idx = feat.index
    out = {}
    out["bm25_only"] = pd.Series(_minmax(feat["bm25_score"]), index=idx)
    out["dense_only"] = pd.Series(_minmax(feat["dense_margin"]), index=idx)
    out["rubric_no_gates"] = pd.Series(rubric.rubric_score(feat), index=idx)
    # rubric + behavioral/location/gates
    rub = rubric.rubric_score(feat)
    mods = modifiers.apply_modifiers(rub, feat, cfg.get("modifiers"))
    out["rubric_full"] = pd.Series(mods["final"], index=idx)
    if model is not None:
        scored = pipeline.score(feat, cfg, model=model, cols=cols)
        out["lgbm_full"] = scored["final"]
    return out


def ablation(feat: pd.DataFrame, labels: pd.DataFrame, cfg: dict,
             model=None, cols=None) -> pd.DataFrame:
    rows = []
    for name, s in variants(feat, cfg, model, cols).items():
        m = metrics_on_labeled(s, labels)
        h = top100_health(s, feat)
        rows.append({"variant": name, "composite": round(m["composite"], 4),
                     "ndcg@10": round(m["ndcg@10"], 4), "ndcg@50": round(m["ndcg@50"], 4),
                     "map": round(m["map"], 4), "p@10": round(m["p@10"], 4),
                     **{f"top100_{k}": v for k, v in h.items()}})
    return pd.DataFrame(rows).set_index("variant")
