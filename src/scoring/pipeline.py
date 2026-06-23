"""Single scoring entry point shared by rank.py, the eval harness, and the sandbox.

score(df) -> df augmented with rubric / lgbm / fit_raw / modifier / final columns, then
ranked. Keeps the precompute-vs-rank split honest: this consumes a feature frame (cached or
freshly built) and does only cheap numpy + a LightGBM predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ensemble, modifiers, ranker, rubric


def score(df: pd.DataFrame, cfg: dict, model=None, cols=None,
          weights=None, penalties=None) -> pd.DataFrame:
    out = df.copy()
    out["rubric"] = rubric.rubric_score(df, weights=weights, penalties=penalties)

    lgbm = None
    if model is not None:
        lgbm = ranker.predict(model, df, cols or ranker.feature_columns(df))
    out["lgbm"] = lgbm if lgbm is not None else np.nan

    w = cfg.get("ensemble", {}).get("blend_lgbm_weight", 0.6)
    out["fit_raw"] = ensemble.blend_fit(out["rubric"].to_numpy(), lgbm, w=w)

    mods = modifiers.apply_modifiers(out["fit_raw"].to_numpy(), df, cfg.get("modifiers"))
    for k, v in mods.items():
        out[k] = v
    return out


def rank_top(df_scored: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Deterministic ranking: final desc, tie-break candidate_id asc."""
    d = df_scored.reset_index()
    id_col = "candidate_id" if "candidate_id" in d.columns else d.columns[0]
    d = d.sort_values(by=["final", id_col], ascending=[False, True], kind="mergesort")
    d = d.head(top_n).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d
