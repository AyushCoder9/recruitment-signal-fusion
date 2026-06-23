"""Multiplicative gates + modifiers applied on top of the role/skill fit.

final_score = fit_raw * availability * location * offtarget_gate * honeypot_gate

The JD endorses behavioral signals as a 'multiplier or modifier on top of skill-match',
states off-target functions are 'not a fit no matter how perfect', and honeypots must be
floored. Off-target SPARES proven ex-builders (career history shows real ML/retrieval work)
— only floors pure off-target. All vectorized over the feature DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df, c):
    return df[c].to_numpy(dtype=float) if c in df else np.zeros(len(df))


def availability_modifier(df: pd.DataFrame, clamp=(0.35, 1.10)) -> np.ndarray:
    raw = (0.45 * _col(df, "recency_score")
           + 0.30 * _col(df, "response_rate_score")
           + 0.10 * (0.5 + 0.5 * _col(df, "open_to_work_flag"))
           + 0.08 * _col(df, "interview_completion_score")
           + 0.07 * _col(df, "offer_acceptance_score"))
    # small recruiter-demand bonus (saved/searched profiles are reachable & wanted)
    raw = raw + 0.05 * _col(df, "recruiter_demand_score")
    lo, hi = clamp
    return np.clip(lo + (hi - lo) * raw, lo, hi)


def location_modifier(df: pd.DataFrame, rng=(0.45, 1.05)) -> np.ndarray:
    lo, hi = rng
    base = _col(df, "location_tier_score")            # 0.30..1.0
    wm = _col(df, "work_mode_fit")                     # 0.6..1.0
    score = np.clip(0.85 * base + 0.15 * wm, 0, 1)
    return np.clip(lo + (hi - lo) * score, lo, hi)


def offtarget_gate(df: pd.DataFrame, gate=0.15) -> np.ndarray:
    off = _col(df, "is_offtarget_current") > 0
    has_relevant = _col(df, "relevant_history_flag") > 0
    built = _col(df, "ever_built_relevant_flag") > 0
    g = np.ones(len(df))
    # pure off-target (no relevant history at all) -> hard floor
    g[off & ~has_relevant] = gate
    # off-target NOW but built relevant systems before -> partial (don't fully kill)
    g[off & has_relevant & ~built] = max(gate, 0.5)
    g[off & built] = 0.7
    return g


def honeypot_gate(df: pd.DataFrame, gate=0.03) -> np.ndarray:
    g = np.ones(len(df))
    g[_col(df, "honeypot_flag") > 0] = gate
    return g


def apply_modifiers(fit_raw: np.ndarray, df: pd.DataFrame, cfg_mod: dict | None = None) -> dict:
    cfg_mod = cfg_mod or {}
    av = availability_modifier(df, tuple(cfg_mod.get("availability_clamp", (0.35, 1.10))))
    loc = location_modifier(df, tuple(cfg_mod.get("location_range", (0.45, 1.05))))
    off = offtarget_gate(df, cfg_mod.get("offtarget_gate", 0.15))
    hp = honeypot_gate(df, cfg_mod.get("honeypot_gate", 0.03))
    final = np.clip(fit_raw, 0, 1) * av * loc * off * hp
    return {"final": final, "availability": av, "location": loc,
            "offtarget_gate": off, "honeypot_gate": hp}
