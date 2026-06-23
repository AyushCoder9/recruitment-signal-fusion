"""Transparent weighted rubric -> role/skill 'fit' in [0,1].

This is the interpretable backbone: it produces a defensible ranking on its own (graceful
fallback if the learned ranker is absent), feeds the learned ranker as a feature, and blends
with it for the final fit. It scores ROLE + SEMANTIC + SKILLS + EXPERIENCE + PRODUCTION and
applies soft disqualifier penalties. Behavioral availability, location, and the hard
off-target / honeypot GATES are applied separately (multiplicatively) in modifiers.py.

Vectorized over the full feature DataFrame so rank-time is a handful of numpy ops on 100k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "role": 0.30,
    "semantic": 0.15,
    "skills": 0.23,
    "experience": 0.12,
    "production": 0.10,
    "must_have": 0.10,
}
DEFAULT_PENALTIES = {           # soft disqualifier weights (subtractive, capped)
    "services_fraction": 0.20,
    "cv_dominant_flag": 0.25,
    "langchain_only_recent_flag": 0.25,
    "pure_research_flag": 0.30,
    "no_recent_code_flag": 0.20,
    "tenure_chaser_flag": 0.15,
}


def _norm(s: pd.Series) -> np.ndarray:
    x = s.to_numpy(dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def rubric_score(df: pd.DataFrame, weights: dict | None = None,
                 penalties: dict | None = None) -> np.ndarray:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    pen = {**DEFAULT_PENALTIES, **(penalties or {})}
    g = lambda c: df[c].to_numpy(dtype=float) if c in df else np.zeros(len(df))

    # role fit: best relevant role, lifted by having actually built in production
    role = np.clip(0.7 * g("role_target_best") + 0.3 * g("ever_built_relevant_flag"), 0, 1)

    # semantic fit: how much more 'ideal' than 'anti-ideal' + must-have overlap
    margin = _norm(df["dense_margin"]) if "dense_margin" in df else np.zeros(len(df))
    semantic = np.clip(0.7 * margin + 0.3 * g("must_have_overlap"), 0, 1)

    # skills fit: trust-weighted core + JD must-have signals, gated by assessment honesty
    skills = np.clip(
        0.4 * g("core_skill_score") + 0.2 * g("embedding_retrieval_signal")
        + 0.15 * g("vector_db_signal") + 0.15 * g("nlp_ir_depth")
        + 0.10 * g("eval_framework_signal"), 0, 1) * (0.6 + 0.4 * g("assessment_alignment"))

    experience = np.clip(0.6 * g("experience_fit") + 0.4 * g("applied_ml_fraction"), 0, 1)

    production = np.clip(0.5 * g("production_signal") + 0.3 * g("scale_signal")
                         + 0.2 * g("ever_built_relevant_flag"), 0, 1)

    must_have = g("must_have_overlap")

    base = (w["role"] * role + w["semantic"] * semantic + w["skills"] * skills
            + w["experience"] * experience + w["production"] * production
            + w["must_have"] * must_have)
    base = base / sum(w.values())

    penalty = np.zeros(len(df))
    for col, pw in pen.items():
        penalty += pw * g(col)
    penalty = np.clip(penalty, 0, 0.85)

    return np.clip(base * (1.0 - penalty), 0, 1)
