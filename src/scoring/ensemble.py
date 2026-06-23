"""Blend the learned ranker with the interpretable rubric -> fit_raw in [0,1].

fit_raw = w * norm(lgbm) + (1-w) * rubric. Falls back to pure rubric when the ranker is
absent (graceful degradation), so the system always produces a valid, strong ranking.
"""
from __future__ import annotations

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def blend_fit(rubric: np.ndarray, lgbm: np.ndarray | None = None,
              w: float = 0.6) -> np.ndarray:
    rubric = np.clip(np.asarray(rubric, dtype=float), 0, 1)
    if lgbm is None:
        return rubric
    return np.clip(w * _minmax(lgbm) + (1.0 - w) * rubric, 0, 1)
