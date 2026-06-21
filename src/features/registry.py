"""Feature registry — assembles every static feature into one row per candidate.

Static features depend ONLY on candidate data (no JD query), so they are precomputed once
into features.parquet (Phase 3). Query-time features (dense_sim_pos/neg, dense_margin,
bm25_score) are appended at rank time. extract_static() is the single source of truth for
the static columns; build_feature_frame() vectorizes it over a candidate stream.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..io.config import load_config
from ..io.schema import Candidate
from . import (behavioral, company, consistency, experience, location,
               negative_patterns, role, semantic, skills)

# modules contributing static features, in a stable order
_MODULES = [role, experience, skills, company, location, behavioral,
            consistency, negative_patterns, semantic]

# query-time columns appended later (declared so the ranker knows the full schema)
QUERY_FEATURES = ["dense_sim_pos", "dense_sim_neg", "dense_margin", "bm25_score"]


def get_reference_now() -> date:
    cfg = load_config()
    ref = cfg.get("reference_now")
    if ref:
        return date.fromisoformat(str(ref))
    return date(2026, 5, 27)  # fallback to the EDA-derived value


def extract_static(c: Candidate, ref_now: date) -> dict:
    row: dict = {"candidate_id": c.candidate_id}
    for m in _MODULES:
        row.update(m.features(c, ref_now))
    return row


def static_feature_names(ref_now: date | None = None) -> list[str]:
    """Column order of the static feature vector (excludes candidate_id)."""
    ref_now = ref_now or get_reference_now()
    probe = Candidate(candidate_id="CAND_0000000")
    cols = [k for k in extract_static(probe, ref_now) if k != "candidate_id"]
    return cols


def build_feature_frame(candidates, ref_now: date | None = None,
                        limit: int | None = None) -> pd.DataFrame:
    ref_now = ref_now or get_reference_now()
    rows = []
    for i, c in enumerate(candidates):
        if limit is not None and i >= limit:
            break
        rows.append(extract_static(c, ref_now))
    df = pd.DataFrame(rows)
    return df.set_index("candidate_id")
