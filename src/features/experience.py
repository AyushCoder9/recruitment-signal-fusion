"""Experience features. JD: 5-9 yrs is a *range not a requirement*; ideal 6-8 with 4-5
in applied ML at product companies; penalize title-chasers and 'moved to architecture,
no recent code'.
"""
from __future__ import annotations

import math
from datetime import date

from ..io.schema import Candidate
from .text import clamp, entry_end, is_relevant_role, months_between, parse_date


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


def features(c: Candidate, ref_now: date) -> dict:
    yoe = float(c.profile.years_of_experience or 0.0)

    durs = [e.duration_months or 0 for e in c.career_history]
    n_roles = len(durs)
    avg_tenure = (sum(durs) / n_roles) if n_roles else 0.0

    applied_ml_months = sum((e.duration_months or 0) for e in c.career_history
                            if is_relevant_role(e))
    applied_ml_years = applied_ml_months / 12.0

    # months since the end of the most recent relevant role (0 if currently relevant)
    last_rel_gap_months = None
    for e in c.career_history:
        if is_relevant_role(e):
            ed = entry_end(e, ref_now)
            gap = months_between(ed, ref_now)
            if gap is not None and gap >= 0:
                last_rel_gap_months = gap if last_rel_gap_months is None else min(last_rel_gap_months, gap)
    years_since_last_relevant = (last_rel_gap_months / 12.0) if last_rel_gap_months is not None else 99.0

    # soft fit: peak at 7, gentle; JD explicitly says range not requirement
    experience_fit = _gaussian(yoe, mu=7.0, sigma=3.0)
    # but don't fully kill very experienced people: floor the tail
    experience_fit = clamp(0.15 + 0.85 * experience_fit)

    # title-chaser: many short stints
    tenure_chaser = 1.0 if (n_roles >= 4 and avg_tenure < 18) else 0.0

    return {
        "yoe": yoe,
        "experience_fit": experience_fit,
        "applied_ml_years": applied_ml_years,
        "applied_ml_fraction": clamp(applied_ml_years / yoe) if yoe > 0 else 0.0,
        "n_roles": float(n_roles),
        "avg_tenure_months": avg_tenure,
        "years_since_last_relevant": years_since_last_relevant,
        "tenure_chaser_flag": tenure_chaser,
    }
