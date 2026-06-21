"""Location features. JD: Pune/Noida preferred (hybrid); Hyderabad/Mumbai/Delhi-NCR/
Bangalore welcome; other India ok; outside India case-by-case, NO visa sponsorship.
willing_to_relocate lifts other-India / borderline cases.
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from . import lexicons as lx
from .text import clamp


def _loc_text(c: Candidate) -> str:
    return f"{c.profile.location} {c.profile.country}".lower()


def features(c: Candidate, ref_now: date) -> dict:
    loc = (c.profile.location or "").lower()
    country = (c.profile.country or "").lower()
    in_india = 1.0 if "india" in country else 0.0
    relocate = 1.0 if c.redrob_signals.willing_to_relocate else 0.0

    if lx.contains_any(loc, lx.LOC_TIER1):
        base = 1.0
    elif lx.contains_any(loc, lx.LOC_TIER2):
        base = 0.85
    elif lx.contains_any(loc, lx.LOC_TIER3) or in_india:
        base = 0.65
    else:
        base = 0.30  # outside India: no visa sponsorship -> case-by-case

    # relocation willingness lifts sub-top-tier candidates toward the preferred range
    if base < 0.85 and relocate:
        base = min(0.85, base + 0.15)

    wm = (c.redrob_signals.preferred_work_mode or "").lower()
    work_mode_fit = {"hybrid": 1.0, "onsite": 0.9, "flexible": 0.95, "remote": 0.6}.get(wm, 0.7)

    return {
        "location_tier_score": base,
        "in_india_flag": in_india,
        "willing_to_relocate_flag": relocate,
        "work_mode_fit": work_mode_fit,
    }
