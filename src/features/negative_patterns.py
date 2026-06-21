"""JD disqualifier patterns not already produced by other modules.

JD disqualifiers: (1) pure research / academic with no production deployment;
(2) senior who hasn't written production code in 18 months because they moved into
'architecture'/'tech lead'. (The langchain-only, all-services, CV-dominant, and
title-chaser disqualifiers live in skills/company/experience.)
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from . import lexicons as lx
from .text import (career_text, entry_end, is_relevant_role, months_between,
                   role_text, title_is_manager)


def features(c: Candidate, ref_now: date) -> dict:
    car = career_text(c)

    # pure research: most roles look academic/research AND no production language anywhere
    research_roles = sum(1 for e in c.career_history if lx.contains_any(role_text(e), lx.RESEARCH_TERMS))
    n = max(1, len(c.career_history))
    has_production = lx.contains_any(car, lx.PRODUCTION_TERMS)
    pure_research_flag = 1.0 if (research_roles / n >= 0.6 and not has_production) else 0.0

    # moved-to-architecture / no recent code: current title is manager/architect AND
    # no relevant role ended within the last 18 months
    has_recent_relevant = False
    for e in c.career_history:
        if is_relevant_role(e):
            ed = entry_end(e, ref_now)
            gap = months_between(ed, ref_now)
            if gap is not None and gap <= 18:
                has_recent_relevant = True
                break
    no_recent_code_flag = 1.0 if (title_is_manager(c.profile.current_title)
                                  and not has_recent_relevant) else 0.0

    return {
        "pure_research_flag": pure_research_flag,
        "no_recent_code_flag": no_recent_code_flag,
    }
