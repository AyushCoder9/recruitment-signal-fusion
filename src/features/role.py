"""Role / function features — the single highest-leverage block.

The JD's central trap: judge function from job TITLES + career history, not the skills
list. A Marketing Manager with 9 AI skills is off-target; a quiet backend engineer who
built vector search is on-target. We therefore score the BEST relevant role across the
whole history (so an ex-builder now in management still gets credit) while flagging an
off-target *current* function for the gate in Phase 6.
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from . import lexicons as lx
from .text import (is_relevant_role, role_text, seniority_of, title_is_manager,
                   title_is_offtarget, title_is_target)


def features(c: Candidate, ref_now: date) -> dict:
    cur = c.profile.current_title or ""

    # current-title function score
    if title_is_target(cur):
        role_target_current = 1.0
    elif title_is_offtarget(cur):
        role_target_current = 0.0
    else:
        role_target_current = 0.4  # neutral (Cloud/QA/Project... ambiguous)

    # best relevant role across current + history (catches plain-language fits)
    best = role_target_current
    relevant_history = 0
    ever_built = 0
    for e in c.career_history:
        if is_relevant_role(e):
            relevant_history = 1
            t = role_text(e)
            score = 1.0
            # bonus if the role text shows actually BUILDING a system in production
            if lx.contains_any(t, lx.PRODUCTION_TERMS) and lx.contains_any(t, lx.CORE_SKILL_TERMS):
                ever_built = 1
            best = max(best, score)

    return {
        "role_target_current": role_target_current,
        "role_target_best": best,
        "relevant_history_flag": float(relevant_history),
        "ever_built_relevant_flag": float(ever_built),
        "is_offtarget_current": float(title_is_offtarget(cur)),
        "is_manager_current": float(title_is_manager(cur)),
        "title_seniority": seniority_of(cur),
    }
