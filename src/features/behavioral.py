"""Behavioral / availability features from the 23 Redrob signals.

CRITICAL (proven in EDA): sentinel -1 means 'no history', NOT 'bad'. github=-1 in 64.6%
of the pool, offer_acceptance=-1 in 59.6%. Mapping those to low values would wrongly
sink most of the pool. We map every sentinel to a NEUTRAL midpoint. The JD endorses using
these as a *multiplier* on skill-match (see modifiers.py, Phase 6); here we emit the
component scores those modifiers consume.
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from .text import clamp, parse_date

NEUTRAL = 0.5


def _recency_score(days: float | None) -> float:
    # 1.0 if active within 30d, decaying to ~0.35 by a year, floored
    if days is None:
        return NEUTRAL
    if days <= 30:
        return 1.0
    if days >= 365:
        return 0.35
    return clamp(1.0 - 0.65 * (days - 30) / (365 - 30), 0.35, 1.0)


def _response_rate_score(rate: float, apps: int) -> float:
    # 0 response w/ 0 applications == nobody messaged them -> neutral, don't punish
    if rate == 0 and apps == 0:
        return NEUTRAL
    return clamp(0.05 + rate)  # 0.0->0.05, 0.8->0.85, 1.0->1.0


def _response_speed_score(hours: float) -> float:
    if hours <= 0:
        return NEUTRAL
    # ~1.0 within a few hours, ~0.5 by 48h, low by a week
    return clamp(1.0 - (hours / 168.0))


def features(c: Candidate, ref_now: date) -> dict:
    s = c.redrob_signals

    la = parse_date(s.last_active_date)
    days_since_active = (ref_now - la).days if la else None

    recency = _recency_score(days_since_active)
    response_rate = _response_rate_score(s.recruiter_response_rate or 0.0,
                                         s.applications_submitted_30d or 0)
    response_speed = _response_speed_score(s.avg_response_time_hours or 0.0)
    open_to_work = 1.0 if s.open_to_work_flag else 0.0
    interview_completion = clamp(s.interview_completion_rate) if s.interview_completion_rate is not None else NEUTRAL

    # sentinels -> neutral
    offer_acceptance = NEUTRAL if (s.offer_acceptance_rate is None or s.offer_acceptance_rate < 0) \
        else clamp(s.offer_acceptance_rate)
    github = NEUTRAL if (s.github_activity_score is None or s.github_activity_score < 0) \
        else clamp(s.github_activity_score / 100.0)

    # recruiter demand (saved + search appearances), log-saturated
    demand_raw = (s.saved_by_recruiters_30d or 0) + 0.2 * (s.search_appearance_30d or 0)
    recruiter_demand = clamp(demand_raw / 30.0)

    profile_completeness = clamp((s.profile_completeness_score or 0) / 100.0)

    np_days = s.notice_period_days if s.notice_period_days is not None else 90
    notice_score = clamp(1.0 - max(0, np_days - 30) / 150.0)  # <=30 best, 180 -> 0

    verification = (int(s.verified_email) + int(s.verified_phone) + int(s.linkedin_connected)) / 3.0

    return {
        "days_since_active": float(days_since_active) if days_since_active is not None else -1.0,
        "recency_score": recency,
        "response_rate_score": response_rate,
        "response_speed_score": response_speed,
        "open_to_work_flag": open_to_work,
        "interview_completion_score": interview_completion,
        "offer_acceptance_score": offer_acceptance,
        "github_score": github,
        "recruiter_demand_score": recruiter_demand,
        "profile_completeness_score": profile_completeness,
        "notice_score": notice_score,
        "verification_score": verification,
    }
