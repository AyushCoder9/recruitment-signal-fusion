"""Consistency / honeypot detection.

EDA finding: the planted honeypots are NOT skill-keyword stuffers (advanced/expert @ 0
duration is ~0% of skill-rows). They are INTERNAL DATE/TIMELINE CONTRADICTIONS — the same
thing a careful recruiter catches ('wait, 8 years at a 3-year-old role?'). We score five
coherence checks; any hard logical impossibility sets honeypot_flag (high precision), and
softer divergences lower a graded consistency_score (a ranker feature).
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from .text import clamp, entry_end, months_between, parse_date


def _checks(c: Candidate, ref_now: date):
    prof = c.profile
    yoe = float(prof.years_of_experience or 0.0)
    reasons = []
    soft = 0.0

    total_dur = 0
    span_min = span_max = None
    for e in c.career_history:
        sd = parse_date(e.start_date)
        ed = entry_end(e, ref_now)
        dur = e.duration_months or 0
        total_dur += dur
        if sd:
            span_min = sd if span_min is None else min(span_min, sd)
        if ed:
            span_max = ed if span_max is None else max(span_max, ed)

        gap = months_between(sd, ed)
        if gap is not None and gap >= 0:
            is_open = e.is_current or e.end_date is None
            if is_open:
                # current role: elapsed>=stated is fine; only stated>>elapsed is impossible
                if dur - gap > 9:
                    reasons.append(f"date_mismatch(open gap={gap},dur={dur})")  # hard
                elif dur - gap > 5:
                    soft += 0.3
            else:
                # ended role: gap should equal stated duration
                diff = abs(gap - dur)
                if diff > 9:
                    reasons.append(f"date_mismatch(gap={gap},dur={dur})")  # hard
                elif diff > 5:
                    soft += 0.3
        # single role longer than entire career
        if yoe > 0 and dur > yoe * 12 + 6:
            reasons.append(f"role_dur({dur}mo)>yoe({yoe})")  # hard

    # YOE vs career span — claiming far more experience than the elapsed career span is a
    # classic honeypot. BUT a senior with an old degree who simply doesn't list early jobs is
    # legitimate, so we SUPPRESS the flag when education supports the claim (years since the
    # earliest graduation can account for the stated yoe). This keeps the planted timeline
    # traps while no longer flooring genuinely strong senior candidates.
    grad_years = [e.end_year for e in c.education if e.end_year]
    earliest_grad = min(grad_years) if grad_years else None
    edu_supports = earliest_grad is not None and (ref_now.year - earliest_grad) + 1.0 >= yoe
    if yoe > 0 and span_min and span_max:
        span_yrs = months_between(span_min, span_max) / 12.0
        if span_yrs + 1.5 < yoe and not edu_supports:  # more exp than elapsed, unexplained
            reasons.append(f"yoe({yoe})>span({round(span_yrs,1)}y)")  # hard
        elif span_yrs + 0.75 < yoe and not edu_supports:
            soft += 0.3
        # sum of durations wildly exceeds span (impossible heavy overlap)
        if total_dur > (months_between(span_min, span_max) or 0) * 1.6 + 12:
            soft += 0.3

    # education-vs-career timeline
    grads = [e.end_year for e in c.education if e.end_year]
    if grads and span_min and yoe >= 5:
        last_grad = max(grads)
        first_job_year = span_min.year
        if last_grad >= ref_now.year - 1 and last_grad - first_job_year > 4:
            reasons.append(f"edu_timeline(grad={last_grad},job_start={first_job_year})")  # hard

    # skill inflation (weak per EDA)
    infl = sum(1 for s in c.skills
               if s.proficiency in ("advanced", "expert") and (s.duration_months or 0) <= 1)
    if infl >= 5:
        soft += 0.3

    return reasons, soft


def features(c: Candidate, ref_now: date) -> dict:
    hard, soft = _checks(c, ref_now)
    n_hard = len(hard)
    penalty = clamp(0.5 * n_hard + soft)
    consistency_score = clamp(1.0 - penalty)
    honeypot_flag = 1.0 if n_hard >= 1 else 0.0
    return {
        "consistency_score": consistency_score,
        "honeypot_flag": honeypot_flag,
        "n_hard_contradictions": float(n_hard),
    }


def honeypot_reasons(c: Candidate, ref_now: date) -> list[str]:
    """Human-readable contradiction reasons (used by EDA / debugging / tests)."""
    return _checks(c, ref_now)[0]
