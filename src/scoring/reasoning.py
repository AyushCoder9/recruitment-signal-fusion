"""Fact-grounded, zero-hallucination reasoning. NO generative model is used at rank time
(forbidden + risks hallucination). Each reasoning string is assembled from the candidate's
TOP contributing factors and injects REAL facts (titles, companies, years, a distinctive
phrase quoted from their best-matching role, response rate, location). Because every clause
is pulled from a structured field, a hallucinated skill is impossible by construction.

Stage-4 guarantees: never all-identical (clauses vary by which factors fire + injected
facts), never name-only, never mentions a capability absent from the profile, acknowledges
concerns honestly, and tone is derived from the score so it stays rank-consistent.
"""
from __future__ import annotations

import re

from ..features.text import is_relevant_role
from ..io.schema import Candidate

_WS = re.compile(r"\s+")


def _distinctive_phrase(c: Candidate, max_words: int = 14) -> str:
    """First substantive clause from the best relevant role description (quoted fact)."""
    for e in c.career_history:
        if is_relevant_role(e) and e.description:
            first = re.split(r"[.;]", e.description)[0].strip()
            words = _WS.sub(" ", first).split()
            if len(words) >= 4:
                return " ".join(words[:max_words])
    # fallback: any role description
    for e in c.career_history:
        if e.description:
            words = _WS.sub(" ", re.split(r"[.;]", e.description)[0]).split()
            if len(words) >= 4:
                return " ".join(words[:max_words])
    return ""


def _tone(score: float) -> str:
    if score >= 0.66:
        return "strong"
    if score >= 0.4:
        return "solid"
    if score >= 0.22:
        return "adjacent"
    return "limited"


def reason_for(c: Candidate, f: dict, rank: int) -> str:
    p = c.profile
    g = lambda k: float(f.get(k, 0) or 0)
    score = g("final") if "final" in f else g("fit_raw")
    tone = _tone(score)
    clauses: list[str] = []

    # 1) role + experience (always, grounded in title + yoe)
    title = p.current_title or "Engineer"
    yoe = p.years_of_experience
    if g("role_target_best") >= 1.0 or g("relevant_history_flag") > 0:
        lead = f"{title}, {yoe:.1f} yrs"
        if g("applied_ml_years") >= 2:
            lead += f" incl. ~{g('applied_ml_years'):.0f} in applied ML"
        clauses.append(lead)
    else:
        clauses.append(f"{title}, {yoe:.1f} yrs")

    # 2) production / what they actually built (quoted from their own description)
    if g("ever_built_relevant_flag") > 0 or g("production_signal") > 0:
        phrase = _distinctive_phrase(c)
        if phrase:
            clauses.append(f"{tone} build record — \"{phrase}\"")
    # 3) retrieval/eval depth (only categories TRUE in their profile text)
    depth = []
    if g("embedding_retrieval_signal") > 0:
        depth.append("embeddings/retrieval")
    if g("vector_db_signal") > 0:
        depth.append("vector search")
    if g("eval_framework_signal") > 0:
        depth.append("ranking evaluation")
    if depth:
        clauses.append(", ".join(depth) + " in profile")

    # 4) location (grounded)
    if g("location_tier_score") >= 1.0:
        clauses.append(f"{p.location}-based (preferred)")
    elif g("in_india_flag") > 0:
        loc = p.location or "India"
        clauses.append(f"{loc}{', open to relocate' if g('willing_to_relocate_flag') > 0 else ''}")
    else:
        clauses.append(f"{p.country}{'; open to relocate' if g('willing_to_relocate_flag') > 0 else ' (outside India)'}")

    # 5) engagement (positive or honest concern), grounded in real signal values
    rr = c.redrob_signals.recruiter_response_rate
    if g("recency_score") >= 0.8 and g("response_rate_score") >= 0.55:
        clauses.append(f"engaged (response {rr:.2f}, active recently)")
    elif g("recency_score") < 0.45 or g("response_rate_score") < 0.2:
        clauses.append(f"low availability (response {rr:.2f}, inactive {int(g('days_since_active'))}d)")

    # 6) honest concerns
    concerns = []
    if g("is_offtarget_current") > 0 and g("relevant_history_flag") == 0:
        concerns.append("off-target current function")
    if g("services_fraction") >= 0.9:
        concerns.append("services-only career")
    if g("honeypot_flag") > 0:
        concerns.append("profile-consistency red flags")
    if g("cv_dominant_flag") > 0:
        concerns.append("CV/speech-leaning, thin NLP/IR")
    if g("langchain_only_recent_flag") > 0:
        concerns.append("recent LLM-wrapper only, limited pre-LLM depth")
    notice = c.redrob_signals.notice_period_days
    if notice and notice > 60:
        concerns.append(f"{notice}-day notice")
    if concerns:
        clauses.append("concern: " + ", ".join(concerns[:2]))

    # low-rank framing (rank-consistent honesty)
    if score < 0.22 and not concerns:
        clauses.append("adjacent fit, included near the cutoff")

    text = "; ".join(clauses) + "."
    return text[0].upper() + text[1:]
