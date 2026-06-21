"""Shared text + date helpers used across feature modules."""
from __future__ import annotations

from datetime import date, datetime

from ..io.schema import Candidate, CareerEntry
from . import lexicons as lx

PROF_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def months_between(a: date | None, b: date | None) -> int | None:
    if not a or not b:
        return None
    return (b.year - a.year) * 12 + (b.month - a.month)


def entry_end(e: CareerEntry, ref_now: date) -> date | None:
    """Effective end of a role: its end_date, or reference_now if current/open."""
    ed = parse_date(e.end_date)
    if ed:
        return ed
    if e.is_current or e.end_date is None:
        return ref_now
    return None


def role_text(e: CareerEntry) -> str:
    return f"{e.title} {e.description}".lower()


def all_titles_text(c: Candidate) -> str:
    parts = [c.profile.current_title]
    parts += [e.title for e in c.career_history]
    return " ".join(p for p in parts if p).lower()


def career_text(c: Candidate) -> str:
    """All role titles + descriptions, lowercased — the substantive experience text."""
    return " ".join(role_text(e) for e in c.career_history)


def skills_text(c: Candidate) -> str:
    return " ".join(s.name for s in c.skills if s.name).lower()


def full_text(c: Candidate) -> str:
    p = c.profile
    return " ".join([p.headline, p.summary, career_text(c), skills_text(c)]).lower()


def build_candidate_text(c: Candidate, weight_career: int = 2) -> str:
    """Text for embedding/BM25, weighted toward career descriptions (the JD reads
    history over the skills list). Career block repeated `weight_career` times."""
    p = c.profile
    head = f"{p.current_title}. {p.headline}. {p.summary}"
    career = career_text(c)
    skills = skills_text(c)
    return " ".join([head] + [career] * weight_career + [skills]).strip()


def is_relevant_role(e: CareerEntry) -> bool:
    """True if a single role is ML/retrieval/ranking-relevant by title OR description."""
    t = role_text(e)
    return lx.contains_any(t, lx.TARGET_TITLE_TERMS) or lx.contains_any(t, lx.CORE_SKILL_TERMS)


def title_is_offtarget(title: str) -> bool:
    return lx.contains_any((title or "").lower(), lx.OFFTARGET_TITLE_TERMS)


def title_is_target(title: str) -> bool:
    return lx.contains_any((title or "").lower(), lx.TARGET_TITLE_TERMS)


def title_is_manager(title: str) -> bool:
    return lx.contains_any((title or "").lower(), lx.MANAGER_TITLE_TERMS)


def seniority_of(title: str) -> float:
    t = (title or "").lower()
    best = 0.5
    for key, val in lx.SENIORITY_MAP.items():
        if key and key.strip() and key in t:
            best = max(best, val)
    return best


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
