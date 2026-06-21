"""Company pedigree. JD negative: candidates who have ONLY ever worked at consulting/
services firms. Mixed history is explicitly fine ("if you have prior product-company
experience, that's fine"), so we score by tenure-fraction and only flag ~all-services.
"""
from __future__ import annotations

from datetime import date

from ..io.schema import Candidate
from . import lexicons as lx
from .text import clamp


def _is_services(company: str) -> bool:
    return lx.contains_any((company or "").lower(), lx.SERVICES_COMPANIES)


def features(c: Candidate, ref_now: date) -> dict:
    total = 0
    services_months = 0
    any_product = 0
    for e in c.career_history:
        m = e.duration_months or 0
        total += m
        if _is_services(e.company):
            services_months += m
        else:
            any_product = 1
    services_fraction = (services_months / total) if total else 0.0

    return {
        "services_fraction": services_fraction,
        "all_services_career_flag": 1.0 if services_fraction >= 0.95 and total > 0 else 0.0,
        "product_company_flag": float(any_product),
        "current_is_services": float(_is_services(c.profile.current_company)),
    }
