"""pydantic models mirroring candidate_schema.json.

Tolerant by design: optional blocks (certifications, languages, grade, tier) may be
missing/null; unknown extra fields are ignored. Parsing 100k rows must never crash on
a single odd record, so validation is lenient (coerce where sane, default where absent).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Profile(_Base):
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""


class CareerEntry(_Base):
    company: str = ""
    title: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


class Education(_Base):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: Optional[str] = None
    tier: str = "unknown"


class Skill(_Base):
    name: str = ""
    proficiency: str = "beginner"
    endorsements: int = 0
    duration_months: int = 0


class Certification(_Base):
    name: str = ""
    issuer: str = ""
    year: Optional[int] = None


class Language(_Base):
    language: str = ""
    proficiency: str = ""


class SalaryRange(_Base):
    min: float = 0.0
    max: float = 0.0


class RedrobSignals(_Base):
    profile_completeness_score: float = 0.0
    signup_date: Optional[str] = None
    last_active_date: Optional[str] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: SalaryRange = Field(default_factory=SalaryRange)
    preferred_work_mode: str = ""
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0       # -1 == no GitHub (NEUTRAL, never penalize)
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0       # -1 == no offer history (NEUTRAL)
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Candidate(_Base):
    candidate_id: str
    profile: Profile = Field(default_factory=Profile)
    career_history: list[CareerEntry] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)
