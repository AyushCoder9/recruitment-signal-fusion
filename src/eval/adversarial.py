"""Adversarial and metamorphic test generators for the ranking system.

Four threat classes:
  1. Keyword-stuffer: a candidate with every AI buzzword but the wrong job function.
  2. Honeypot: internally contradictory timeline (impossible years-of-experience).
  3. Metamorphic invariance: reordering JSON fields / paraphrasing descriptions
     must not materially change a candidate's rank.
  4. Neutral-sentinel: a candidate where ALL behavioral signals are -1 (no history)
     must be ranked NEUTRALLY, not penalized.

Usage in tests/test_adversarial.py.
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any


def keyword_stuffer_profile(seed: int = 1) -> dict:
    """HR Manager with every AI keyword — should rank near the bottom."""
    rng = random.Random(seed)
    return {
        "candidate_id": f"CAND_SYN_{seed:07d}",
        "profile": {
            "anonymized_name": "Stuffer Candidate",
            "current_title": "HR Manager",
            "current_company": "TechBuzz Inc",
            "current_industry": "Human Resources",
            "current_company_size": "51-200",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 6.0,
            "github_activity_score": -1,
            "skills": [
                {"name": s, "proficiency": "Expert", "years_of_experience": 5,
                 "endorsements": 99, "assessment_score": None}
                for s in ["LLM", "RAG", "Vector Search", "FAISS", "Pinecone",
                          "Sentence Transformers", "LangChain", "Embeddings",
                          "NDCG", "MRR", "Retrieval", "Reranking", "BM25",
                          "Elasticsearch", "Neural IR"]
            ],
            "bio": ("Expert in RAG, LLM, embeddings, vector search, FAISS, Pinecone, "
                    "sentence-transformers, LangChain, NDCG, MRR, reranking, and "
                    "everything AI. 10x performance gains in retrieval pipelines."),
        },
        "career_history": [
            {
                "title": "HR Manager",
                "company": "TechBuzz Inc",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "description": ("Managed onboarding, performance review cycles, "
                                "and employee engagement programs. Organized team off-sites."),
                "is_current": True,
            },
            {
                "title": "Recruitment Coordinator",
                "company": "HireWell",
                "start_date": "2018-06-01",
                "end_date": "2021-01-01",
                "duration_months": 31,
                "description": ("Coordinated recruitment calendars and candidate logistics. "
                                "Maintained HRIS data and generated weekly hiring dashboards."),
                "is_current": False,
            },
        ],
        "education": [
            {"institution": "Symbiosis Institute", "degree": "MBA",
             "field": "Human Resources", "start_year": 2016, "end_year": 2018,
             "grade": None, "tier": "Tier-3"}
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "recruiter_response_rate": 0.45,
            "avg_response_time_hours": 12,
            "offer_acceptance_rate": -1,
            "interview_completion_rate": 0.5,
            "last_active_date": "2026-05-01",
            "profile_completeness_score": 0.9,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "willing_to_relocate": False,
            "preferred_work_mode": "hybrid",
            "applications_submitted_30d": 5,
            "github_activity_score": -1,
            "skill_assessment_scores": {},
        },
    }


def honeypot_profile(seed: int = 2) -> dict:
    """Timeline-contradiction honeypot: 22 years YOE but only 15 calendar years of career."""
    return {
        "candidate_id": f"CAND_SYN_{seed:07d}",
        "profile": {
            "anonymized_name": "Honeypot Candidate",
            "current_title": "Staff ML Engineer",
            "current_company": "DeepSearch AI",
            "current_industry": "Technology",
            "current_company_size": "51-200",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 22.0,  # impossible: only 2009-2024 = 15 yrs
            "github_activity_score": 4.8,
            "skills": [
                {"name": s, "proficiency": "Expert", "years_of_experience": 8,
                 "endorsements": 50, "assessment_score": 0.95}
                for s in ["Sentence Transformers", "FAISS", "LightGBM", "Python",
                          "Elasticsearch", "Vector Search"]
            ],
            "bio": "Built retrieval systems at scale. Expert in dense + sparse hybrid search.",
        },
        "career_history": [
            {
                "title": "Staff ML Engineer",
                "company": "DeepSearch AI",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 52,
                "description": "Built production embedding pipeline serving 10M users.",
                "is_current": True,
            },
            {
                "title": "ML Engineer",
                "company": "Flipkart",
                "start_date": "2015-06-01",
                "end_date": "2020-01-01",
                "duration_months": 55,
                "description": "Search ranking and retrieval at scale.",
                "is_current": False,
            },
            {
                "title": "Software Engineer",
                "company": "Infosys",
                "start_date": "2009-07-01",
                "end_date": "2015-06-01",
                "duration_months": 71,
                "description": "Backend Java development.",
                "is_current": False,
            },
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech",
             "field": "Computer Science", "start_year": 2005, "end_year": 2009,
             "grade": "9.1", "tier": "Tier-1"}
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "recruiter_response_rate": 0.88,
            "avg_response_time_hours": 2,
            "offer_acceptance_rate": 0.9,
            "interview_completion_rate": 0.95,
            "last_active_date": "2026-05-20",
            "profile_completeness_score": 1.0,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "willing_to_relocate": True,
            "preferred_work_mode": "hybrid",
            "applications_submitted_30d": 10,
            "github_activity_score": 4.8,
            "skill_assessment_scores": {"Python": 0.95, "Machine Learning": 0.92},
        },
    }


def neutral_sentinel_profile(seed: int = 3) -> dict:
    """All behavioral signals are -1 (no history) — must not be penalized."""
    return {
        "candidate_id": f"CAND_SYN_{seed:07d}",
        "profile": {
            "anonymized_name": "Fresh Profile",
            "current_title": "Senior ML Engineer",
            "current_company": "Swiggy",
            "current_industry": "Technology",
            "current_company_size": "1001-5000",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 7.0,
            "github_activity_score": -1,
            "skills": [
                {"name": s, "proficiency": "Advanced", "years_of_experience": 4,
                 "endorsements": 0, "assessment_score": None}
                for s in ["Python", "Elasticsearch", "Sentence Transformers",
                          "FAISS", "Retrieval", "LightGBM"]
            ],
            "bio": "Building search and recommendation systems in production.",
        },
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "Swiggy",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 28,
                "description": ("Built hybrid BM25 + dense retrieval pipeline for restaurant "
                                "search, achieving 15% NDCG@10 improvement."),
                "is_current": True,
            },
            {
                "title": "ML Engineer",
                "company": "Razorpay",
                "start_date": "2019-03-01",
                "end_date": "2022-01-01",
                "duration_months": 34,
                "description": ("Deployed embedding-based fraud detection model, "
                                "latency p99 < 50ms."),
                "is_current": False,
            },
        ],
        "education": [
            {"institution": "NIT Trichy", "degree": "B.Tech",
             "field": "Computer Science", "start_year": 2015, "end_year": 2019,
             "grade": None, "tier": "Tier-2"}
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "recruiter_response_rate": -1,
            "avg_response_time_hours": 0,
            "offer_acceptance_rate": -1,
            "interview_completion_rate": -1,
            "last_active_date": None,
            "profile_completeness_score": 0.7,
            "notice_period_days": 0,
            "open_to_work_flag": False,
            "willing_to_relocate": False,
            "preferred_work_mode": "",
            "applications_submitted_30d": 0,
            "github_activity_score": -1,
            "skill_assessment_scores": {},
        },
    }


def permute_json_fields(profile: dict, seed: int = 42) -> dict:
    """Shuffle all JSON field orders recursively — ranks should be identical."""
    rng = random.Random(seed)

    def _shuffle(obj: Any) -> Any:
        if isinstance(obj, dict):
            items = list(obj.items())
            rng.shuffle(items)
            return {k: _shuffle(v) for k, v in items}
        if isinstance(obj, list):
            return [_shuffle(x) for x in obj]
        return obj

    return _shuffle(copy.deepcopy(profile))


def generate_test_batch() -> list[dict]:
    """Return a mixed batch of 4 synthetic candidates for testing."""
    return [
        keyword_stuffer_profile(seed=1),
        honeypot_profile(seed=2),
        neutral_sentinel_profile(seed=3),
        # a second stuffer with different seed
        keyword_stuffer_profile(seed=4),
    ]
