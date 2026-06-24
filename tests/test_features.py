"""Phase 2 — feature tests on hand-built fixtures spanning the key archetypes."""
from datetime import date

from src.features import consistency
from src.features.registry import extract_static
from src.io.schema import Candidate

REF = date(2026, 5, 27)


def cand(**kw):
    base = {"candidate_id": "CAND_0000000", "profile": {}, "career_history": [],
            "education": [], "skills": [], "redrob_signals": {}}
    base.update(kw)
    return Candidate.model_validate(base)


def clear_fit():
    return cand(
        profile={"current_title": "Senior ML Engineer", "current_company": "Swiggy",
                 "current_industry": "Food Delivery", "location": "Pune",
                 "country": "India", "years_of_experience": 7.0,
                 "headline": "ML engineer — retrieval & ranking",
                 "summary": "Built production semantic search and recommendation systems."},
        career_history=[
            {"company": "Swiggy", "title": "Senior ML Engineer", "start_date": "2021-02-01",
             "end_date": None, "duration_months": 52, "is_current": True,
             "description": "Shipped a production vector search recommendation system to "
                            "real users; tuned ranking with NDCG/MRR offline eval and A/B tests; "
                            "embeddings with sentence-transformers, FAISS, learning to rank."},
            {"company": "Flipkart", "title": "ML Engineer", "start_date": "2017-01-01",
             "end_date": "2021-01-01", "duration_months": 48, "is_current": False,
             "description": "Recommendation ranking with xgboost and collaborative filtering "
                            "in production at scale."},
        ],
        skills=[{"name": "Embeddings", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
                {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 15, "duration_months": 50}],
        redrob_signals={"last_active_date": "2026-05-20", "open_to_work_flag": True,
                        "recruiter_response_rate": 0.8, "applications_submitted_30d": 3,
                        "github_activity_score": 70, "offer_acceptance_rate": -1,
                        "notice_period_days": 30, "willing_to_relocate": True,
                        "preferred_work_mode": "hybrid", "profile_completeness_score": 95},
    )


def offtarget_stuffer():
    return cand(
        profile={"current_title": "Marketing Manager", "current_company": "Infosys",
                 "current_industry": "IT Services", "location": "Bangalore",
                 "country": "India", "years_of_experience": 8.0,
                 "summary": "Marketing leader. Skilled in machine learning, NLP, LLM, RAG."},
        career_history=[{"company": "Infosys", "title": "Marketing Manager",
                         "start_date": "2018-01-01", "end_date": None, "duration_months": 96,
                         "is_current": True, "description": "Led marketing campaigns."}],
        skills=[{"name": "Machine Learning", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
                {"name": "NLP", "proficiency": "expert", "endorsements": 3, "duration_months": 0},
                {"name": "LLM", "proficiency": "advanced", "endorsements": 2, "duration_months": 0}],
        redrob_signals={"last_active_date": "2026-05-25", "recruiter_response_rate": 0.5,
                        "skill_assessment_scores": {"Machine Learning": 30, "NLP": 25}},
    )


def honeypot():
    return cand(
        profile={"current_title": "Recommendation Systems Engineer", "location": "Pune",
                 "country": "India", "years_of_experience": 3.0,
                 "summary": "Recommendation systems, ranking, retrieval."},
        career_history=[{"company": "Acme", "title": "Recommendation Systems Engineer",
                         "start_date": "2023-01-01", "end_date": None, "duration_months": 200,
                         "is_current": True, "description": "Built ranking and retrieval."}],
        redrob_signals={"last_active_date": "2026-05-25"},
    )


def foreign():
    return cand(
        profile={"current_title": "ML Engineer", "current_company": "Google",
                 "location": "San Francisco", "country": "USA",
                 "years_of_experience": 7.0, "summary": "Retrieval and ranking."},
        career_history=[{"company": "Google", "title": "ML Engineer", "start_date": "2019-01-01",
                         "end_date": None, "duration_months": 88, "is_current": True,
                         "description": "Production ranking and embeddings retrieval at scale."}],
        redrob_signals={"last_active_date": "2026-05-25", "willing_to_relocate": False},
    )


def low_engagement():
    c = clear_fit()
    c.redrob_signals.last_active_date = "2025-06-01"   # ~360d before ref
    c.redrob_signals.recruiter_response_rate = 0.05
    c.redrob_signals.applications_submitted_30d = 4
    c.redrob_signals.open_to_work_flag = False
    return c


def test_clear_fit():
    f = extract_static(clear_fit(), REF)
    assert f["role_target_best"] == 1.0
    assert f["is_offtarget_current"] == 0.0
    assert f["embedding_retrieval_signal"] == 1.0
    assert f["eval_framework_signal"] == 1.0
    assert f["vector_db_signal"] == 1.0
    assert f["location_tier_score"] == 1.0
    assert f["services_fraction"] == 0.0
    assert f["honeypot_flag"] == 0.0
    assert f["recency_score"] == 1.0
    assert f["applied_ml_years"] > 6


def test_offtarget_stuffer_demoted_by_role_not_skills():
    f = extract_static(offtarget_stuffer(), REF)
    assert f["is_offtarget_current"] == 1.0
    assert f["role_target_current"] == 0.0
    assert f["all_services_career_flag"] == 1.0
    # assessment cross-check should flag the inflated expert claims
    assert f["assessment_alignment"] < 0.7
    # claimed expert skills with 0 duration -> inflation
    assert f["skill_claim_inflation"] >= 2


def test_honeypot_flagged():
    f = extract_static(honeypot(), REF)
    assert f["honeypot_flag"] == 1.0
    assert f["n_hard_contradictions"] >= 1
    reasons = consistency.honeypot_reasons(honeypot(), REF)
    assert any("role_dur" in r or "yoe" in r for r in reasons)


def test_foreign_location_penalized_but_not_zero():
    f = extract_static(foreign(), REF)
    assert f["in_india_flag"] == 0.0
    assert f["location_tier_score"] == 0.30   # no relocate -> outside-India floor
    assert f["role_target_best"] == 1.0       # still a strong role fit


def test_sentinels_neutral_not_punitive():
    f = extract_static(clear_fit(), REF)        # offer_acceptance=-1, but github=70
    assert f["offer_acceptance_score"] == 0.5   # -1 -> neutral
    assert f["github_score"] == 0.7


def test_low_engagement_downweighted():
    f = extract_static(low_engagement(), REF)
    assert f["recency_score"] <= 0.4
    assert f["response_rate_score"] <= 0.2
    assert f["open_to_work_flag"] == 0.0
    assert f["role_target_best"] == 1.0   # skills unchanged; only availability drops
