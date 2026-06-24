"""Phase 9 — reasoning guarantees (the Stage-4 checks):
no hallucinated skills, not all-identical, rank-consistent tone, honest concerns.
"""
from datetime import date

from src.features.registry import extract_static
from src.io.schema import Candidate
from src.scoring.reasoning import reason_for

REF = date(2026, 5, 27)


def _c(**kw):
    base = {"candidate_id": "CAND_0000000", "profile": {}, "career_history": [],
            "education": [], "skills": [], "redrob_signals": {}}
    base.update(kw)
    return Candidate.model_validate(base)


def fit_candidate():
    return _c(profile={"current_title": "ML Engineer", "location": "Pune", "country": "India",
                       "years_of_experience": 7.0, "summary": "retrieval and ranking"},
              career_history=[{"company": "Swiggy", "title": "ML Engineer",
                               "start_date": "2019-01-01", "end_date": None,
                               "duration_months": 88, "is_current": True,
                               "description": "Built a production vector search recommendation "
                                              "system with embeddings and NDCG evaluation."}],
              redrob_signals={"last_active_date": "2026-05-20", "recruiter_response_rate": 0.8,
                              "applications_submitted_30d": 3, "open_to_work_flag": True})


def offtarget_candidate():
    return _c(profile={"current_title": "Marketing Manager", "current_company": "Infosys",
                       "location": "Bangalore", "country": "India", "years_of_experience": 8.0},
              career_history=[{"company": "Infosys", "title": "Marketing Manager",
                               "start_date": "2018-01-01", "end_date": None,
                               "duration_months": 96, "is_current": True,
                               "description": "Led marketing campaigns."}])


def _reason(c, rank):
    f = extract_static(c, REF)
    f["final"] = 0.9 if rank <= 5 else 0.15
    return reason_for(c, f, rank)


def test_not_all_identical():
    rs = [_reason(fit_candidate(), 1), _reason(offtarget_candidate(), 90),
          _reason(fit_candidate(), 3)]
    assert len(set(rs)) >= 2


def test_grounds_real_facts():
    r = _reason(fit_candidate(), 1)
    assert "ML Engineer" in r and "Pune" in r and "7.0 yrs" in r


def test_no_hallucinated_skill():
    # candidate profile mentions NO 'kubernetes'/'pinecone' -> reasoning must not invent them
    r = _reason(fit_candidate(), 2).lower()
    for ghost in ["kubernetes", "pinecone", "tensorflow", "java", "golang"]:
        assert ghost not in r


def test_offtarget_acknowledges_concern():
    r = _reason(offtarget_candidate(), 95).lower()
    assert "concern" in r or "off-target" in r


def test_rank_consistent_tone():
    hi = _reason(fit_candidate(), 1)
    lo = _reason(fit_candidate(), 98)
    assert "strong" in hi.lower()
    assert ("limited" in lo.lower() or "adjacent" in lo.lower() or "cutoff" in lo.lower())
