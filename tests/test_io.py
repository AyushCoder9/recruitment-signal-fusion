"""Phase 1 — IO/schema tests. Parse the 50 samples + first 1000 of the full set."""
from src.io.config import load_config, resolve_path
from src.io.loaders import iter_candidates, load_sample_candidates
from src.io.schema import Candidate


def test_config_interpolation():
    cfg = load_config()
    cand = cfg["paths"]["candidates_jsonl"]
    assert "${" not in cand and cand.endswith("candidates.jsonl")


def test_parse_50_samples():
    cands = load_sample_candidates()
    assert len(cands) == 50
    for c in cands:
        assert isinstance(c, Candidate)
        assert c.candidate_id.startswith("CAND_")
    # Ira Vora sanity (the data-engineer trap case)
    ira = cands[0]
    assert ira.candidate_id == "CAND_0000001"
    assert ira.profile.country == "Canada"
    assert ira.career_history and ira.career_history[0].duration_months > 0


def test_sentinel_defaults_preserved():
    # at least some real candidates carry -1 sentinels; ensure they parse as -1, not 0
    seen_gh_neg1 = False
    for c in iter_candidates(limit=2000):
        if c.redrob_signals.github_activity_score == -1:
            seen_gh_neg1 = True
            break
    assert seen_gh_neg1


def test_stream_first_1000_full_set():
    n = 0
    for c in iter_candidates(limit=1000):
        assert c.candidate_id and c.profile is not None
        assert isinstance(c.redrob_signals.skill_assessment_scores, dict)
        n += 1
    assert n == 1000


def test_missing_optional_blocks_ok():
    # construct a minimal record missing certifications/languages/education
    raw = {
        "candidate_id": "CAND_9999999",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 7},
        "career_history": [],
        "skills": [],
        "redrob_signals": {},
    }
    c = Candidate.model_validate(raw)
    assert c.certifications == [] and c.languages == []
    assert c.redrob_signals.offer_acceptance_rate == -1.0  # default sentinel
