"""Phase 9 — end-to-end: scoring -> top-100 CSV must pass the OFFICIAL validator.
Also asserts the deterministic tie-break (candidate_id ascending on equal scores).
"""
import csv
import importlib.util
import os

import pytest

from src.io.artifacts import load_features
from src.io.config import load_config, resolve_path
from src.scoring import pipeline

FEAT = resolve_path(load_config()["artifacts"]["features"])
VALIDATOR = resolve_path(load_config()["paths"]["validator"])


def _load_validator():
    spec = importlib.util.spec_from_file_location("vsub", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.exists(FEAT), reason="needs precomputed features.parquet")
def test_generated_csv_passes_official_validator(tmp_path):
    cfg = load_config()
    feat = load_features(FEAT)
    scored = pipeline.score(feat, cfg)         # rubric-only path (model optional)
    top = pipeline.rank_top(scored, top_n=100)

    out = tmp_path / "team_test.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for _, r in top.iterrows():
            w.writerow([r["candidate_id"], int(r["rank"]), round(float(r["final"]), 6), "ok"])

    errors = _load_validator().validate_submission(str(out))
    assert errors == [], f"validator errors: {errors}"


@pytest.mark.skipif(not os.path.exists(FEAT), reason="needs precomputed features.parquet")
def test_tiebreak_candidate_id_ascending():
    cfg = load_config()
    feat = load_features(FEAT)
    top = pipeline.rank_top(pipeline.score(feat, cfg), top_n=100).reset_index(drop=True)
    for i in range(len(top) - 1):
        s1, s2 = top.loc[i, "final"], top.loc[i + 1, "final"]
        assert s1 >= s2 - 1e-12
        if abs(s1 - s2) < 1e-12:
            assert top.loc[i, "candidate_id"] < top.loc[i + 1, "candidate_id"]
