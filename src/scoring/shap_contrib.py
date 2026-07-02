"""Per-candidate LightGBM SHAP attribution.

Uses LightGBM's native pred_contrib (Tree SHAP) — exact, not approximate.
Output shape: (n_candidates, n_features), with the final column being the bias term.

At rank time this is computed for the top-N candidates only (cheap: pure numpy,
no network, <1s on top-100 from the cached feature frame).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_LABELS = {
    "role_target_best":          "Role relevance (best title)",
    "role_target_current":       "Role relevance (current title)",
    "relevant_history_flag":     "Has relevant career history",
    "ever_built_relevant_flag":  "Built a relevant system",
    "is_offtarget_current":      "Current role off-target (neg)",
    "applied_ml_years":          "Applied ML years",
    "experience_fit":            "Experience fit (6–8 yr peak)",
    "yoe":                       "Total years of experience",
    "core_skill_score":          "Core skills (trust-weighted)",
    "embedding_retrieval_signal":"Embeddings / retrieval depth",
    "vector_db_signal":          "Vector DB experience",
    "eval_framework_signal":     "Ranking eval (NDCG/MRR/MAP)",
    "nlp_ir_depth":              "NLP / IR depth",
    "must_have_overlap":         "Must-have keyword overlap",
    "dense_margin":              "Semantic similarity margin",
    "bm25_score":                "BM25 lexical relevance",
    "production_signal":         "Production system signals",
    "scale_signal":              "Scale / high-load experience",
    "services_fraction":         "Services-company fraction (neg)",
    "all_services_career_flag":  "Entire career in services (neg)",
    "cv_dominant_flag":          "CV/speech-heavy, thin NLP (neg)",
    "langchain_only_recent_flag":"LLM-wrapper only, no pre-LLM (neg)",
    "honeypot_flag":             "Timeline inconsistency (neg)",
    "in_india_flag":             "India-based",
    "location_tier_score":       "Location tier score",
    "willing_to_relocate_flag":  "Open to relocate",
    "recency_score":             "Activity recency",
    "response_rate_score":       "Recruiter response rate",
    "assessment_alignment":      "Self-claimed vs assessed skills",
}


def top_contributions(
    model,
    feat_df: pd.DataFrame,
    cols: list[str],
    top_k: int = 6,
) -> pd.DataFrame:
    """Return a DataFrame of top_k positive and negative SHAP contributions per candidate.

    Returns columns: candidate_id, factor, label, shap_value.
    """
    X = feat_df[cols].to_numpy(dtype=float)
    contribs = model.predict(feat_df[cols], pred_contrib=True)
    # last col is bias — drop it
    contrib_df = pd.DataFrame(contribs[:, :-1], index=feat_df.index, columns=cols)
    bias = contribs[:, -1]  # noqa: F841 — available if needed

    rows = []
    for cid in feat_df.index:
        c = contrib_df.loc[cid]
        # pick top_k by absolute value
        top_idx = c.abs().nlargest(top_k).index
        for col in top_idx:
            rows.append({
                "candidate_id": cid,
                "factor": col,
                "label": FACTOR_LABELS.get(col, col),
                "shap_value": float(c[col]),
            })
    return pd.DataFrame(rows)


def candidate_shap_dict(
    model,
    feat_df: pd.DataFrame,
    cols: list[str],
    candidate_id: str,
    top_k: int = 8,
) -> dict[str, float]:
    """SHAP values for a single candidate — returns {factor_label: shap_value}."""
    row = feat_df.loc[[candidate_id], cols]
    contribs = model.predict(row, pred_contrib=True)[0, :-1]
    result = {cols[i]: float(contribs[i]) for i in range(len(cols))}
    # sort by abs value, top_k
    sorted_items = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return {FACTOR_LABELS.get(k, k): v for k, v in sorted_items}
