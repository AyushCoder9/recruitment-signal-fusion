"""Redrob ranker sandbox — Streamlit. Upload a <=100-candidate JSONL, get the ranked table
with grounded reasoning, a per-candidate factor-contribution chart, and a downloadable CSV.
Runs on CPU in seconds for <=100 candidates (embeds on the fly).

Run:  streamlit run sandbox/app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.registry import get_reference_now
from src.io.config import load_config
from src.io.schema import Candidate
from src.live_features import build_live_features
from src.scoring import pipeline, ranker
from src.scoring.reasoning import reason_for

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
cfg = load_config()

JD_SUMMARY = """\
**Senior AI Engineer — Redrob (Series-A, Pune/Noida hybrid).** Owns the intelligence layer:
ranking, retrieval, candidate-JD matching. Must-haves: production embeddings retrieval,
vector-DB/hybrid search, strong Python, ranking-evaluation (NDCG/MRR/MAP/A·B). Ideal: 6-8 yrs,
4-5 in applied ML at product cos, shipped a search/recsys/ranking system to real users.
Disqualifiers: off-target functions (Marketing/HR/Sales...), services-only careers, pure
research, LangChain-only-recent, CV/speech without NLP/IR. Behavioral signals = a multiplier.
"""

FACTOR_COLS = ["role_target_best", "core_skill_score", "embedding_retrieval_signal",
               "vector_db_signal", "eval_framework_signal", "experience_fit",
               "applied_ml_fraction", "production_signal", "dense_margin", "must_have_overlap",
               "availability", "location", "services_fraction", "honeypot_flag"]

st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.markdown(JD_SUMMARY)

up = st.file_uploader("Upload candidates JSONL (≤100)", type=["jsonl", "json"])
if up is None:
    st.info("Upload a JSONL of candidate profiles (schema = candidate_schema.json).")
    st.stop()

raw = up.read().decode("utf-8")
records = []
for line in raw.splitlines():
    line = line.strip()
    if line:
        records.append(json.loads(line))
if len(records) == 1 and isinstance(records[0], list):
    records = records[0]
records = records[:100]
cands = [Candidate.model_validate(r) for r in records]
st.success(f"Loaded {len(cands)} candidates.")

with st.spinner("Embedding + scoring on CPU..."):
    feat = build_live_features(cands, cfg)
    model, cols = ranker.load(cfg["artifacts"]["ranker"])
    scored = pipeline.score(feat, cfg, model=model, cols=cols)
    top = pipeline.rank_top(scored, top_n=min(100, len(cands)))
    objs = {c.candidate_id: c for c in cands}
    top["reasoning"] = [reason_for(objs[r.candidate_id], r.to_dict(), int(r["rank"]))
                        for _, r in top.iterrows()]

show = top[["rank", "candidate_id", "final", "reasoning"]].rename(columns={"final": "score"})
show["score"] = show["score"].round(4)
st.subheader("Ranked candidates")
st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button("⬇ Download submission CSV",
                   show.rename(columns={"score": "score"}).to_csv(index=False),
                   "submission.csv", "text/csv")

st.subheader("Per-candidate factor breakdown")
pick = st.selectbox("Candidate", top["candidate_id"].tolist())
row = scored.loc[pick]
chart = pd.Series({c: float(row.get(c, 0)) for c in FACTOR_COLS if c in scored.columns})
st.bar_chart(chart)
st.caption(f"model: {'LightGBM (distilled) + rubric' if model else 'rubric-only'} · "
           f"reference date {get_reference_now()}")
