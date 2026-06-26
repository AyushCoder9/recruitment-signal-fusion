"""Recruitment Signal Fusion — live ranking sandbox (Streamlit).

Upload a <=100-candidate file (JSONL or JSON array), get a ranked table with grounded reasoning,
KPI cards, top-3 highlights, and a per-candidate factor breakdown. Runs on CPU in seconds and
embeds uploaded profiles on the fly. No generative model at rank time — reasoning is fact-grounded.

Run:  streamlit run sandbox/app.py
"""
import json
import os
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.registry import get_reference_now
from src.io.config import load_config
from src.io.schema import Candidate
from src.live_features import build_live_features
from src.scoring import pipeline, ranker
from src.scoring.reasoning import reason_for

st.set_page_config(page_title="Recruitment Signal Fusion · Candidate Ranker",
                   page_icon="🎯", layout="wide",
                   menu_items={"about": "Recruitment Signal Fusion — Team NullSet · "
                               "JD-as-rubric candidate ranking. github.com/AyushCoder9/recruitment-signal-fusion"})

PURPLE, DEEP, DARK, MUTED = "#7d45e0", "#5b2bc4", "#1b2227", "#6b7480"

# --------------------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: 'Manrope', system-ui, sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
.hero {{ background: linear-gradient(110deg, {DEEP} 0%, {PURPLE} 60%, #9a6cf0 100%);
        border-radius: 18px; padding: 28px 34px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 12px 34px rgba(125,69,224,.28); }}
.hero h1 {{ font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -.5px; }}
.hero p {{ font-size: 15px; opacity: .92; margin: 8px 0 0; font-weight: 500; max-width: 880px; }}
.hero .pills {{ margin-top: 14px; }}
.hero .pill {{ display:inline-block; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
        padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 8px; }}
.step {{ background:#fff; border:1px solid #ece6fb; border-radius:14px; padding:18px 18px 16px;
        height:100%; box-shadow:0 4px 16px rgba(27,34,39,.04); }}
.step .n {{ display:inline-flex; width:30px; height:30px; align-items:center; justify-content:center;
        background:{PURPLE}; color:#fff; border-radius:9px; font-weight:800; font-size:14px; }}
.step h4 {{ margin:12px 0 4px; font-size:15px; font-weight:700; color:{DARK}; }}
.step p {{ margin:0; font-size:13px; color:{MUTED}; line-height:1.5; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #ece6fb; border-radius:14px;
        padding:14px 18px; box-shadow:0 4px 16px rgba(27,34,39,.04); }}
div[data-testid="stMetricValue"] {{ font-weight:800; color:{DARK}; }}
div[data-testid="stMetricLabel"] p {{ font-weight:600; color:{MUTED}; }}
.podium {{ background:#fff; border:1px solid #ece6fb; border-left:5px solid {PURPLE}; border-radius:14px;
        padding:16px 18px; box-shadow:0 4px 16px rgba(27,34,39,.05); height:100%; }}
.podium .rk {{ font-size:12px; font-weight:800; color:{PURPLE}; letter-spacing:.5px; }}
.podium .ti {{ font-size:15px; font-weight:700; color:{DARK}; margin:4px 0 2px; }}
.podium .id {{ font-size:12px; color:{MUTED}; font-family:ui-monospace,monospace; }}
.podium .sc {{ font-size:24px; font-weight:800; color:{DEEP}; margin-top:8px; }}
.stDownloadButton button {{ background:{PURPLE}; color:#fff; border:0; border-radius:10px;
        font-weight:700; padding:.5rem 1.1rem; }}
.stDownloadButton button:hover {{ background:{DEEP}; color:#fff; }}
section[data-testid="stSidebar"] {{ background:#faf8ff; border-right:1px solid #efeafc; }}
.sb-card {{ background:#fff; border:1px solid #ece6fb; border-radius:12px; padding:14px 15px; font-size:13px;
        color:#3a4148; line-height:1.5; }}
.foot {{ text-align:center; color:{MUTED}; font-size:12px; margin-top:30px; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <h1>🎯 Intelligent Candidate Discovery &amp; Ranking</h1>
  <p>The job description is the scoring rubric. This engine reproduces a careful recruiter applying
  it — judging by career trajectory and job titles, not a keyword list — fusing dense retrieval, BM25,
  a transparent rubric, and an LLM-judge-distilled LightGBM ranker, with honeypot &amp; off-target gates.</p>
  <div class="pills">
    <span class="pill">CPU-only</span><span class="pill">No network at rank time</span>
    <span class="pill">Zero-hallucination reasoning</span><span class="pill">Honeypot-aware</span>
  </div>
</div>
""", unsafe_allow_html=True)

cfg = load_config()
JD_SUMMARY = (
    "**Senior AI Engineer — Redrob** (Series-A · Pune/Noida hybrid). Owns the intelligence layer: "
    "ranking, retrieval, candidate–JD matching.\n\n"
    "**Must-haves:** production embeddings retrieval · vector-DB / hybrid search · strong Python · "
    "ranking-evaluation (NDCG/MRR/MAP/A·B).\n\n"
    "**Ideal:** 6–8 yrs, 4–5 in applied ML at product companies; shipped a search/recsys/ranking "
    "system to real users.\n\n"
    "**Disqualifiers:** off-target functions (Marketing/HR/Sales) · services-only careers · pure "
    "research · LangChain-only-recent · CV/speech without NLP/IR. *Behavioral signals = a multiplier.*"
)
FACTOR_LABELS = {
    "role_target_best": "Role / title fit", "core_skill_score": "Core skills (trust-weighted)",
    "embedding_retrieval_signal": "Embeddings / retrieval", "vector_db_signal": "Vector DB / search",
    "eval_framework_signal": "Ranking evaluation", "experience_fit": "Experience fit",
    "applied_ml_fraction": "Applied-ML share", "production_signal": "Production / shipped",
    "dense_margin": "Semantic margin", "must_have_overlap": "Must-have coverage",
    "availability": "Availability", "location": "Location", "services_fraction": "Services-co (−)",
    "honeypot_flag": "Honeypot flag (−)",
}
FACTOR_COLS = list(FACTOR_LABELS)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### The role")
    st.markdown(f'<div class="sb-card">{JD_SUMMARY}</div>', unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown(
        '<div class="sb-card">1. Parse + embed each profile on CPU.<br>'
        '2. Score with rubric + distilled LightGBM ranker.<br>'
        '3. Apply availability / location / off-target / honeypot gates.<br>'
        '4. Rank, tie-break, and explain — grounded in real fields.</div>', unsafe_allow_html=True)
    st.markdown("### Links")
    st.markdown("- [GitHub repository](https://github.com/AyushCoder9/recruitment-signal-fusion)\n"
                "- Team **NullSet**")

# --------------------------------------------------------------------------- upload
up = st.file_uploader("Upload candidates — JSONL (one object per line) or a JSON array, ≤ 100",
                      type=["jsonl", "json"])

if up is None:
    st.markdown("#### How the sandbox works")
    c1, c2, c3 = st.columns(3)
    for col, n, h, p in [
        (c1, "1", "Upload profiles", "Drop a JSONL or JSON array of up to 100 candidates "
         "matching candidate_schema.json (the bundled sample_candidates.json works)."),
        (c2, "2", "Rank on CPU", "Profiles are embedded and scored in seconds — same engine that "
         "ranks the full 100k pool, no network, no hosted LLM."),
        (c3, "3", "Inspect & export", "See the ranked table with grounded reasoning, a top-3 view, "
         "a per-candidate factor breakdown, and a downloadable CSV.")]:
        col.markdown(f'<div class="step"><span class="n">{n}</span><h4>{h}</h4><p>{p}</p></div>',
                     unsafe_allow_html=True)
    st.info("⬆ Upload a candidate file above to run the ranker.")
    st.stop()

# --------------------------------------------------------------------------- parse (robust)
raw = up.read().decode("utf-8", errors="replace").strip()
records = []
if raw:
    try:
        obj = json.loads(raw)                       # whole-file JSON: array or single object
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        for line in raw.splitlines():               # JSONL fallback
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
records = [r for r in records if isinstance(r, dict)][:100]
if not records:
    st.error("No candidate records found. Upload a JSONL file (one JSON object per line) or a "
             "JSON array of candidates matching candidate_schema.json.")
    st.stop()

cands, skipped = [], 0
for r in records:
    try:
        cands.append(Candidate.model_validate(r))
    except Exception:
        skipped += 1
if not cands:
    st.error("Could not parse any candidate against the expected schema (candidate_schema.json).")
    st.stop()

# --------------------------------------------------------------------------- score
with st.spinner("Embedding + scoring on CPU…"):
    t0 = time.time()
    feat = build_live_features(cands, cfg)
    model, cols = ranker.load(cfg["artifacts"]["ranker"])
    scored = pipeline.score(feat, cfg, model=model, cols=cols)
    top = pipeline.rank_top(scored, top_n=min(100, len(cands)))
    objs = {c.candidate_id: c for c in cands}
    top["reasoning"] = [reason_for(objs[r.candidate_id], r.to_dict(), int(r["rank"]))
                        for _, r in top.iterrows()]
    elapsed = time.time() - t0

n = len(top)
hp = int((scored.get("honeypot_flag", pd.Series(dtype=float)) > 0).sum())
best = float(top["final"].iloc[0])
st.success(f"Ranked {n} candidates in {elapsed:.1f}s." +
           (f"  ({skipped} skipped — schema mismatch)" if skipped else ""))

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidates scored", f"{n}")
k2.metric("Honeypots flagged", f"{hp}", help="Internal-timeline-impossible profiles, floored by the gate.")
k3.metric("Top score", f"{best:.3f}")
k4.metric("Ranked in", f"{elapsed:.1f}s", help="CPU-only, no network.")

# top-3 podium
st.markdown("#### Top picks")
cols3 = st.columns(min(3, n))
for i, col in enumerate(cols3):
    r = top.iloc[i]
    cid = r["candidate_id"]
    title = (objs[cid].profile.current_title or "—")
    col.markdown(
        f'<div class="podium"><div class="rk">RANK #{int(r["rank"])}</div>'
        f'<div class="ti">{title}</div><div class="id">{cid}</div>'
        f'<div class="sc">{float(r["final"]):.3f}</div></div>', unsafe_allow_html=True)

# ranked table
st.markdown("#### Full ranking")
show = top[["rank", "candidate_id", "final", "reasoning"]].rename(columns={"final": "score"})
maxv = max(float(show["score"].max()), 1e-6)
st.dataframe(
    show, use_container_width=True, hide_index=True, height=min(460, 60 + 36 * n),
    column_config={
        "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
        "candidate_id": st.column_config.TextColumn("Candidate", width="small"),
        "score": st.column_config.ProgressColumn("Fit score", min_value=0.0, max_value=maxv,
                                                 format="%.3f", width="small"),
        "reasoning": st.column_config.TextColumn("Why this rank (grounded)", width="large"),
    })
st.download_button("⬇  Download ranked CSV", show.to_csv(index=False),
                   "team_nullset_sample.csv", "text/csv")

# factor breakdown
st.markdown("#### Per-candidate factor breakdown")
left, right = st.columns([1, 2.4])
pick = left.selectbox("Candidate", top["candidate_id"].tolist())
left.caption(f"**{objs[pick].profile.current_title or '—'}**  ·  "
             f"{objs[pick].profile.years_of_experience:.1f} yrs")
row = scored.loc[pick]
fac = pd.DataFrame({"factor": [FACTOR_LABELS[c] for c in FACTOR_COLS if c in scored.columns],
                    "value": [float(row.get(c, 0)) for c in FACTOR_COLS if c in scored.columns]})
chart = (alt.Chart(fac).mark_bar(cornerRadiusEnd=4, color=PURPLE)
         .encode(x=alt.X("value:Q", title=None),
                 y=alt.Y("factor:N", sort="-x", title=None),
                 tooltip=["factor", alt.Tooltip("value:Q", format=".3f")])
         .properties(height=34 * len(fac)))
right.altair_chart(chart, use_container_width=True)

st.markdown(
    f'<div class="foot">Recruitment Signal Fusion · Team NullSet · model: '
    f'{"LightGBM (distilled) + rubric" if model else "rubric-only"} · reference date '
    f'{get_reference_now()} · reasoning is fact-grounded, no generative model at rank time</div>',
    unsafe_allow_html=True)
