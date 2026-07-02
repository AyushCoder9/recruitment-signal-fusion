"""Recruitment Signal Fusion — live ranking sandbox (Streamlit).

Upload a <=100-candidate file (JSONL or JSON array), get a ranked table with grounded reasoning,
KPI cards, top-3 highlights, per-candidate SHAP factor breakdown, and an optional side-by-side
comparison against a naive keyword-count baseline to demonstrate stuffer-demotion.

Run:  streamlit run sandbox/app.py
"""
import json
import os
import re
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
from src.scoring.shap_contrib import FACTOR_LABELS, candidate_shap_dict

st.set_page_config(page_title="Recruitment Signal Fusion · Candidate Ranker",
                   page_icon="🎯", layout="wide",
                   menu_items={"about": "Recruitment Signal Fusion — Team NullSet · "
                               "JD-as-rubric candidate ranking. github.com/AyushCoder9/recruitment-signal-fusion"})

PURPLE, DEEP, DARK, MUTED = "#7d45e0", "#5b2bc4", "#1b2227", "#6b7480"
GREEN, RED, AMBER = "#22c55e", "#ef4444", "#f59e0b"

# ───────────────────────────────────────────────────────────────────── CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: 'Manrope', system-ui, sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1360px; }}
.hero {{ background: linear-gradient(110deg, {DEEP} 0%, {PURPLE} 60%, #9a6cf0 100%);
        border-radius: 18px; padding: 28px 34px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 12px 34px rgba(125,69,224,.28); }}
.hero h1 {{ font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -.5px; }}
.hero p {{ font-size: 15px; opacity: .92; margin: 8px 0 0; font-weight: 500; max-width: 900px; }}
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
.sb-card {{ background:#fff; border:1px solid #ece6fb; border-radius:12px; padding:14px 15px;
        font-size:13px; color:#3a4148; line-height:1.5; }}
.foot {{ text-align:center; color:{MUTED}; font-size:12px; margin-top:30px; }}
.badge-hp {{ background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:8px 12px;
        color:{RED}; font-size:12.5px; font-weight:600; }}
.badge-ok {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:8px 12px;
        color:{GREEN}; font-size:12.5px; font-weight:600; }}
.naive-badge {{ display:inline-block; background:#fef3c7; border:1px solid #fcd34d; color:#92400e;
        border-radius:8px; padding:3px 8px; font-size:11px; font-weight:700; margin-left:6px; }}
.ours-badge {{ display:inline-block; background:#ede9fe; border:1px solid #c4b5fd; color:{DEEP};
        border-radius:8px; padding:3px 8px; font-size:11px; font-weight:700; margin-left:6px; }}
.compare-header {{ font-size:13px; font-weight:700; color:{DARK}; padding:10px 0 6px; }}
.rank-delta-up {{ color:{GREEN}; font-weight:700; }}
.rank-delta-down {{ color:{RED}; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────── hero
st.markdown(f"""
<div class="hero">
  <h1>🎯 Intelligent Candidate Discovery &amp; Ranking</h1>
  <p>The job description is the scoring rubric. This engine reproduces a careful recruiter applying
  it — judging by career trajectory and job titles, not a keyword list — fusing dense retrieval, BM25,
  a transparent rubric, and an LLM-judge-distilled LightGBM ranker, with honeypot &amp; off-target gates.
  Switch to <strong>Compare Mode</strong> to see live stuffer-demotion vs a naive baseline.</p>
  <div class="pills">
    <span class="pill">CPU-only · ~1s per 50 candidates</span>
    <span class="pill">No network at rank time</span>
    <span class="pill">Zero-hallucination reasoning</span>
    <span class="pill">SHAP factor attribution</span>
    <span class="pill">Honeypot-aware</span>
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
    "**Disqualifiers:** off-target functions (Marketing/HR/Sales) · services-only careers · "
    "LangChain-only-recent · CV/speech without NLP/IR. *Behavioral signals = a gentle multiplier.*"
)

# ───────────────────────────────────────────────────────────────────── sidebar
with st.sidebar:
    st.markdown("### The role")
    st.markdown(f'<div class="sb-card">{JD_SUMMARY}</div>', unsafe_allow_html=True)
    st.markdown("### Mode")
    mode = st.radio("Display", ["Our Engine", "Compare: Naive vs Ours"],
                    help="'Compare' shows keyword-count baseline vs our system — stuffers visibly crash.")
    st.markdown("### How it works")
    st.markdown(
        '<div class="sb-card">'
        '1. Parse + embed each profile on CPU (no network).<br>'
        '2. Score: rubric + distilled LightGBM ranker (59 features).<br>'
        '3. Apply availability / location / off-target / honeypot gates.<br>'
        '4. Rank, tie-break, explain — all grounded in real fields.<br>'
        '5. SHAP Tree-SHAP shows exact per-feature attributions.</div>', unsafe_allow_html=True)
    st.markdown("### Links")
    st.markdown("- [GitHub repository](https://github.com/AyushCoder9/recruitment-signal-fusion)\n"
                "- Team **NullSet**")

# ───────────────────────────────────────────────────────────────────── upload
up = st.file_uploader("Upload candidates — JSONL (one object per line) or a JSON array, ≤ 100",
                      type=["jsonl", "json"])

if up is None:
    st.markdown("#### How the sandbox works")
    c1, c2, c3, c4 = st.columns(4)
    for col, n, h, p in [
        (c1, "1", "Upload profiles",
         "Drop a JSONL or JSON array of up to 100 candidates. sample_candidates.json from the "
         "challenge bundle works directly."),
        (c2, "2", "Rank on CPU",
         "Embed and score in seconds — same engine that ranks the full 100k pool. No network, "
         "no hosted LLM at rank time."),
        (c3, "3", "SHAP attribution",
         "Click any candidate to see its exact Tree-SHAP breakdown — which features pushed the "
         "rank up or down, with real values."),
        (c4, "4", "Compare baselines",
         "Switch to Compare Mode to see keyword-count vs our engine side-by-side. "
         "Stuffers crash; genuine engineers rise."),
    ]:
        col.markdown(f'<div class="step"><span class="n">{n}</span><h4>{h}</h4><p>{p}</p></div>',
                     unsafe_allow_html=True)
    st.info("⬆ Upload a candidate file above to run the ranker.")
    st.stop()

# ───────────────────────────────────────────────────────────────────── parse (robust)
raw = up.read().decode("utf-8", errors="replace").strip()
records = []
if raw:
    try:
        obj = json.loads(raw)
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        for line in raw.splitlines():
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
             "JSON array matching candidate_schema.json.")
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

# ───────────────────────────────────────────────────────────────────── score
with st.spinner("Embedding + scoring on CPU…"):
    t0 = time.time()
    feat = build_live_features(cands, cfg)
    model_obj, model_cols = None, None
    try:
        model_obj, model_cols = ranker.load(cfg["artifacts"]["ranker"])
    except Exception:
        pass
    scored = pipeline.score(feat, cfg, model=model_obj, cols=model_cols)
    top = pipeline.rank_top(scored, top_n=min(100, len(cands)))
    objs = {c.candidate_id: c for c in cands}
    top["reasoning"] = [reason_for(objs[r["candidate_id"]], r.to_dict(), int(r["rank"]))
                        for _, r in top.iterrows()]
    elapsed = time.time() - t0

n = len(top)
hp_count = int((feat.get("honeypot_flag", pd.Series(0, index=feat.index)) > 0).sum()
               if hasattr(feat, 'get') else 0)
best = float(top["final"].iloc[0])

# ───────────────────────────────────────────────────────────────────── KPIs
if skipped:
    st.warning(f"{skipped} candidate(s) skipped — schema mismatch.")
st.success(f"Ranked {n} candidate{'s' if n != 1 else ''} in {elapsed:.1f}s.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidates scored", f"{n}")
k2.metric("Honeypots flagged", f"{hp_count}",
          help="Internal-timeline-impossible profiles — gated to near-zero score.")
k3.metric("Top fit score", f"{best:.3f}")
k4.metric("Ranked in", f"{elapsed:.1f}s", help="CPU-only, no network.")

# ───────────────────────────────────────────────────────────────────── top-3 podium
st.markdown("#### Top picks")
cols3 = st.columns(min(3, n))
for i, col in enumerate(cols3):
    r = top.iloc[i]
    cid = r["candidate_id"]
    c = objs[cid]
    title = c.profile.current_title or "—"
    loc = c.profile.location or c.profile.country or "—"
    col.markdown(
        f'<div class="podium">'
        f'<div class="rk">RANK #{int(r["rank"])}</div>'
        f'<div class="ti">{title}</div>'
        f'<div class="id">{cid} · {loc}</div>'
        f'<div class="sc">{float(r["final"]):.3f}</div></div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────── compare / single
if mode == "Compare: Naive vs Ours":
    _naive_keywords = [
        "embedding", "retrieval", "vector", "faiss", "pinecone", "weaviate", "qdrant",
        "elasticsearch", "opensearch", "bm25", "semantic search", "dense retrieval",
        "sparse retrieval", "hybrid search", "sentence transformer", "sentence-transformer",
        "sbert", "e5", "bge", "rag", "llm", "rerank", "reranking", "cross-encoder",
        "ndcg", "mrr", "map", "precision@", "recall@", "a/b test", "a/b testing",
        "ranking", "recommendation", "recsys", "neural ir", "nlp", "bert", "transformer",
        "pytorch", "tensorflow", "lightgbm", "xgboost", "sklearn", "langchain",
    ]

    def _naive_score(c: Candidate) -> float:
        text = " ".join([
            c.profile.bio or "",
            " ".join(s.name for s in c.profile.skills),
            " ".join(e.description or "" for e in c.career_history),
        ]).lower()
        return float(sum(1 for kw in _naive_keywords if kw in text))

    naive_scores = {c.candidate_id: _naive_score(c) for c in cands}
    max_naive = max(naive_scores.values()) or 1.0
    naive_df = pd.DataFrame([
        {"candidate_id": cid, "naive_score": s / max_naive, "naive_rank": 0}
        for cid, s in sorted(naive_scores.items(), key=lambda x: -x[1])
    ])
    naive_df["naive_rank"] = range(1, len(naive_df) + 1)

    merged = top[["candidate_id", "rank", "final", "reasoning"]].merge(
        naive_df, on="candidate_id", how="left")
    merged["rank_delta"] = merged["naive_rank"] - merged["rank"]

    st.markdown("---")
    st.markdown("""
**Compare Mode** — left: naive keyword-count ranker (like the sample submission);
right: our engine. Watch keyword-stuffers crash and genuine engineers rise.
""")
    left_h, right_h = st.columns(2)
    left_h.markdown(
        '<div class="compare-header">⚠️ Naive baseline <span class="naive-badge">keyword count</span></div>',
        unsafe_allow_html=True)
    right_h.markdown(
        f'<div class="compare-header">✅ Our engine <span class="ours-badge">LightGBM + rubric + gates</span></div>',
        unsafe_allow_html=True)

    left_c, right_c = st.columns(2)
    naive_top = naive_df.head(n).copy()
    naive_top["title"] = [objs[cid].profile.current_title or "—"
                          for cid in naive_top["candidate_id"]]
    naive_top["naive_score_raw"] = [naive_scores[cid] for cid in naive_top["candidate_id"]]
    is_hp_naive = [feat.loc[cid, "honeypot_flag"] > 0 if cid in feat.index else False
                   for cid in naive_top["candidate_id"]]
    naive_top["⚠️"] = ["🚨 Honeypot" if h else "" for h in is_hp_naive]

    left_c.dataframe(
        naive_top[["naive_rank", "candidate_id", "title", "naive_score_raw", "⚠️"]].rename(
            columns={"naive_rank": "Rank", "candidate_id": "ID",
                     "title": "Current title", "naive_score_raw": "Keyword hits"}),
        use_container_width=True, hide_index=True, height=min(460, 60 + 36 * n),
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Keyword hits": st.column_config.ProgressColumn(
                min_value=0, max_value=float(max(naive_scores.values()) or 1),
                format="%d", width="small"),
        })

    ours_top = top[["rank", "candidate_id", "final"]].copy()
    ours_top["title"] = [objs[cid].profile.current_title or "—"
                         for cid in ours_top["candidate_id"]]
    ours_top["honeypot"] = ["🚨" if (feat.loc[cid, "honeypot_flag"] > 0
                                      if cid in feat.index else False) else "✓"
                             for cid in ours_top["candidate_id"]]
    right_c.dataframe(
        ours_top[["rank", "candidate_id", "title", "final", "honeypot"]].rename(
            columns={"rank": "Rank", "candidate_id": "ID",
                     "title": "Current title", "final": "Fit score", "honeypot": "HP"}),
        use_container_width=True, hide_index=True, height=min(460, 60 + 36 * n),
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Fit score": st.column_config.ProgressColumn(
                min_value=0.0, max_value=max(float(top["final"].max()), 1e-6),
                format="%.3f", width="small"),
            "HP": st.column_config.TextColumn(width="small"),
        })

    # biggest movers callout
    st.markdown("#### Biggest rank changes (naive → ours)")
    movers = merged.copy()
    movers["title"] = [objs[cid].profile.current_title or "—" for cid in movers["candidate_id"]]
    movers = movers.sort_values("rank_delta", ascending=False)
    m1, m2 = st.columns(2)

    def delta_str(d):
        if d > 0:
            return f"<span class='rank-delta-up'>▼ {d} (demoted by naive)</span>"
        if d < 0:
            return f"<span class='rank-delta-down'>▲ {abs(d)} (promoted by ours)</span>"
        return "no change"

    m1.markdown("**Most demoted by naive (keyword stuffers)**")
    for _, r in movers.head(3).iterrows():
        m1.markdown(f"- **{r['title']}** (`{r['candidate_id']}`): "
                    f"naive rank #{int(r['naive_rank'])} → our rank #{int(r['rank'])} "
                    f"{delta_str(int(r['rank_delta']))}", unsafe_allow_html=True)

    m2.markdown("**Most promoted by ours (genuine engineers)**")
    for _, r in movers.tail(3).iterrows():
        m2.markdown(f"- **{r['title']}** (`{r['candidate_id']}`): "
                    f"naive rank #{int(r['naive_rank'])} → our rank #{int(r['rank'])} "
                    f"{delta_str(int(r['rank_delta']))}", unsafe_allow_html=True)

else:
    # ─── ranked table ──────────────────────────────────────────────────────
    st.markdown("#### Full ranking")
    show = top[["rank", "candidate_id", "final", "reasoning"]].rename(columns={"final": "score"})
    maxv = max(float(show["score"].max()), 1e-6)
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=min(480, 60 + 36 * n),
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
            "candidate_id": st.column_config.TextColumn("Candidate", width="small"),
            "score": st.column_config.ProgressColumn("Fit score", min_value=0.0, max_value=maxv,
                                                     format="%.3f", width="small"),
            "reasoning": st.column_config.TextColumn("Why this rank (grounded)", width="large"),
        })
    st.download_button("⬇  Download ranked CSV", show.to_csv(index=False),
                       "team_nullset_sample.csv", "text/csv")

# ───────────────────────────────────────────────────────────────────── SHAP breakdown
st.markdown("---")
st.markdown("#### Per-candidate SHAP factor breakdown")
st.caption(
    "Tree-SHAP attributions from the LightGBM ranker — exact, not approximate. "
    "Positive values push rank up; negative values push it down. Every value "
    "traces to a real profile field — no generative text.")

fa_left, fa_right = st.columns([1, 2.4])
pick = fa_left.selectbox("Inspect candidate", top["candidate_id"].tolist())
c = objs[pick]
r_row = top[top["candidate_id"] == pick].iloc[0]
fa_left.markdown(
    f"**{c.profile.current_title or '—'}**  \n"
    f"`{pick}` · {c.profile.location or c.profile.country or '—'}  \n"
    f"**Fit score: {float(r_row['final']):.3f}** · Rank #{int(r_row['rank'])}"
)

# Honeypot badge
hp_flag = float(feat.loc[pick, "honeypot_flag"]) > 0 if pick in feat.index else False
if hp_flag:
    # Try to surface the exact contradiction
    yoe = c.profile.years_of_experience
    if c.career_history:
        dates = []
        for e in c.career_history:
            if e.start_date:
                try:
                    y = int(e.start_date.split("-")[0])
                    dates.append(y)
                except Exception:
                    pass
            if e.end_date:
                try:
                    y = int(e.end_date.split("-")[0])
                    dates.append(y)
                except Exception:
                    pass
        span = (max(dates) - min(dates)) if len(dates) >= 2 else 0
        fa_left.markdown(
            f'<div class="badge-hp">🚨 Honeypot flagged<br>'
            f'Claimed YOE: {yoe:.0f} yrs | Career span: {span} yrs<br>'
            f'Gap: {yoe - span:.0f} yrs cannot be explained by employment history.</div>',
            unsafe_allow_html=True)
else:
    fa_left.markdown('<div class="badge-ok">✓ No timeline inconsistencies detected</div>',
                     unsafe_allow_html=True)

# SHAP chart
if model_obj is not None and pick in feat.index:
    try:
        shap_vals = candidate_shap_dict(model_obj, feat.loc[[pick]], model_cols, pick, top_k=10)
        shap_df = pd.DataFrame([{"factor": k, "shap": v} for k, v in shap_vals.items()])
        shap_df = shap_df.sort_values("shap")
        shap_df["color"] = shap_df["shap"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
        chart = (alt.Chart(shap_df)
                 .mark_bar(cornerRadiusEnd=4)
                 .encode(
                     x=alt.X("shap:Q", title="SHAP value (contribution to ranking score)"),
                     y=alt.Y("factor:N", sort=alt.EncodingSortField("shap", order="descending"),
                             title=None),
                     color=alt.Color("color:N", scale=None),
                     tooltip=["factor", alt.Tooltip("shap:Q", format=".4f")])
                 .properties(height=38 * len(shap_df), title="Tree-SHAP attributions (LightGBM)"))
        fa_right.altair_chart(chart, use_container_width=True)
    except Exception as e:
        # Fallback to simple rubric factor bars
        FACTOR_COLS_FALLBACK = [k for k in [
            "role_target_best", "core_skill_score", "embedding_retrieval_signal",
            "vector_db_signal", "eval_framework_signal", "experience_fit",
            "production_signal", "dense_margin", "must_have_overlap",
            "services_fraction", "honeypot_flag",
        ] if k in feat.columns]
        row_vals = feat.loc[pick]
        fac = pd.DataFrame({
            "factor": [FACTOR_LABELS.get(c2, c2) for c2 in FACTOR_COLS_FALLBACK],
            "value": [float(row_vals.get(c2, 0)) for c2 in FACTOR_COLS_FALLBACK],
        })
        chart = (alt.Chart(fac).mark_bar(cornerRadiusEnd=4, color=PURPLE)
                 .encode(x=alt.X("value:Q", title=None),
                         y=alt.Y("factor:N", sort="-x", title=None),
                         tooltip=["factor", alt.Tooltip("value:Q", format=".3f")])
                 .properties(height=34 * len(fac)))
        fa_right.altair_chart(chart, use_container_width=True)
else:
    # rubric-only mode or candidate not in feature store
    FACTOR_COLS_FALLBACK = [k for k in [
        "role_target_best", "core_skill_score", "embedding_retrieval_signal",
        "vector_db_signal", "eval_framework_signal", "experience_fit",
        "production_signal", "dense_margin", "must_have_overlap",
        "services_fraction", "honeypot_flag",
    ] if k in (feat.columns if hasattr(feat, "columns") else [])]
    if pick in feat.index and FACTOR_COLS_FALLBACK:
        row_vals = feat.loc[pick]
        fac = pd.DataFrame({
            "factor": [FACTOR_LABELS.get(c2, c2) for c2 in FACTOR_COLS_FALLBACK],
            "value": [float(row_vals.get(c2, 0)) for c2 in FACTOR_COLS_FALLBACK],
        })
        chart = (alt.Chart(fac).mark_bar(cornerRadiusEnd=4, color=PURPLE)
                 .encode(x=alt.X("value:Q", title=None),
                         y=alt.Y("factor:N", sort="-x", title=None),
                         tooltip=["factor", alt.Tooltip("value:Q", format=".3f")])
                 .properties(height=34 * len(fac)))
        fa_right.altair_chart(chart, use_container_width=True)
    else:
        fa_right.info("Select a candidate to see factor breakdown.")

st.markdown(
    f'<div class="foot">Recruitment Signal Fusion · Team NullSet · model: '
    f'{"LightGBM (distilled) + rubric" if model_obj else "rubric-only"} · reference date '
    f'{get_reference_now()} · reasoning is fact-grounded, no generative model at rank time · '
    f'<a href="https://github.com/AyushCoder9/recruitment-signal-fusion" style="color:{PURPLE}">GitHub</a></div>',
    unsafe_allow_html=True)
