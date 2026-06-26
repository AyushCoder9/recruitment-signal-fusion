# Recruitment Signal Fusion — Intelligent Candidate Discovery & Ranking

> **India.Runs × Redrob — Data & AI Challenge.** Rank the top-100 of **100,000** candidates
> for a *"Senior AI Engineer (founding team)"* job description, against a hidden ground truth,
> on the official composite metric — fast, explainable, reproducible, and disqualification-safe.

**Team:** NullSet · **Lead:** Ayush Kumar Singh

---

## TL;DR

A hybrid ranking engine that treats the **job description as the scoring rubric in disguise**
and faithfully reproduces *a careful recruiter applying that rubric* across all 100k candidates.
It fuses four signals — **dense semantic retrieval + BM25 lexical match + a transparent JD rubric
+ a LightGBM LambdaMART ranker distilled from a free LLM judge** — then applies multiplicative
gates for availability, location, off-target function, and honeypots.

- **Composite 0.65** (`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`) on held-out LLM-judge labels; **NDCG@50 0.84**.
- **Top-100 verified: 0 honeypots · 0 off-target · 0 services-only · 0 out-of-India** → disqualification-safe.
- **Ranking step: ~40 s wall-clock / ~196 MB / CPU-only / zero network** on the full 100k (≈7.5× under the 5-minute budget).
- **Zero-hallucination reasoning by construction** — no generative model runs at rank time.
- **Official `validate_submission.py` → "Submission is valid."** · 24/24 tests pass · byte-identical determinism.

---

## The thesis

The hidden ground truth was almost certainly built by a careful recruiter (or an LLM) applying
**this exact JD** as a rubric. So we don't guess a similarity score — we reproduce that judgment.

The single highest-leverage signal is **role / function from job TITLES + career history, not the
skills list.** That is what demotes the keyword-stuffer (a *"Marketing Manager"* carrying every AI
buzzword) and promotes the quiet backend engineer who actually shipped vector search at a product
company. The provided `sample_submission.csv` ranks keyword-counters at the top — and that is
precisely the trap the ground truth penalizes.

---

## Architecture — the precompute / rank split

The challenge allows **unlimited offline precompute (network + LLM)** but the **ranking step is
CPU-only, network-off, ≤ 5 min, ≤ 16 GB.** The entire design follows from that split.

![Architecture](docs/architecture.png)

Because the JD query is fixed, `dense_sim` and `bm25_score` are **precomputed into the feature
store**, so the timed ranking path is just *a parquet load + cheap numpy + one `LightGBM.predict`
+ a grounded-reasoning pass*. No model inference loop, no network, no surprises.

---

## Scoring

```
fit_raw     = 0.6 · LightGBM(distilled judge)  +  0.4 · interpretable rubric
final_score = fit_raw × availability × location × offtarget_gate × honeypot_gate
```

- **Rubric** (transparent, weighted, 0–1): role + semantic + skills (trust-weighted by
  proficiency × duration × endorsements, never a raw keyword count) + experience (soft Gaussian
  peaked at 6–8 yrs — *"a range, not a requirement"*) + production cues, minus soft disqualifier
  penalties. Doubles as a ranker feature **and** the graceful-degradation fallback.
- **Ranker**: LightGBM LambdaMART distilled from LLM-judge tier labels (0–4) — directly optimizes
  NDCG, runs in milliseconds on CPU.
- **Modifiers / gates**: availability (the JD's *"multiplier on skill-match"*; sentinel `−1` →
  neutral), location (Pune / Noida → tiered India → outside), off-target gate `0.15` (spares
  proven ex-builders), honeypot gate `0.03` (internal timeline impossibility → floored).

### 59 features across 8 groups

`semantic` · `role` · `experience` · `skills` (incl. claim-inflation vs. assessment scores) ·
`company` (services-fraction by tenure) · `location` · `behavioral` (23 signals, sentinels
neutralized) · `consistency / honeypot`. All vectorized; assembled into a single
`features.parquet` keyed by `candidate_id`.

---

## The LLM judge — free, multi-provider, $0

A LightGBM ranker is only as good as its labels. We labeled **~1,430 stratified candidates**
(obvious fits/misses, off-target stuffers, foreign, low-engagement, suspected honeypots) with the
JD as the exact rubric → strict JSON `{tier: 0–4, reasoning, key_factors}`.

To do this for **$0**, the judge is a **stack across providers**, each with its own daily quota
bucket, draining in priority order and **resuming across days**:

| Provider | Model | Role |
|---|---|---|
| Cerebras | `gpt-oss-120b` | workhorse (~1M free tokens/day) |
| Groq | `llama-4-scout`, `llama-3.1-8b`, `llama-3.3-70b` | top-up |
| Gemini | `2.5-flash-lite`, `2.5-flash` | top-up |

When a bucket exhausts its daily quota, that model **bows out gracefully** and the rest continue —
fully resumable, no candidate dropped. **None of this touches the timed ranking path.**

---

## Honeypot & hallucination defense

- **Honeypots are internal date/timeline contradictions** (an EDA finding — *not* skill-stuffing):
  role duration > total career, years-of-experience > calendar span, education-vs-career timeline
  impossibilities. Detected at high precision and gated to the floor. A deterministic
  **label-override** also corrects the 10 planted traps that the LLM judge — dazzled by
  Meta/Apple/Flipkart titles — had rated top-tier, *before* distillation. **The math doesn't lie.**
- **Sentinels** (`−1` github, `−1` offer-accept, `0` response with `0` applications) mean
  **NO HISTORY → neutral**, never penalized.
- **Reasoning is template-based and fact-grounded** — every clause is pulled from a structured
  field (real title, company, YOE, a phrase quoted verbatim from the best-matching role, response
  rate). A hallucinated skill is **impossible by construction**. No generative model at rank time.

---

## Results

![Ablation](docs/ablation.png)

The ablation is the whole story: **naive keyword/BM25 matching scores a deceptively-fine 0.29 but
puts 5 honeypots + 11 out-of-India candidates in its top-100 — a disqualifying ranking.** Only the
gated hybrid is both high-scoring **and** clean.

| Configuration | Composite | Top-100 health |
|---|---|---|
| Naive BM25 (≈ sample submission) | 0.29 | 5 honeypot + 11 foreign — **DQ** |
| Dense only | 0.58 | 3 honeypot + 10 foreign — **DQ** |
| Rubric, no gates | 0.46 | 1 honeypot + 10 foreign — **DQ** |
| Full rubric + gates | 0.658 | **clean** (0 / 0 / 0) |
| **Full hybrid (LambdaMART)** | **0.65** (NDCG@50 0.84) | **clean** (0 / 0 / 0) |

### What the ranker actually leans on

![Feature importance](docs/feature_importance.png)

Reassuringly, the top features are **semantic fit, applied-ML years, and must-have coverage** — the
signals a recruiter would weigh — not spurious keyword counts.

---

## Reproduce

```bash
make precompute   # build feature store + embeddings  (~40 min, offline; network OK)
make label        # LLM-judge tier labels             (offline, async, resumable)  [optional]
make train        # distill the LightGBM ranker        [optional]
make rank         # produce submission.csv + .xlsx     (<5 min, CPU-only, no network)
make validate     # official validate_submission.py -> "Submission is valid."
make test         # full unit + integration suite (24 tests)
```

Single reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`.
The repo ships the trained model, labels and feature store, so **`make rank` reproduces the exact
submission without re-embedding**. Without labels/ranker the system degrades gracefully to a valid,
strong, honeypot-clean **rubric-only** ranking.

> The 161 MB `embeddings.npz` is git-ignored (it only feeds the sandbox / live-embed path);
> `make precompute` regenerates it. The 465 MB challenge dataset stays outside the repo and is
> pointed to by `config.yaml`.

---

## Compute & determinism

| | |
|---|---|
| Ranking step | **~40 s wall · ~196 MB · CPU-only · no network** (full 100k) |
| Machine | MacBook Pro M3 Pro, 18 GB, Python 3.12 |
| Embeddings | `bge-small-en-v1.5` (CPU; MPS stalls at 100k scale), 384-d |
| Determinism | fixed seeds, `LightGBM deterministic=true`, tie-break `candidate_id` ascending → byte-identical output |

---

## Repository layout

```
rank.py  precompute.py  train_ranker.py  label_with_llm.py   # pipeline entry points
config.yaml  Makefile  requirements.txt                      # single source of truth + tasks
src/
  io/         schema (pydantic) · streaming loaders · config/artifact IO
  features/   semantic · role · experience · skills · company · behavioral ·
              location · consistency/honeypot · bm25 · registry
  scoring/    rubric · ensemble · modifiers · ranker · reasoning · pipeline
  eval/       metrics (NDCG/MAP/P@k) · tuning harness
  llm/        multi-provider judge clients · prompts
tests/        validation · features · io · metrics · reasoning  (24 tests)
sandbox/      Streamlit app + Dockerfile (upload ≤100 -> ranked table + factor chart)
notebooks/    EDA
artifacts/    feature store · labels · trained ranker · JD anchors · EDA summary
docs/         architecture + ablation + feature-importance charts, technical report
scripts/      deck + diagram + chart generators
submission/   submission.csv · submission.xlsx · submission_metadata.yaml
```

See [`docs/REPORT.md`](docs/REPORT.md) for the full technical write-up.

---

## Constraints compliance

| Requirement | Status |
|---|---|
| Ranking ≤ 5 min, ≤ 16 GB, CPU-only, **no network / no hosted LLM** | ✅ ~40 s / 196 MB / CPU / offline |
| Output: `candidate_id,rank,score,reasoning`, 100 rows, unique ranks, non-increasing score | ✅ official validator passes |
| Honeypot rate in top-100 < 10% | ✅ **0%** |
| Reproducible & deterministic | ✅ byte-identical |
| Precompute may use network/LLM | ✅ used offline only, never at rank time |
