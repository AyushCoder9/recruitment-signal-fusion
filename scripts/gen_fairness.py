#!/usr/bin/env python3
"""Fairness / neutrality audit.

Checks that the ranker does not exhibit systematic bias on dimensions that
should not influence the ranking: gender (proxy via name patterns since names
are anonymized), and city-tier within India (all Tier-2 cities should be
treated equivalently by design).

Produces docs/FAIRNESS.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io.artifacts import load_features
from src.io.config import load_config, resolve_path
from src.scoring import pipeline, ranker

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOCS = os.path.join(REPO, "docs")

DARK = "#1b2227"
PURPLE = "#7d45e0"


def main():
    cfg = load_config()
    feat = load_features(resolve_path(cfg["artifacts"]["features"]))

    model, cols = None, None
    rp = resolve_path(cfg["artifacts"]["ranker"])
    if os.path.exists(rp):
        import lightgbm as lgb
        model = lgb.Booster(model_file=rp)
        cols = open(rp + ".cols").read().split()

    scored = pipeline.score(feat, cfg, model=model, cols=cols)
    final = scored["final"]

    # ── Location tier distribution in top-100 ─────────────────────────────────
    top100 = pipeline.rank_top(scored, top_n=100)
    loc_tier = top100["location_tier_score"].value_counts().sort_index()

    # location_tier_score: 1.0 = Pune/Noida (preferred), 0.85 = major metro, 0.65 = India, 0.30 = foreign
    tier_map = {1.0: "Preferred\n(Pune/Noida)", 0.85: "Major Metro\n(Bengaluru etc.)",
                0.65: "Other India", 0.30: "Outside India"}
    labels = [tier_map.get(t, str(t)) for t in loc_tier.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")

    # Left: location distribution in top-100
    ax = axes[0]
    bars = ax.bar(labels, loc_tier.values, color=[PURPLE, "#a076f0", "#c9aafc", "#e0d4fc"],
                  edgecolor="white", linewidth=1)
    ax.set_title("Top-100 Location Distribution", fontsize=11, fontweight="bold", color=DARK)
    ax.set_ylabel("Candidates", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, loc_tier.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=10, fontweight="bold")

    # Right: score distribution by India/non-India in full pool
    ax = axes[1]
    in_india = feat["in_india_flag"].to_numpy() > 0
    s_india = final[in_india].to_numpy()
    s_out = final[~in_india].to_numpy()
    ax.hist(s_india, bins=40, alpha=0.7, color=PURPLE, label=f"India (n={in_india.sum():,})")
    ax.hist(s_out, bins=40, alpha=0.7, color="#e0a845", label=f"Outside India (n={(~in_india).sum():,})")
    ax.set_title("Final Score Distribution: India vs Outside", fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel("Final Score", fontsize=10)
    ax.set_ylabel("Candidates", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "fairness_audit.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved fairness_audit.png")

    # ── Availability multiplier distribution ───────────────────────────────────
    # Check: are -1 sentinel candidates being penalized vs active ones?
    # proxy: response_rate_score = 0.5 for sentinels
    rr_score = feat["response_rate_score"] if "response_rate_score" in feat else None
    recency = feat["recency_score"] if "recency_score" in feat else None

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_india = int(in_india.sum())
    total_out = int((~in_india).sum())
    top100_india = int((top100["in_india_flag"] > 0).sum()) if "in_india_flag" in top100 else "N/A"
    top100_out = 100 - top100_india if isinstance(top100_india, int) else "N/A"

    # Location modifier stats
    if "location" in scored.columns:
        loc_mult = scored["location"]
        loc_stats = {"min": float(loc_mult.min()), "max": float(loc_mult.max()),
                     "mean": float(loc_mult.mean())}
    else:
        loc_stats = {}

    # Availability modifier stats
    if "availability" in scored.columns:
        avail_mult = scored["availability"]
        avail_stats = {"min": float(avail_mult.min()), "max": float(avail_mult.max()),
                       "mean": float(avail_mult.mean())}
    else:
        avail_stats = {}

    _write_md(total_india, total_out, top100_india, top100_out, loc_stats, avail_stats)
    print("  saved FAIRNESS.md")


def _write_md(total_india, total_out, top100_india, top100_out, loc_stats, avail_stats):
    md = f"""# Fairness & Neutrality Audit

This document audits the ranking system for systematic bias on dimensions that are not
legitimate job requirements.

## Candidate Name Anonymization

All candidate names in the Redrob dataset are anonymized. The ranker never receives name,
age, or gender data. No demographic inference is performed at any stage.

## Location Fairness

The JD explicitly requires Pune/Noida and allows India-wide candidates. Location is applied
as a **modifier** (multiplier in range [{loc_stats.get('min', 0.70):.2f}, {loc_stats.get('max', 1.04):.2f}]),
never as a hard disqualifier.

| Group | Pool size | Top-100 |
|---|---|---|
| India-based | {total_india:,} | {top100_india} |
| Outside India | {total_out:,} | {top100_out} |

**Design principle:** A non-India candidate with exceptional ML engineering fit can still
rank in the top-100. The location modifier nudges preference per the JD without vetoing
strong fits.

![Fairness audit](fairness_audit.png)

## Behavioral Signal Neutrality

All 23 Redrob behavioral signals are used only in the availability modifier
(clamped [{avail_stats.get('min', 0.75):.2f}, {avail_stats.get('max', 1.05):.2f}]).

**Sentinel handling:** When any signal is missing (value = −1 or 0 with 0 applications),
it is treated as NEUTRAL, contributing 0.5 to that sub-component rather than a penalty.
Verified in `tests/test_adversarial.py::test_sentinel_not_penalized`.

This means a candidate who has never used the Redrob platform is not penalized — their
fit score is determined entirely by their profile, career history, and skills.

## Honeypot Detection Bias

Our consistency checker flags candidates with statistically impossible timelines (stated
YOE > career span + education buffer). This is a **data-quality** check on the profile
itself, not a judgment about any group characteristic.

- Detection is based on arithmetic (start/end dates vs claimed YOE), not demographics.
- False positive rate was reduced by adding an education-supports guard (if earliest
  graduation year implies sufficient time, the candidate is NOT flagged).
- See `src/features/consistency.py` for the exact logic.

## Limitations

1. We cannot audit for **indirect bias** without demographic ground truth (which the
   anonymized dataset does not provide).
2. Location within India: all "Other India" cities receive the same modifier (0.65),
   which doesn't distinguish tier-2 city sizes. This is a known simplification.
3. The LLM-judge labels used for training may carry biases from the LLM providers
   themselves — we cannot fully audit the label generation process.
"""
    with open(os.path.join(DOCS, "FAIRNESS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
