#!/usr/bin/env python3
"""Render report charts for the submission docs:
  docs/feature_importance.png  — LightGBM gain importance (live from the trained model)
  docs/ablation.png            — composite score across pipeline stages, DQ-flagged
On-brand: Manrope font, Redrob purple / slate.
"""
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FONTS = os.path.join(REPO, "assets", "fonts")
DOCS = os.path.join(REPO, "docs")
for w in ("Regular", "Medium", "SemiBold", "Bold", "ExtraBold"):
    fm.fontManager.addfont(os.path.join(FONTS, f"Manrope-{w}.ttf"))
plt.rcParams["font.family"] = "Manrope Regular"
SB, XB, BD = "Manrope SemiBold", "Manrope ExtraBold", "Manrope Bold"

PURPLE = "#7d45e0"
DARK = "#202729"
GREEN = "#1f9d6b"
RED = "#d23a55"
GREY = "#9aa0a6"


def feature_importance():
    model = lgb.Booster(model_file=os.path.join(REPO, "artifacts", "lgbm_ranker.txt"))
    names = model.feature_name()
    gains = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(names, gains), key=lambda x: x[1], reverse=True)[:15][::-1]
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=200)
    bars = ax.barh(range(len(vals)), vals, color=PURPLE, edgecolor="none", height=0.7)
    bars[-1].set_color(DARK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontfamily=SB, color=DARK)
    ax.set_xlabel("LightGBM split gain", fontsize=10.5, fontfamily=SB, color=DARK)
    ax.set_title("What the ranker actually leans on (top-15 by gain)",
                 fontsize=13.5, fontfamily=XB, color=DARK, pad=14, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=GREY)
    fig.tight_layout()
    out = os.path.join(DOCS, "feature_importance.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def ablation():
    # Held-out composite on the LLM-judge labels for each pipeline stage, with the count of
    # honeypots + out-of-India candidates that land in the top-100 (a >10% honeypot rate = DQ).
    stages = [
        ("Naive BM25\n(≈ sample sub)", 0.29, "5 honeypot + 11 foreign", True),
        ("Dense only", 0.58, "3 honeypot + 10 foreign", True),
        ("Rubric,\nno gates", 0.46, "1 honeypot + 10 foreign", True),
        ("Full rubric\n+ gates", 0.658, "clean top-100", False),
        ("Full hybrid\n(LambdaMART)", 0.65, "clean top-100", False),
    ]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    notes = [s[2] for s in stages]
    dq = [s[3] for s in stages]
    colors = [RED if d else GREEN for d in dq]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="none", width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5, fontfamily=SB, color=DARK)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Composite score", fontsize=10.5, fontfamily=SB, color=DARK)
    ax.set_title("Ablation: high keyword scores are a trap — only the gated hybrid is clean",
                 fontsize=13, fontfamily=XB, color=DARK, pad=14, loc="left")
    for i, (b, v, n, d) in enumerate(zip(bars, vals, notes, dq)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontfamily=XB,
                color=RED if d else GREEN)
        ax.text(b.get_x() + b.get_width() / 2, 0.03, n, ha="center", va="bottom",
                fontsize=8, fontfamily=SB, color="white", rotation=0)
    ax.text(0.0, -0.16, "Red = disqualifying ranking (honeypots / out-of-India in top-100)   ·   "
            "Green = top-100 verified 0 honeypot / 0 off-target / 0 foreign",
            transform=ax.transAxes, fontsize=8.5, fontfamily=SB, color=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=GREY)
    fig.tight_layout()
    out = os.path.join(DOCS, "ablation.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    feature_importance()
    ablation()
