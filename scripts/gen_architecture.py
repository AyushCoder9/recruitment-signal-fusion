#!/usr/bin/env python3
"""Render the system-architecture diagram (precompute / rank split) as a crisp PNG for the
submission deck. On-brand: Manrope font, Redrob purple, dark slate."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FONTS = os.path.join(REPO, "assets", "fonts")
for w in ("Regular", "Medium", "SemiBold", "Bold", "ExtraBold"):
    fm.fontManager.addfont(os.path.join(FONTS, f"Manrope-{w}.ttf"))
plt.rcParams["font.family"] = "Manrope Regular"
SB, XB, BD = "Manrope SemiBold", "Manrope ExtraBold", "Manrope Bold"

PURPLE = "#7d45e0"
DARK = "#202729"
SLATE = "#32363f"
LILAC = "#efeafc"
INK = "#1b1530"
GREEN = "#1f9d6b"
RED = "#d23a55"

fig, ax = plt.subplots(figsize=(13.0, 6.0), dpi=300)
ax.set_xlim(0, 130)
ax.set_ylim(0, 60)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, tc="white", fs=10.5, family=SB, round=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={round}",
                                linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontfamily=family, zorder=3, linespacing=1.25)


def arrow(x1, y1, x2, y2, color=PURPLE, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, zorder=1, shrinkA=2, shrinkB=2))


# ---- lane labels ----
ax.add_patch(FancyBboxPatch((1, 40.5), 128, 18.5, boxstyle="round,pad=0.1,rounding_size=0.8",
             facecolor="#faf8ff", edgecolor="#e3d8fb", lw=1.2, zorder=0))
ax.text(3.2, 56.6, "PRECOMPUTE  ·  offline, unlimited (network + LLM allowed)",
        color=PURPLE, fontsize=11, fontfamily=XB)

ax.add_patch(FancyBboxPatch((1, 1.5), 128, 16.5, boxstyle="round,pad=0.1,rounding_size=0.8",
             facecolor="#f3f9f7", edgecolor="#cfe9df", lw=1.2, zorder=0))
ax.text(3.2, 15.6, "RANK  ·  the timed step — CPU-only, no network, ≤ 5 min, ≤ 16 GB",
        color=GREEN, fontsize=11, fontfamily=XB)

# ---- precompute row ----
y = 44.5
h = 8.5
xs = [3.5, 24, 44.5, 65, 85.5, 106]
w = 18
labels = ["100k JSONL\n(streaming parse)", "Embed all\nBGE-small · CPU", "BM25\nindex",
          "59-feature\nstore", "LLM-judge\nlabels (free stack)", "Train\nLambdaMART"]
fcs = [SLATE, PURPLE, SLATE, PURPLE, INK, PURPLE]
for x, lab, fc in zip(xs, labels, fcs):
    box(x, y, w, h, lab, fc, fc, fs=9.2)
for i in range(len(xs) - 1):
    arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

# ---- artifact store (center bridge) ----
box(30, 27.5, 70, 7.5, "Artifact store   ·   embeddings.npz · jd_query.npz · bm25.pkl · "
    "features.parquet · labels.parquet · lgbm_ranker.txt",
    LILAC, "#cdb8f4", tc=INK, fs=8.8, family=BD)
arrow(40, y, 50, 35, color="#b79cef")       # train/precompute -> store
arrow(95, y, 90, 35, color="#b79cef")
arrow(50, 27.5, 45, 18, color="#9ad0bf")     # store -> rank
arrow(80, 27.5, 86, 18, color="#9ad0bf")

# ---- rank row ----
y2 = 5.5
xs2 = [3.5, 26, 49.5, 73, 96.5]
labels2 = ["Load artifacts\n+ query features", "Blend\nranker + rubric",
           "Gates + modifiers\navailability · location\noff-target · honeypot",
           "Sort + tie-break\ntop-100", "Grounded reasoning\n→ .csv / .xlsx"]
fcs2 = [SLATE, PURPLE, RED, SLATE, GREEN]
w2 = 20.5
for x, lab, fc in zip(xs2, labels2, fcs2):
    box(x, y2, w2, h, lab, fc, fc, fs=8.6)
for i in range(len(xs2) - 1):
    arrow(xs2[i] + w2, y2 + h / 2, xs2[i + 1], y2 + h / 2, color=GREEN)

out = os.path.join(REPO, "artifacts", "architecture.png")
plt.savefig(out, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("wrote", out)
