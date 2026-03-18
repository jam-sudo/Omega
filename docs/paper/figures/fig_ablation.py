"""
fig_ablation.py
Publication-quality ablation figure for Omega PBPK paper.

Shows the impact of each pipeline component by running the 8-config ablation
and plotting a horizontal bar chart sorted by AAFE (worst to best).

Ablation results (from scripts/run_ablation.py, 2026-03-17):
  FULL:         AAFE 1.793  [1.52, 2.18]  %2-fold 83%
  NO_ENSEMBLE:  AAFE 1.875  [1.62, 2.21]  %2-fold 67%
  NO_RIDGE:     AAFE 1.793  [1.52, 2.18]  %2-fold 83%
  NO_VDSS:      AAFE 1.836  [1.53, 2.25]  %2-fold 75%   (XGBoost VDss predictor)
  NO_ENS_RIDGE: AAFE 1.875  [1.62, 2.21]  %2-fold 67%
  NO_HYBRID:    AAFE 2.346  [1.87, 2.96]  %2-fold 46%
  NO_PGP:       AAFE 1.818  [1.54, 2.21]  %2-fold 79%
  BARE:         AAFE 2.465  [1.96, 3.09]  %2-fold 42%

Output:
  fig_ablation.pdf
  fig_ablation.png  (300 DPI)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data (from ablation run 2026-03-17)
# ---------------------------------------------------------------------------

# (internal_name, display_label, aafe, ci_lo, ci_hi, pct_2fold)
configs_raw = [
    ("BARE", "Raw ODE (no corrections)", 2.465, 1.96, 3.09, 42),
    ("NO_HYBRID", "No hybrid Cmax selector", 2.346, 1.87, 2.96, 46),
    ("NO_ENS_RIDGE", "No ensemble + no ridge", 1.875, 1.62, 2.21, 67),
    ("NO_ENSEMBLE", "No PBPK/ML ensemble", 1.875, 1.62, 2.21, 67),
    ("NO_VDSS", "No XGBoost VDss predictor", 1.836, 1.53, 2.25, 75),
    ("NO_PGP", "No P-gp correction", 1.818, 1.54, 2.21, 79),
    ("NO_RIDGE", "No ridge correction", 1.793, 1.52, 2.18, 83),
    ("FULL", "Full pipeline (baseline)", 1.793, 1.52, 2.18, 83),
]

# Sort worst-to-best so the chart reads top=worst, bottom=best
configs = sorted(configs_raw, key=lambda x: x[2], reverse=True)

labels = [c[1] for c in configs]
names = [c[0] for c in configs]
aafe = np.array([c[2] for c in configs])
ci_lo = np.array([c[3] for c in configs])
ci_hi = np.array([c[4] for c in configs])
pct_2fold = [c[5] for c in configs]

full_aafe = 1.793  # baseline
delta = aafe - full_aafe  # positive = worse without component

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
#  FULL    → green  (baseline)
#  NEUTRAL → same AAFE as full (NO_RIDGE) → light grey
#  Others  → orange-to-red gradient by degradation magnitude


def _delta_to_color(name, d):
    if name == "FULL":
        return "#2ca02c"  # green
    if abs(d) < 0.005:
        return "#aaaaaa"  # neutral grey (NO_RIDGE)
    # Normalise delta to [0,1] over observed range [0.02, 0.68]
    t = min(max((d - 0.02) / (0.68 - 0.02), 0.0), 1.0)
    # Interpolate orange (0.99, 0.55, 0.24) → dark red (0.60, 0.05, 0.05)
    r = 0.99 + (0.60 - 0.99) * t
    g = 0.55 + (0.05 - 0.55) * t
    b = 0.24 + (0.05 - 0.24) * t
    return (r, g, b)


bar_colors = [_delta_to_color(n, d) for n, d in zip(names, delta)]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.subplots_adjust(left=0.29, right=0.82, top=0.88, bottom=0.13)

n = len(configs)
y_pos = np.arange(n)
xerr_lo = aafe - ci_lo
xerr_hi = ci_hi - aafe

bars = ax.barh(
    y_pos,
    aafe,
    xerr=[xerr_lo, xerr_hi],
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    height=0.60,
    capsize=4,
    error_kw={"elinewidth": 1.3, "ecolor": "#444444", "capthick": 1.3},
    zorder=3,
)

# Baseline dashed line
ax.axvline(
    full_aafe,
    color="#2ca02c",
    linestyle="--",
    linewidth=1.5,
    zorder=2,
    label=f"Full pipeline AAFE = {full_aafe:.3f}",
)

# 2-fold acceptability line
ax.axvline(
    2.0,
    color="#888888",
    linestyle=":",
    linewidth=1.1,
    zorder=2,
    label="AAFE = 2.0 (acceptability threshold)",
)

# Annotate each bar: AAFE value + Δ + %2-fold
# Place annotation to the right of the CI whisker; clip at x_max so it stays visible
x_max = 4.0  # matches ax.set_xlim upper
for _i, (bar, a, p, d, ci_h) in enumerate(zip(bars, aafe, pct_2fold, delta, xerr_hi)):
    if d > 0.005:
        annotation = f"{a:.3f}  (Δ+{d:.3f})   {p}% 2-fold"
    elif abs(d) < 0.005:
        annotation = f"{a:.3f}  (Δ=0)   {p}% 2-fold"
    else:
        annotation = f"{a:.3f}  (Δ{d:+.3f})   {p}% 2-fold"

    label_x = a + ci_h + 0.05
    ax.text(
        label_x,
        bar.get_y() + bar.get_height() / 2,
        annotation,
        va="center",
        ha="left",
        fontsize=8.5,
        color="#222222",
        clip_on=False,
    )

# Y axis
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()  # worst at top, best at bottom

# X axis
ax.set_xlabel("Cmax AAFE (average absolute fold error, lower is better)", fontsize=10)
ax.set_xlim(0.0, 4.0)
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)

# Title
ax.set_title(
    "Ablation Study: Contribution of Pipeline Components",
    fontsize=12,
    fontweight="bold",
    loc="left",
    pad=10,
)

# Subtitle / caption
fig.text(
    0.33,
    0.01,
    "24-drug gold-tier benchmark (N=24).  Error bars = 95% bootstrap CI (N=10,000 resamples).  "
    "Δ = AAFE change vs full pipeline (positive = component helps).",
    ha="left",
    va="bottom",
    fontsize=8,
    color="#555555",
    style="italic",
)

# Legend
ax.legend(fontsize=9, loc="lower right", framealpha=0.90)

# Color legend for bar shading
green_patch = mpatches.Patch(color="#2ca02c", label="Full pipeline (baseline)")
grey_patch = mpatches.Patch(color="#aaaaaa", label="Neutral (no effect)")
orange_patch = mpatches.Patch(color=(0.99, 0.55, 0.24), label="Moderate degradation")
red_patch = mpatches.Patch(color=(0.60, 0.05, 0.05), label="Severe degradation")

leg2 = ax.legend(
    handles=[green_patch, grey_patch, orange_patch, red_patch],
    fontsize=8.5,
    bbox_to_anchor=(1.01, 0.5),
    loc="center left",
    framealpha=0.90,
    title="Component contribution",
    title_fontsize=8.5,
    borderaxespad=0,
)
# leg2 goes in the figure's right margin (clip_on=False ensures it renders)
ax.add_artist(leg2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_dir = "/home/jam/Omega/docs/paper/figures"
pdf_path = f"{out_dir}/fig_ablation.pdf"
png_path = f"{out_dir}/fig_ablation.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
