"""
fig3_fold_error_bars.py
Publication-quality fold-error bar chart for the gold-tier 24-drug benchmark.

Horizontal bar chart of Cmax fold errors, sorted best → worst (left to right).
Colour coding: green ≤2×, orange 2–4×, red >4×.
Dashed reference line at FE = 2.0.
AAFE annotated in corner.

Data source: outputs/benchmark_2026-03-17.json
  AAFE 1.793 [95% CI 1.52, 2.18], %2-fold 83% (N=24)

Output:
  fig3_fold_error_bars.pdf
  fig3_fold_error_bars.png  (300 DPI)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — benchmark_2026-03-17.json, sorted best→worst
# ---------------------------------------------------------------------------

BENCHMARK_RAW = [
    ("caffeine", 1.377),
    ("metoprolol", 1.919),
    ("midazolam", 1.978),
    ("propranolol", 1.452),
    ("warfarin", 6.735),
    ("d-amphetamine", 1.890),
    ("ibuprofen", 5.394),
    ("acetaminophen", 1.287),
    ("amoxicillin", 1.643),
    ("atorvastatin", 1.772),
    ("carbamazepine", 1.693),
    ("diazepam", 1.968),
    ("digoxin", 1.060),
    ("fluoxetine", 1.415),
    ("nifedipine", 1.081),
    ("omeprazole", 1.830),
    ("phenytoin", 1.531),
    ("theophylline", 1.122),
    ("verapamil", 1.483),
    ("atenolol", 1.371),
    ("fluconazole", 3.253),
    ("furosemide", 2.652),
    ("gabapentin", 1.536),
    ("metformin", 1.370),
]

AAFE = 1.793
AAFE_CI_LO = 1.52
AAFE_CI_HI = 2.18
PCT_2FOLD = 83.3

# Sort best → worst
BENCHMARK = sorted(BENCHMARK_RAW, key=lambda x: x[1])

drugs = [r[0] for r in BENCHMARK]
fe = np.array([r[1] for r in BENCHMARK])

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


def _color(f):
    if f <= 2.0:
        return "#2ca02c"
    elif f <= 4.0:
        return "#ff7f0e"
    else:
        return "#d62728"


bar_colors = [_color(f) for f in fe]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.13)

y_pos = np.arange(len(drugs))

bars = ax.barh(
    y_pos,
    fe,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    height=0.65,
    zorder=3,
)

# 2-fold acceptability line
ax.axvline(
    2.0,
    color="#555555",
    linestyle="--",
    linewidth=1.3,
    zorder=2,
    label="2-fold acceptability threshold",
)

# Annotate each bar with its FE value
for bar, f in zip(bars, fe):
    label_x = f + 0.05
    ax.text(
        label_x,
        bar.get_y() + bar.get_height() / 2,
        f"{f:.2f}×",
        va="center",
        ha="left",
        fontsize=8.5,
        color="#222222",
    )

# AAFE summary box (top-left)
textstr = (
    f"AAFE = {AAFE:.3f}\n"
    f"95% CI [{AAFE_CI_LO:.2f}, {AAFE_CI_HI:.2f}]\n"
    f"{PCT_2FOLD:.0f}% within 2-fold"
)
ax.text(
    0.98,
    0.03,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
)

ax.set_yticks(y_pos)
ax.set_yticklabels([d.capitalize() for d in drugs], fontsize=9.5)

ax.set_xlabel("Cmax fold error  (max(pred/obs, obs/pred))", fontsize=10)
ax.set_xlim(0.0, 8.5)
ax.set_ylim(-0.6, len(drugs) - 0.4)
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)

ax.set_title(
    "Per-Drug Cmax Fold Error — Gold Tier (N=24)",
    fontsize=11,
    fontweight="bold",
    loc="left",
    pad=8,
)

# Legend
green_patch = mpatches.Patch(color="#2ca02c", label="≤ 2-fold")
orange_patch = mpatches.Patch(color="#ff7f0e", label="2–4-fold")
red_patch = mpatches.Patch(color="#d62728", label="> 4-fold")
ax.legend(
    handles=[green_patch, orange_patch, red_patch],
    fontsize=9,
    loc="lower right",
    framealpha=0.90,
    title="Fold error",
    title_fontsize=8.5,
)

fig.suptitle(
    "Omega PBPK  |  Gold-tier benchmark  (2026-03-17)\n"
    r"$^\dagger$Bootstrap 95% CI from N=10,000 resamples",
    fontsize=9.5,
    y=0.99,
    color="#444444",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_DIR = "/home/jam/Omega/docs/paper/figures"
pdf_path = f"{OUT_DIR}/fig3_fold_error_bars.pdf"
png_path = f"{OUT_DIR}/fig3_fold_error_bars.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
