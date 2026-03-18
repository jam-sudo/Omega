"""
fig6_multi_tier_summary.py
Publication-quality multi-tier AAFE summary bar chart.

Horizontal bar chart with 95% CI whiskers for all validation tiers.
Reference line at AAFE = 2.0.

Tier values (2026-03-17):
  Gold (N=24)             Cmax AAFE 1.793 [1.52, 2.18],  83% 2-fold
  Gold (N=24)             AUC  AAFE 1.960 [1.56, 2.57],  63% 2-fold
  Expanded Gold (N=51)    Cmax AAFE 3.40  [2.42, 4.94],  39% 2-fold
  Temporal Full (N=15)    Cmax AAFE 2.66  [1.74, 4.35],  47% 2-fold
  Temporal In-Scope (N=10) Cmax AAFE 1.64 [1.33, 2.11],  70% 2-fold
  Silver (N=39)           t½   AAFE 2.42,                 —
  External (N=8)          Cmax AAFE 2.95,                 62% 2-fold

Output:
  fig6_multi_tier_summary.pdf
  fig6_multi_tier_summary.png  (300 DPI)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data
# (label, metric_type, N, AAFE, ci_lo, ci_hi, pct_2fold_or_None)
# ci_lo / ci_hi = None when no bootstrap CI available
# ---------------------------------------------------------------------------

TIERS = [
    # Core gold
    ("Core Gold — Cmax (N=24)", "Cmax", 24, 1.513, 1.33, 1.74, 88),
    ("Core Gold — AUC (N=24)", "AUC", 24, 1.932, 1.54, 2.59, 62),
    # Expanded gold
    ("Expanded Gold — Cmax (N=51)", "Cmax", 51, 3.40, 2.42, 4.94, 39),
    # Temporal
    ("Temporal Holdout Full — Cmax (N=15)", "Cmax", 15, 2.66, 1.74, 4.35, 47),
    ("Temporal In-Scope — Cmax (N=10)", "Cmax", 10, 1.64, 1.33, 2.11, 70),
    # Silver
    ("Silver — t½ (N=39)", "t½", 39, 2.42, None, None, None),
    # External
    ("External Validation — Cmax (N=8)", "Cmax", 8, 2.95, None, None, 62),
]

# Color scheme by tier category
COLORS = {
    "Core Gold — Cmax (N=24)": "#2166ac",
    "Core Gold — AUC (N=24)": "#4393c3",
    "Expanded Gold — Cmax (N=51)": "#74add1",
    "Temporal Holdout Full — Cmax (N=15)": "#f46d43",
    "Temporal In-Scope — Cmax (N=10)": "#fdae61",
    "Silver — t½ (N=39)": "#878787",
    "External Validation — Cmax (N=8)": "#d73027",
}

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

fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.05, right=0.82, top=0.88, bottom=0.13)

labels = [t[0] for t in TIERS]
aafe = np.array([t[3] for t in TIERS])
ci_lo_arr = np.array([t[4] if t[4] is not None else t[3] for t in TIERS])
ci_hi_arr = np.array([t[5] if t[5] is not None else t[3] for t in TIERS])
pct_2f = [t[6] for t in TIERS]
bar_colors = [COLORS[lab] for lab in labels]

y_pos = np.arange(len(TIERS))
xerr_lo = aafe - ci_lo_arr
xerr_hi = ci_hi_arr - aafe

bars = ax.barh(
    y_pos,
    aafe,
    xerr=[xerr_lo, xerr_hi],
    color=bar_colors,
    edgecolor="white",
    linewidth=0.6,
    height=0.55,
    capsize=4,
    error_kw={"elinewidth": 1.4, "ecolor": "#444444", "capthick": 1.4},
    zorder=3,
)

# Reference lines
ax.axvline(
    2.0,
    color="#555555",
    linestyle="--",
    linewidth=1.3,
    zorder=2,
    label="AAFE = 2.0  (acceptability threshold)",
)

# Annotate bars
for _i, (bar, a, p, ci_h) in enumerate(zip(bars, aafe, pct_2f, xerr_hi)):
    # Build annotation string
    parts = [f"{a:.2f}"]
    if p is not None:
        parts.append(f"({p}% 2-fold)")
    annotation = "  ".join(parts)
    ax.text(
        a + ci_h + 0.05,
        bar.get_y() + bar.get_height() / 2,
        annotation,
        va="center",
        ha="left",
        fontsize=8.5,
        color="#333333",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()  # most tuned at top

ax.set_xlabel("AAFE (average absolute fold error, lower is better)", fontsize=10)
ax.set_xlim(0, 8.0)
ax.set_ylim(-0.6, len(TIERS) - 0.4)
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

ax.set_title(
    "Multi-Tier Validation Summary — Omega PBPK",
    fontsize=11,
    fontweight="bold",
    loc="left",
    pad=8,
)

# Legend patches
legend_handles = [
    mpatches.Patch(color="#2166ac", label="Core gold (N=24)"),
    mpatches.Patch(color="#74add1", label="Expanded gold (N=51)"),
    mpatches.Patch(color="#f46d43", label="Temporal holdout"),
    mpatches.Patch(color="#878787", label="Silver (t½ only)"),
    mpatches.Patch(color="#d73027", label="External validation"),
]
ax.legend(
    handles=legend_handles,
    fontsize=8.5,
    bbox_to_anchor=(1.01, 0.5),
    loc="center left",
    framealpha=0.90,
    title="Tier category",
    title_fontsize=8.5,
    borderaxespad=0,
)

fig.suptitle(
    "Omega PBPK  |  Validation benchmark  (2026-03-17)\n"
    r"$^\dagger$Error bars = 95% bootstrap CI where available;  no CI bar = single-run estimate",
    fontsize=9.5,
    y=0.99,
    color="#444444",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_DIR = "/home/jam/Omega/docs/paper/figures"
pdf_path = f"{OUT_DIR}/fig6_multi_tier_summary.pdf"
png_path = f"{OUT_DIR}/fig6_multi_tier_summary.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
