"""
fig1_pred_vs_obs_cmax.py
Publication-quality pred-vs-obs Cmax scatter plot for the gold-tier 24-drug benchmark.

Panel: log10(pred_cmax) vs log10(obs_cmax)
 - 1:1 line (solid grey)
 - 2-fold dashed lines
 - Points coloured: green (≤2×), orange (2–4×), red (>4×)
 - Worst outliers (FE > 3×) annotated with drug name

Data source: outputs/benchmark_2026-03-18.json
  AAFE 1.513 [95% CI 1.33, 1.74], %2-fold 88% (N=24)

Output:
  fig1_pred_vs_obs_cmax.pdf
  fig1_pred_vs_obs_cmax.png  (300 DPI)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — benchmark_2026-03-17.json
# ---------------------------------------------------------------------------

BENCHMARK = [
    ("caffeine", 1.5507, 2.6035, 1.678),
    ("metoprolol", 0.1084, 0.0663, 1.635),
    ("midazolam", 0.0069, 0.0275, 3.986),
    ("propranolol", 0.0551, 0.0822, 1.492),
    ("warfarin", 0.6764, 1.2783, 1.889),
    ("d-amphetamine", 0.0953, 0.0504, 1.891),
    ("ibuprofen", 19.064, 19.000, 1.003),
    ("acetaminophen", 14.245, 15.180, 1.066),
    ("amoxicillin", 6.983, 7.500, 1.074),
    ("atorvastatin", 0.0073, 0.0117, 1.607),
    ("carbamazepine", 0.8637, 1.3636, 1.579),
    ("diazepam", 0.1361, 0.2360, 1.734),
    ("digoxin", 0.0014, 0.0015, 1.097),
    ("fluoxetine", 0.0116, 0.0150, 1.293),
    ("nifedipine", 0.0678, 0.0751, 1.108),
    ("omeprazole", 0.2979, 0.1412, 2.109),
    ("phenytoin", 5.2314, 5.2895, 1.011),
    ("theophylline", 5.6302, 8.7247, 1.550),
    ("verapamil", 0.0402, 0.0770, 1.915),
    ("atenolol", 0.2647, 0.3000, 1.133),
    ("fluconazole", 2.3923, 6.7200, 2.811),
    ("furosemide", 0.7689, 1.1100, 1.444),
    ("gabapentin", 3.2008, 2.9000, 1.104),
    ("metformin", 1.4433, 1.0300, 1.401),
]

AAFE = 1.513
AAFE_CI_LO = 1.33
AAFE_CI_HI = 1.74
PCT_2FOLD = 87.5

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
# Derived arrays
# ---------------------------------------------------------------------------

drugs = [r[0] for r in BENCHMARK]
pred = np.array([r[1] for r in BENCHMARK])
obs = np.array([r[2] for r in BENCHMARK])
fe = np.array([r[3] for r in BENCHMARK])

log_pred = np.log10(pred)
log_obs = np.log10(obs)


def _color(f):
    if f <= 2.0:
        return "#2ca02c"  # green
    elif f <= 4.0:
        return "#ff7f0e"  # orange
    else:
        return "#d62728"  # red


point_colors = [_color(f) for f in fe]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6.5))
fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.12)

# --- reference lines ---
lim_lo = min(log_obs.min(), log_pred.min()) - 0.4
lim_hi = max(log_obs.max(), log_pred.max()) + 0.4
x_ref = np.linspace(lim_lo, lim_hi, 200)

ax.plot(x_ref, x_ref, color="#888888", linewidth=1.2, zorder=1, label="Unity (1:1)")
ax.plot(
    x_ref,
    x_ref + np.log10(2),
    color="#bbbbbb",
    linewidth=1.0,
    linestyle="--",
    zorder=1,
    label="2-fold boundary",
)
ax.plot(
    x_ref,
    x_ref - np.log10(2),
    color="#bbbbbb",
    linewidth=1.0,
    linestyle="--",
    zorder=1,
)

# --- scatter ---
ax.scatter(
    log_obs,
    log_pred,
    c=point_colors,
    s=52,
    edgecolors="white",
    linewidths=0.5,
    zorder=3,
)

# --- annotate outliers (FE > 3×) ---
OUTLIER_THRESH = 3.0
for drug, lp, lo, f in zip(drugs, log_pred, log_obs, fe):
    if f > OUTLIER_THRESH:
        ax.annotate(
            drug,
            xy=(lo, lp),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=8,
            color="#d62728",
            fontweight="bold",
        )

# --- axes ---
pad = 0.3
all_vals = np.concatenate([log_obs, log_pred])
ax_lo = all_vals.min() - pad
ax_hi = all_vals.max() + pad
ax.set_xlim(ax_lo, ax_hi)
ax.set_ylim(ax_lo, ax_hi)
ax.set_aspect("equal")

ax.set_xlabel("Observed Cmax  (log₁₀, mg/L)", fontsize=10)
ax.set_ylabel("Predicted Cmax  (log₁₀, mg/L)", fontsize=10)
ax.set_title(
    "Predicted vs Observed Cmax — Gold Tier (N=24)",
    fontsize=11,
    fontweight="bold",
    loc="left",
    pad=8,
)

# AAFE annotation box
textstr = (
    f"AAFE = {AAFE:.3f}\n"
    f"95% CI [{AAFE_CI_LO:.2f}, {AAFE_CI_HI:.2f}]\n"
    f"{PCT_2FOLD:.0f}% within 2-fold"
)
ax.text(
    0.03,
    0.97,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
)

# --- legend ---
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

ax.grid(linestyle="--", linewidth=0.45, alpha=0.5, zorder=0)

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
pdf_path = f"{OUT_DIR}/fig1_pred_vs_obs_cmax.pdf"
png_path = f"{OUT_DIR}/fig1_pred_vs_obs_cmax.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
