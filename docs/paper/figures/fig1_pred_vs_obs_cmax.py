"""
fig1_pred_vs_obs_cmax.py
Publication-quality pred-vs-obs Cmax scatter plot for the gold-tier 24-drug benchmark.

Panel: log10(pred_cmax) vs log10(obs_cmax)
 - 1:1 line (solid grey)
 - 2-fold dashed lines
 - Points coloured: green (≤2×), orange (2–4×), red (>4×)
 - Worst outliers (FE > 3×) annotated with drug name

Data source: outputs/benchmark_2026-03-17.json
  AAFE 1.793 [95% CI 1.52, 2.18], %2-fold 83% (N=24)

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
    ("caffeine", 1.4821, 2.0408, 1.377),
    ("metoprolol", 0.1273, 0.0663, 1.919),
    ("midazolam", 0.0114, 0.0058, 1.978),
    ("propranolol", 0.0566, 0.0822, 1.452),
    ("warfarin", 0.1898, 1.2783, 6.735),
    ("d-amphetamine", 0.0952, 0.0504, 1.890),
    ("ibuprofen", 3.5272, 19.0273, 5.394),
    ("acetaminophen", 14.1043, 10.9590, 1.287),
    ("amoxicillin", 5.7981, 9.5238, 1.643),
    ("atorvastatin", 0.0207, 0.0117, 1.772),
    ("carbamazepine", 0.8054, 1.3636, 1.693),
    ("diazepam", 0.1199, 0.2360, 1.968),
    ("digoxin", 0.0014, 0.0015, 1.060),
    ("fluoxetine", 0.0212, 0.0150, 1.415),
    ("nifedipine", 0.0695, 0.0751, 1.081),
    ("omeprazole", 0.3249, 0.5946, 1.830),
    ("phenytoin", 3.4544, 5.2895, 1.531),
    ("theophylline", 6.4381, 7.2257, 1.122),
    ("verapamil", 0.0825, 0.0556, 1.483),
    ("atenolol", 0.2771, 0.3800, 1.371),
    ("fluconazole", 2.2133, 7.2000, 3.253),
    ("furosemide", 0.7540, 2.0000, 2.652),
    ("gabapentin", 1.8876, 2.9000, 1.536),
    ("metformin", 0.9856, 1.3500, 1.370),
]

AAFE = 1.793
AAFE_CI_LO = 1.52
AAFE_CI_HI = 2.18
PCT_2FOLD = 83.3

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
