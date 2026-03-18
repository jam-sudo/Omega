"""fig_population.py — Publication-quality population PK figure.

Generates a 2-panel figure from population simulation results:

  Panel A (left): Violin + box plot showing the Cmax distribution for each drug,
    coloured by subpopulation (CYP3A4: continuous gradient; CYP2D6: PM/EM/UM).

  Panel B (right): Scatter plot of individual Cmax vs CYP activity multiplier,
    showing the dose-response relationship with a LOESS trend line.

Usage as standalone script::

    python docs/paper/figures/fig_population.py \
        results/population_sim_midazolam_N100_seed42.json \
        results/population_sim_tramadol_N100_seed42.json

Usage from run_population_sim.py::

    The generate_figure(summaries) function accepts a list of summary dicts
    returned by run_population_simulation().

Output files (in same directory as this script):
  fig_population.pdf
  fig_population.png  (300 DPI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants (match existing paper figures)
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
DPI = 300
FIGURE_SIZE = (10.0, 4.5)

# Colours
COL_3A4 = "#2196F3"  # blue — CYP3A4 substrate
COL_2D6_PM = "#F44336"  # red — PM
COL_2D6_EM = "#4CAF50"  # green — EM
COL_2D6_UM = "#FF9800"  # orange — UM
COL_2D6_ALL = "#9C27B0"  # purple — all CYP2D6

PHENOTYPE_COLOURS = {"PM": COL_2D6_PM, "EM": COL_2D6_EM, "UM": COL_2D6_UM}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loess_smooth(
    x: np.ndarray, y: np.ndarray, frac: float = 0.4, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Simple local polynomial smoother (tricubic kernel, degree-1).

    Returns (x_smooth, y_smooth) over the range of x.
    Falls back to linear fit if fewer than 5 points.
    """
    if len(x) < 5:
        p = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), n_points)
        return xf, np.polyval(p, xf)

    xf = np.linspace(x.min(), x.max(), n_points)
    half_window = max(int(frac * len(x)), 3)
    yf = np.empty(n_points)
    for i, xi in enumerate(xf):
        dists = np.abs(x - xi)
        idx = np.argsort(dists)[:half_window]
        xw = x[idx]
        yw = y[idx]
        dmax = dists[idx].max()
        if dmax == 0:
            yf[i] = yw.mean()
            continue
        u = dists[idx] / dmax
        weights = (1.0 - u**3) ** 3
        # Weighted linear fit
        sw = weights.sum()
        swx = (weights * xw).sum()
        swy = (weights * yw).sum()
        swx2 = (weights * xw**2).sum()
        swxy = (weights * xw * yw).sum()
        det = sw * swx2 - swx**2
        if abs(det) < 1e-12:
            yf[i] = swy / sw
        else:
            b1 = (sw * swxy - swx * swy) / det
            b0 = (swy - b1 * swx) / sw
            yf[i] = b0 + b1 * xi
    return xf, yf


def _violin_parts(ax: Any, data: np.ndarray, x_pos: float, color: str, width: float = 0.3) -> None:
    """Draw a violin + embedded box at position x_pos."""
    if len(data) < 3:
        ax.scatter([x_pos] * len(data), data, color=color, zorder=3, s=20, alpha=0.7)
        return
    parts = ax.violinplot(
        data, positions=[x_pos], widths=[width], showmedians=False, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    # Box whiskers
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)
    ax.plot([x_pos, x_pos], [lo, hi], color="black", linewidth=1.2, zorder=4)
    ax.plot([x_pos - 0.04, x_pos + 0.04], [q1, q1], color="black", linewidth=1.2, zorder=4)
    ax.plot([x_pos - 0.04, x_pos + 0.04], [q3, q3], color="black", linewidth=1.2, zorder=4)
    ax.scatter([x_pos], [med], color="white", edgecolor="black", s=22, zorder=5, linewidths=0.8)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_figure(summaries: list[dict[str, Any]], output_dir: Path | None = None) -> None:
    """Generate 2-panel population PK figure.

    Args:
        summaries: List of dicts returned by run_population_simulation().
            Each entry must have keys: drug, subjects, primary_cyp, cmax_stats, etc.
        output_dir: Directory to write fig_population.{pdf,png}. Defaults to
            the directory containing this script.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    ax_violin = axes[0]
    ax_scatter = axes[1]

    # Collect data per drug
    violin_entries: list[dict[str, Any]] = []
    for summary in summaries:
        drug = summary["drug"]
        cyp = summary["primary_cyp"]
        subjects = summary["subjects"]
        cmax_arr = np.array([s["cmax_mg_L"] for s in subjects])
        cv = summary["cmax_stats"]["cv_pct"]
        violin_entries.append(
            {
                "drug": drug,
                "cyp": cyp,
                "cmax": cmax_arr,
                "cv": cv,
                "subjects": subjects,
            }
        )

    # -----------------------------------------------------------------------
    # Panel A: Violin + box per drug, sub-grouped by phenotype for CYP2D6
    # -----------------------------------------------------------------------
    x_tick_positions = []
    x_tick_labels = []
    legend_handles = []
    legend_seen: set[str] = set()

    x_cursor = 0.5
    for entry in violin_entries:
        drug = entry["drug"]
        cyp = entry["cyp"]
        cmax_all = entry["cmax"]

        if cyp == "CYP3A4":
            # Single violin for CYP3A4 (continuous activity, no discrete subgroups)
            _violin_parts(ax_violin, cmax_all, x_cursor, color=COL_3A4, width=0.35)
            x_tick_positions.append(x_cursor)
            x_tick_labels.append(f"{drug.title()}\n(CYP3A4)\nCV={entry['cv']:.0f}%")
            if "CYP3A4" not in legend_seen:
                legend_handles.append(
                    mpatches.Patch(color=COL_3A4, alpha=0.7, label="CYP3A4 substrate")
                )
                legend_seen.add("CYP3A4")
            x_cursor += 1.0

        else:
            # CYP2D6: 3 sub-violins (PM, EM, UM)
            subj_list = entry["subjects"]
            subgroups = {"PM": [], "EM": [], "UM": []}
            for s in subj_list:
                phen = s["cyp2d6_phenotype"]
                if phen in subgroups:
                    subgroups[phen].append(s["cmax_mg_L"])

            # Positions: PM left, EM centre, UM right
            sub_offsets = {"PM": -0.30, "EM": 0.0, "UM": +0.30}
            centre = x_cursor + 0.0
            for phen, offset in sub_offsets.items():
                sub_cmax = np.array(subgroups[phen])
                if len(sub_cmax) > 0:
                    _violin_parts(
                        ax_violin,
                        sub_cmax,
                        centre + offset,
                        color=PHENOTYPE_COLOURS[phen],
                        width=0.22,
                    )
                if phen not in legend_seen:
                    legend_handles.append(
                        mpatches.Patch(
                            color=PHENOTYPE_COLOURS[phen],
                            alpha=0.7,
                            label=f"CYP2D6 {phen}",
                        )
                    )
                    legend_seen.add(phen)

            x_tick_positions.append(centre)
            x_tick_labels.append(f"{drug.title()}\n(CYP2D6)\nCV={entry['cv']:.0f}%")
            x_cursor += 1.1

    ax_violin.set_xticks(x_tick_positions)
    ax_violin.set_xticklabels(x_tick_labels, ha="center")
    ax_violin.set_ylabel("C$_{max}$ (mg/L)", fontsize=10)
    ax_violin.set_title(
        "A   Population C$_{max}$ Distributions", loc="left", fontweight="bold", fontsize=10
    )
    ax_violin.set_xlim(0.0, x_cursor)
    ax_violin.spines["top"].set_visible(False)
    ax_violin.spines["right"].set_visible(False)
    ax_violin.legend(handles=legend_handles, loc="upper right", frameon=False)

    # -----------------------------------------------------------------------
    # Panel B: Scatter — Cmax vs CYP activity multiplier
    # -----------------------------------------------------------------------
    for entry in violin_entries:
        cyp = entry["cyp"]
        subj_list = entry["subjects"]
        drug = entry["drug"]

        if cyp == "CYP3A4":
            acts = np.array([s["cyp3a4_activity"] for s in subj_list])
            cmax = np.array([s["cmax_mg_L"] for s in subj_list])
            ax_scatter.scatter(
                acts,
                cmax,
                color=COL_3A4,
                alpha=0.45,
                s=18,
                linewidths=0,
                label=f"{drug.title()} (CYP3A4)",
            )
            xf, yf = _loess_smooth(acts, cmax, frac=0.4)
            ax_scatter.plot(xf, yf, color=COL_3A4, linewidth=1.5, zorder=5)

        else:
            # CYP2D6: scatter by phenotype colour
            for phen, col in PHENOTYPE_COLOURS.items():
                mask = np.array([s["cyp2d6_phenotype"] == phen for s in subj_list])
                acts = np.array([s["cyp2d6_activity"] for s in subj_list])[mask]
                cmax = np.array([s["cmax_mg_L"] for s in subj_list])[mask]
                if len(acts) > 0:
                    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(acts))
                    ax_scatter.scatter(
                        acts + jitter,
                        cmax,
                        color=col,
                        alpha=0.55,
                        s=18,
                        linewidths=0,
                        label=f"{drug.title()} {phen}",
                    )

    ax_scatter.set_xlabel("CYP Activity Multiplier (relative to EM)", fontsize=10)
    ax_scatter.set_ylabel("C$_{max}$ (mg/L)", fontsize=10)
    ax_scatter.set_title(
        "B   C$_{max}$ vs CYP Activity", loc="left", fontweight="bold", fontsize=10
    )
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    # Reference line at EM (activity = 1.0)
    ax_scatter.axvline(
        1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="EM (activity=1.0)"
    )
    ax_scatter.legend(loc="upper right", frameon=False, ncol=1)

    # -----------------------------------------------------------------------
    # Caption-level annotation
    # -----------------------------------------------------------------------
    fig.suptitle(
        "Population Pharmacokinetic Variability — Phase 5.4",
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )

    # Per-drug CV annotation at bottom
    cv_texts = []
    for entry in violin_entries:
        ref = summaries[violin_entries.index(entry)].get("ref_cmax_cv_pct")
        ref_str = f", ref {ref:.0f}%" if ref else ""
        cv_texts.append(f"{entry['drug'].title()}: sim CV={entry['cv']:.1f}%{ref_str}")
    fig.text(
        0.5,
        -0.02,
        "  |  ".join(cv_texts),
        ha="center",
        fontsize=7.5,
        color="dimgray",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1.0, 1.0])

    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig_population.{ext}"
        fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python fig_population.py results/population_sim_*.json")
        sys.exit(1)

    summaries = []
    for path_str in sys.argv[1:]:
        p = Path(path_str)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.", file=sys.stderr)
            continue
        with open(p) as f:
            summaries.append(json.load(f))

    if not summaries:
        print("No valid JSON files provided.", file=sys.stderr)
        sys.exit(1)

    generate_figure(summaries)


if __name__ == "__main__":
    main()
