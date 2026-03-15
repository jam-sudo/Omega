#!/usr/bin/env python3
"""Generate publication-quality figures for the Omega PBPK paper.

Produces 6 figures from benchmark results and saves them to docs/paper/figures/
as both PNG (300 DPI) and PDF.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_PATH = ROOT / "outputs" / "benchmark_2026-03-15.json"
SILVER_PATH = ROOT / "outputs" / "silver_tier_results.json"
BRONZE_PATH = ROOT / "outputs" / "bronze_tier_results.json"
TEMPORAL_PATH = ROOT / "outputs" / "temporal_holdout_results.json"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fall back to default

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

# Professional color palette
C_TEAL = "#2b8c8c"
C_BLUE = "#3a7ebf"
C_GREEN = "#4caf50"
C_YELLOW = "#ffc107"
C_RED = "#e53935"
C_GREY = "#757575"
C_LIGHT_BLUE = "#90caf9"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name: str):
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  Saved {p}")
    plt.close(fig)


# ===================================================================
# Figure 1 — Predicted vs Observed Cmax (log-log scatter)
# ===================================================================
def fig1_pred_vs_obs_cmax(data: dict):
    drugs = data.get("per_drug", [])
    names, pred, obs = [], [], []
    for d in drugs:
        p, o = d.get("pred_cmax"), d.get("obs_cmax")
        if p is not None and o is not None and p > 0 and o > 0:
            names.append(d.get("drug", ""))
            pred.append(p)
            obs.append(o)

    pred, obs = np.array(pred), np.array(obs)
    fig, ax = plt.subplots(figsize=(8, 6))

    # boundaries
    lo = min(pred.min(), obs.min()) * 0.3
    hi = max(pred.max(), obs.max()) * 3
    span = np.array([lo, hi])
    ax.plot(span, span, "-", color=C_GREY, lw=1.5, label="Unity (y = x)")
    ax.plot(span, span * 2, "--", color=C_YELLOW, lw=1, alpha=0.7, label="2-fold")
    ax.plot(span, span / 2, "--", color=C_YELLOW, lw=1, alpha=0.7)
    ax.fill_between(span, span / 2, span * 2, alpha=0.07, color=C_YELLOW)

    ax.scatter(obs, pred, s=60, c=C_TEAL, edgecolors="white", linewidths=0.5, zorder=5)

    # labels
    for n, px, ox in zip(names, pred, obs):
        ax.annotate(n, (ox, px), fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed Cmax (mg/L)")
    ax.set_ylabel("Predicted Cmax (mg/L)")
    ax.set_title("Predicted vs. Observed Cmax (20 drugs)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    aafe = data.get("aafe_cmax", 1.90)
    pct = data.get("pct_2fold_cmax", 70.0)
    ax.text(
        0.97,
        0.05,
        f"AAFE = {aafe:.2f}\n{pct:.0f}% within 2-fold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GREY, alpha=0.9),
    )

    _save(fig, "fig1_pred_vs_obs_cmax")


# ===================================================================
# Figure 2 — Predicted vs Observed AUC (log-log scatter)
# ===================================================================
def fig2_pred_vs_obs_auc(data: dict):
    drugs = data.get("per_drug", [])
    names, pred, obs = [], [], []
    for d in drugs:
        p, o = d.get("pred_auc"), d.get("obs_auc")
        if p is not None and o is not None and p > 0 and o > 0:
            names.append(d.get("drug", ""))
            pred.append(p)
            obs.append(o)

    pred, obs = np.array(pred), np.array(obs)
    fig, ax = plt.subplots(figsize=(8, 6))

    lo = min(pred.min(), obs.min()) * 0.3
    hi = max(pred.max(), obs.max()) * 3
    span = np.array([lo, hi])
    ax.plot(span, span, "-", color=C_GREY, lw=1.5, label="Unity (y = x)")
    ax.plot(span, span * 2, "--", color=C_YELLOW, lw=1, alpha=0.7, label="2-fold")
    ax.plot(span, span / 2, "--", color=C_YELLOW, lw=1, alpha=0.7)
    ax.fill_between(span, span / 2, span * 2, alpha=0.07, color=C_YELLOW)

    ax.scatter(obs, pred, s=60, c=C_BLUE, edgecolors="white", linewidths=0.5, zorder=5)

    for n, px, ox in zip(names, pred, obs):
        ax.annotate(n, (ox, px), fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed AUC (mg*h/L)")
    ax.set_ylabel("Predicted AUC (mg*h/L)")
    ax.set_title("Predicted vs. Observed AUC (20 drugs)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    aafe = data.get("aafe_auc", 1.66)
    pct = data.get("pct_2fold_auc", 70.0)
    ax.text(
        0.97,
        0.05,
        f"AAFE = {aafe:.2f}\n{pct:.0f}% within 2-fold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GREY, alpha=0.9),
    )

    _save(fig, "fig2_pred_vs_obs_auc")


# ===================================================================
# Figure 3 — Per-Drug Cmax Fold Error (horizontal bar chart)
# ===================================================================
def fig3_fold_error_bars(data: dict):
    drugs = data.get("per_drug", [])
    entries = []
    for d in drugs:
        fe = d.get("fe_cmax")
        if fe is not None:
            entries.append((d.get("drug", ""), fe))

    # sort ascending
    entries.sort(key=lambda x: x[1])
    names = [e[0] for e in entries]
    fes = np.array([e[1] for e in entries])

    colors = [C_GREEN if fe <= 2 else (C_YELLOW if fe <= 3 else C_RED) for fe in fes]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, fes, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Fold Error (Cmax)")
    ax.set_title("Per-Drug Cmax Fold Error")
    ax.axvline(x=2.0, color=C_GREY, linestyle="--", lw=1.2, label="2-fold threshold")

    # legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=C_GREEN, label="<= 2-fold"),
        Patch(facecolor=C_YELLOW, label="2-3 fold"),
        Patch(facecolor=C_RED, label="> 3-fold"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.set_xlim(0, max(fes) * 1.15)
    ax.invert_yaxis()

    _save(fig, "fig3_fold_error_bars")


# ===================================================================
# Figure 4 — Silver Tier: Predicted vs Observed t1/2 (log-log)
# ===================================================================
def fig4_silver_thalf(data: dict):
    drugs = data.get("per_drug", [])
    names, pred, obs = [], [], []
    for d in drugs:
        p = d.get("pred_thalf_h")
        o = d.get("obs_thalf_h")
        if p is not None and o is not None and p > 0 and o > 0:
            names.append(d.get("drug_name", ""))
            pred.append(p)
            obs.append(o)

    pred, obs = np.array(pred), np.array(obs)
    fig, ax = plt.subplots(figsize=(8, 6))

    lo = min(pred.min(), obs.min()) * 0.4
    hi = max(pred.max(), obs.max()) * 3
    span = np.array([lo, hi])
    ax.plot(span, span, "-", color=C_GREY, lw=1.5, label="Unity (y = x)")
    ax.plot(span, span * 2, "--", color=C_YELLOW, lw=1, alpha=0.7, label="2-fold")
    ax.plot(span, span / 2, "--", color=C_YELLOW, lw=1, alpha=0.7)
    ax.fill_between(span, span / 2, span * 2, alpha=0.07, color=C_YELLOW)

    # color by within_2fold
    within = np.array(
        [
            d.get("within_2fold", False)
            for d in drugs
            if d.get("pred_thalf_h") and d.get("obs_thalf_h")
        ]
    )
    c = [C_TEAL if w else C_RED for w in within]
    ax.scatter(obs, pred, s=50, c=c, edgecolors="white", linewidths=0.5, zorder=5)

    for n, px, ox in zip(names, pred, obs):
        ax.annotate(n, (ox, px), fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed t$_{1/2}$ (h)")
    ax.set_ylabel("Predicted t$_{1/2}$ (h)")
    ax.set_title(f"Predicted vs. Observed Half-Life ({len(pred)} drugs)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    summary = data.get("summary", {})
    aafe = summary.get("AAFE", 2.42)
    pct = summary.get("pct_within_2fold", 51.3)
    ax.text(
        0.97,
        0.05,
        f"AAFE = {aafe:.2f}\n{pct:.0f}% within 2-fold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GREY, alpha=0.9),
    )

    _save(fig, "fig4_silver_thalf")


# ===================================================================
# Figure 5 — Bronze Tier: Per-ADME-Property AAFE (grouped bar)
# ===================================================================
def fig5_bronze_adme(data: dict):
    props_data = data.get("properties", {})
    prop_order = ["logP", "fup", "rbp", "peff", "clint_3a4"]
    display_names = ["logP", "fup", "RBP", "Peff", "CLint"]

    aafes, pcts, n_vals = [], [], []
    for prop in prop_order:
        info = props_data.get(prop, {})
        aafes.append(info.get("aafe", 0))
        pcts.append(info.get("pct_within_2fold", 0))
        n_vals.append(info.get("n", 0))

    aafes = np.array(aafes)
    colors = [C_GREEN if a <= 2 else (C_YELLOW if a <= 3 else C_RED) for a in aafes]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(display_names))
    bars = ax.bar(x, aafes, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    # value labels on bars
    for bar, aafe, pct in zip(bars, aafes, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{aafe:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.0f}%\n2-fold",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.axhline(y=2.0, color=C_GREY, linestyle="--", lw=1.2, label="AAFE = 2.0 target")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("AAFE")
    ax.set_title("ADME Property Prediction Accuracy (153 compounds)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(aafes) * 1.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=C_GREEN, label="AAFE <= 2"),
        Patch(facecolor=C_YELLOW, label="AAFE 2-3"),
        Patch(facecolor=C_RED, label="AAFE > 3"),
        plt.Line2D([0], [0], color=C_GREY, linestyle="--", label="Target (2.0)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    _save(fig, "fig5_bronze_adme")


# ===================================================================
# Figure 6 — Multi-Tier Summary
# ===================================================================
def fig6_multi_tier_summary(gold, silver, bronze, temporal):
    """Summary panel: AAFE across all tiers as grouped bar + table."""
    tiers = []
    metrics = []

    # Gold: Cmax and AUC
    if gold:
        tiers.append("Gold\nCmax")
        metrics.append(gold.get("aafe_cmax", None))
        tiers.append("Gold\nAUC")
        metrics.append(gold.get("aafe_auc", None))

    # Silver: t1/2
    if silver:
        s = silver.get("summary", {})
        tiers.append("Silver\nt$_{1/2}$")
        metrics.append(s.get("AAFE", None))

    # Bronze: individual ADME properties
    if bronze:
        props = bronze.get("properties", {})
        for prop, display in [
            ("logP", "Bronze\nlogP"),
            ("fup", "Bronze\nfup"),
            ("rbp", "Bronze\nRBP"),
            ("peff", "Bronze\nPeff"),
            ("clint_3a4", "Bronze\nCLint"),
        ]:
            info = props.get(prop, {})
            tiers.append(display)
            metrics.append(info.get("aafe", None))

    # Temporal holdout
    if temporal:
        ts = temporal.get("summary", {}).get("thalf", {})
        aafe_t = ts.get("AAFE")
        if aafe_t is not None:
            tiers.append("Temporal\nt$_{1/2}$")
            metrics.append(aafe_t)

    # Filter out Nones
    valid = [(t, m) for t, m in zip(tiers, metrics) if m is not None]
    if not valid:
        print("  WARNING: No data for fig6, skipping.")
        return

    tier_labels = [v[0] for v in valid]
    vals = np.array([v[1] for v in valid])

    # Color by tier group
    tier_colors = []
    for t in tier_labels:
        if "Gold" in t:
            tier_colors.append("#1976d2")  # dark blue
        elif "Silver" in t:
            tier_colors.append("#78909c")  # blue-grey
        elif "Bronze" in t:
            tier_colors.append("#f9a825")  # amber
        else:
            tier_colors.append("#7b1fa2")  # purple for temporal

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tier_labels))
    bars = ax.bar(x, vals, color=tier_colors, edgecolor="white", linewidth=0.8, width=0.65)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(y=2.0, color=C_RED, linestyle="--", lw=1.2, alpha=0.7)
    ax.axhline(y=3.0, color=C_RED, linestyle=":", lw=1.0, alpha=0.5)

    ax.text(
        len(tier_labels) - 0.5, 2.05, "AAFE = 2.0", fontsize=8, color=C_RED, ha="right", alpha=0.8
    )
    ax.text(
        len(tier_labels) - 0.5, 3.05, "AAFE = 3.0", fontsize=8, color=C_RED, ha="right", alpha=0.6
    )

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel("AAFE")
    ax.set_title("Omega PBPK: Multi-Tier Validation Summary")
    ax.set_ylim(0, max(vals) * 1.25)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1976d2", label="Gold (Cmax/AUC, 20 drugs)"),
        Patch(facecolor="#78909c", label="Silver (t1/2, 39 drugs)"),
        Patch(facecolor="#f9a825", label="Bronze (ADME, 153 compounds)"),
        Patch(facecolor="#7b1fa2", label="Temporal holdout (5 drugs)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    _save(fig, "fig6_multi_tier_summary")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Loading data...")
    gold = _load_json(GOLD_PATH)
    silver = _load_json(SILVER_PATH)
    bronze = _load_json(BRONZE_PATH)
    temporal = _load_json(TEMPORAL_PATH)

    print("\nGenerating figures...")

    if gold:
        print("\n[1/6] Predicted vs Observed Cmax")
        fig1_pred_vs_obs_cmax(gold)
        print("[2/6] Predicted vs Observed AUC")
        fig2_pred_vs_obs_auc(gold)
        print("[3/6] Per-Drug Cmax Fold Error bars")
        fig3_fold_error_bars(gold)
    else:
        print("Skipping figs 1-3 (no gold tier data)")

    if silver:
        print("[4/6] Silver Tier: t1/2 scatter")
        fig4_silver_thalf(silver)
    else:
        print("Skipping fig 4 (no silver tier data)")

    if bronze:
        print("[5/6] Bronze Tier: ADME property AAFE")
        fig5_bronze_adme(bronze)
    else:
        print("Skipping fig 5 (no bronze tier data)")

    print("[6/6] Multi-Tier Summary")
    fig6_multi_tier_summary(gold, silver, bronze, temporal)

    print(f"\nAll figures saved to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
