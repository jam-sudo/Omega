"""
Phase 365 — Clearance-Volume Relationship Analysis

Analyzes the CL-Vd relationship across compounds or conditions to understand
the PK design space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CLVdAnalysisResult:
    dataset_name: str
    n_compounds: int

    # Input data
    compound_names: list[str]
    cl_values_L_per_h: list[float]
    vd_values_L: list[float]
    t_half_values_h: list[float]  # Computed from CL and Vd

    # Statistical summary
    cl_mean: float
    cl_cv_pct: float
    cl_median: float
    cl_range: tuple[float, float]

    vd_mean: float
    vd_cv_pct: float
    vd_median: float

    t_half_mean: float
    t_half_cv_pct: float

    # Correlation
    pearson_r_cl_vd: float
    pearson_p_cl_vd: float

    # Classification
    high_cl_fraction: float    # Fraction with CL > 30 L/h (high extraction)
    high_vd_fraction: float    # Fraction with Vd > 50 L (extensive distribution)
    long_thalf_fraction: float  # Fraction with t½ > 12 h

    # Optimal PK classification
    n_once_daily_candidates: int   # t½ 12-36 h (good for QD dosing)
    n_twice_daily_candidates: int  # t½ 6-12 h
    n_rapid_clearance: int         # t½ < 2 h

    notes: str


def analyze_cl_vd_dataset(
    dataset_name: str,
    compound_names: list[str],
    cl_values_L_per_h: list[float],
    vd_values_L: list[float],
) -> CLVdAnalysisResult:
    """Analyze the CL-Vd relationship for a dataset of compounds.

    Parameters
    ----------
    dataset_name:
        Descriptive name for the dataset.
    compound_names:
        Names of each compound.
    cl_values_L_per_h:
        Clearance values (L/h) for each compound.
    vd_values_L:
        Volume of distribution values (L) for each compound.

    Returns
    -------
    CLVdAnalysisResult
    """
    # --- Input validation ---
    n = len(compound_names)
    if len(cl_values_L_per_h) != n or len(vd_values_L) != n:
        raise ValueError(
            "compound_names, cl_values_L_per_h, and vd_values_L must have equal length."
        )
    if n < 3:
        raise ValueError("At least 3 compounds are required for analysis.")
    cl_arr = np.array(cl_values_L_per_h, dtype=float)
    vd_arr = np.array(vd_values_L, dtype=float)
    if np.any(cl_arr <= 0):
        raise ValueError("All cl_values_L_per_h must be > 0.")
    if np.any(vd_arr <= 0):
        raise ValueError("All vd_values_L must be > 0.")

    # --- Derived: half-life ---
    t_half_arr = 0.693 * vd_arr / cl_arr

    # --- Statistical summary: CL ---
    cl_mean = float(np.mean(cl_arr))
    cl_std = float(np.std(cl_arr, ddof=1))
    cl_cv_pct = (cl_std / cl_mean * 100.0) if cl_mean > 0 else 0.0
    cl_median = float(np.median(cl_arr))
    cl_range = (float(np.min(cl_arr)), float(np.max(cl_arr)))

    # --- Statistical summary: Vd ---
    vd_mean = float(np.mean(vd_arr))
    vd_std = float(np.std(vd_arr, ddof=1))
    vd_cv_pct = (vd_std / vd_mean * 100.0) if vd_mean > 0 else 0.0
    vd_median = float(np.median(vd_arr))

    # --- Statistical summary: t½ ---
    t_half_mean = float(np.mean(t_half_arr))
    t_half_std = float(np.std(t_half_arr, ddof=1))
    t_half_cv_pct = (t_half_std / t_half_mean * 100.0) if t_half_mean > 0 else 0.0

    # --- Correlation ---
    if n >= 2:
        r, p = stats.pearsonr(cl_arr, vd_arr)
        pearson_r_cl_vd = float(r)
        pearson_p_cl_vd = float(p)
    else:
        pearson_r_cl_vd = 0.0
        pearson_p_cl_vd = 1.0

    # --- Classification fractions ---
    high_cl_fraction = float(np.sum(cl_arr > 30.0) / n)
    high_vd_fraction = float(np.sum(vd_arr > 50.0) / n)
    long_thalf_fraction = float(np.sum(t_half_arr > 12.0) / n)

    # --- Dosing interval candidates ---
    n_once_daily_candidates = int(np.sum((t_half_arr >= 12.0) & (t_half_arr <= 36.0)))
    n_twice_daily_candidates = int(np.sum((t_half_arr >= 6.0) & (t_half_arr < 12.0)))
    n_rapid_clearance = int(np.sum(t_half_arr < 2.0))

    # --- Notes ---
    notes = (
        f"Dataset '{dataset_name}': {n} compounds analysed. "
        f"Mean CL={cl_mean:.2f} L/h, Mean Vd={vd_mean:.2f} L, Mean t½={t_half_mean:.2f} h."
    )

    return CLVdAnalysisResult(
        dataset_name=dataset_name,
        n_compounds=n,
        compound_names=list(compound_names),
        cl_values_L_per_h=list(cl_values_L_per_h),
        vd_values_L=list(vd_values_L),
        t_half_values_h=list(t_half_arr),
        cl_mean=cl_mean,
        cl_cv_pct=cl_cv_pct,
        cl_median=cl_median,
        cl_range=cl_range,
        vd_mean=vd_mean,
        vd_cv_pct=vd_cv_pct,
        vd_median=vd_median,
        t_half_mean=t_half_mean,
        t_half_cv_pct=t_half_cv_pct,
        pearson_r_cl_vd=pearson_r_cl_vd,
        pearson_p_cl_vd=pearson_p_cl_vd,
        high_cl_fraction=high_cl_fraction,
        high_vd_fraction=high_vd_fraction,
        long_thalf_fraction=long_thalf_fraction,
        n_once_daily_candidates=n_once_daily_candidates,
        n_twice_daily_candidates=n_twice_daily_candidates,
        n_rapid_clearance=n_rapid_clearance,
        notes=notes,
    )


def find_optimal_pk_window(
    target_thalf_h: float,
    target_cl_L_per_h: float,
    cl_values_L_per_h: list[float],
    vd_values_L: list[float],
    compound_names: list[str],
    tolerance_pct: float = 30.0,
) -> list[str]:
    """Find compounds within tolerance of target CL and t½.

    Parameters
    ----------
    target_thalf_h:
        Target half-life (h).
    target_cl_L_per_h:
        Target clearance (L/h).
    cl_values_L_per_h:
        Clearance values for each compound.
    vd_values_L:
        Volume of distribution values for each compound.
    compound_names:
        Names of compounds.
    tolerance_pct:
        Percentage tolerance around each target (default 30 %).

    Returns
    -------
    List of compound names matching both criteria.
    """
    cl_arr = np.array(cl_values_L_per_h, dtype=float)
    vd_arr = np.array(vd_values_L, dtype=float)
    t_half_arr = 0.693 * vd_arr / cl_arr

    tol = tolerance_pct / 100.0
    cl_lo = target_cl_L_per_h * (1.0 - tol)
    cl_hi = target_cl_L_per_h * (1.0 + tol)
    th_lo = target_thalf_h * (1.0 - tol)
    th_hi = target_thalf_h * (1.0 + tol)

    matches = []
    for i, name in enumerate(compound_names):
        if cl_lo <= cl_arr[i] <= cl_hi and th_lo <= t_half_arr[i] <= th_hi:
            matches.append(name)
    return matches


def plot_cl_vd_summary(result: CLVdAnalysisResult) -> str:
    """Return a formatted ASCII text table summarising the analysis.

    Parameters
    ----------
    result:
        A ``CLVdAnalysisResult`` from :func:`analyze_cl_vd_dataset`.

    Returns
    -------
    Multi-line string formatted as an ASCII table.
    """
    sep = "=" * 60
    lines = [
        sep,
        f"  CL-Vd Analysis: {result.dataset_name}",
        f"  Compounds: {result.n_compounds}",
        sep,
        f"  {'Parameter':<25} {'Mean':>10} {'Median':>10} {'CV%':>8}",
        f"  {'-'*55}",
        f"  {'CL (L/h)':<25} {result.cl_mean:>10.2f} {result.cl_median:>10.2f} {result.cl_cv_pct:>8.1f}",
        f"  {'Vd (L)':<25} {result.vd_mean:>10.2f} {result.vd_median:>10.2f} {result.vd_cv_pct:>8.1f}",
        f"  {'t½ (h)':<25} {result.t_half_mean:>10.2f} {'—':>10} {result.t_half_cv_pct:>8.1f}",
        sep,
        f"  Pearson r (CL vs Vd): {result.pearson_r_cl_vd:.3f}  (p={result.pearson_p_cl_vd:.4f})",
        sep,
        f"  High-CL fraction (>30 L/h):    {result.high_cl_fraction:.1%}",
        f"  High-Vd fraction (>50 L):      {result.high_vd_fraction:.1%}",
        f"  Long t½ fraction (>12 h):      {result.long_thalf_fraction:.1%}",
        sep,
        f"  QD candidates (t½ 12-36 h):    {result.n_once_daily_candidates}",
        f"  BID candidates (t½ 6-12 h):    {result.n_twice_daily_candidates}",
        f"  Rapid clearance (t½ <2 h):     {result.n_rapid_clearance}",
        sep,
    ]

    # Per-compound table
    lines += [
        f"  {'Compound':<20} {'CL (L/h)':>10} {'Vd (L)':>10} {'t½ (h)':>10}",
        f"  {'-'*55}",
    ]
    for i, name in enumerate(result.compound_names):
        lines.append(
            f"  {name:<20} {result.cl_values_L_per_h[i]:>10.2f} "
            f"{result.vd_values_L[i]:>10.2f} {result.t_half_values_h[i]:>10.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)
