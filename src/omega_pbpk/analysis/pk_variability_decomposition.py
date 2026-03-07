"""PK variability source decomposition — Phase 452.

Decomposes total pharmacokinetic variability into between-subject (BSV),
between-occasion (BOR), and residual components using geometric CV and
ANOVA-based methods.

All calculations use pure Python (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "VariabilityResult",
    "decompose_variability",
    "calculate_geometric_cv",
    "anova_variability",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class VariabilityResult:
    """Results of PK variability decomposition.

    Attributes
    ----------
    n_subjects : int
        Number of observations (subjects).
    cmax_gcv_pct : float
        Geometric CV% for Cmax.
    auc_gcv_pct : float
        Geometric CV% for AUC.
    cmax_p5 : float
        5th percentile of Cmax values.
    cmax_p50 : float
        Median (50th percentile) of Cmax values.
    cmax_p95 : float
        95th percentile of Cmax values.
    auc_p5 : float
        5th percentile of AUC values.
    auc_p50 : float
        Median (50th percentile) of AUC values.
    auc_p95 : float
        95th percentile of AUC values.
    between_subject_cv_pct : float
        Between-subject CV% (from ANOVA on log-Cmax if group_ids provided;
        otherwise equals cmax_gcv_pct).
    within_subject_cv_pct : float
        Within-subject (residual) CV% from ANOVA. 0.0 if no groups provided.
    notes : str
        Descriptive summary.
    """

    n_subjects: int
    cmax_gcv_pct: float
    auc_gcv_pct: float
    cmax_p5: float
    cmax_p50: float
    cmax_p95: float
    auc_p5: float
    auc_p50: float
    auc_p95: float
    between_subject_cv_pct: float
    within_subject_cv_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Pure-Python math helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Sample variance (Bessel's correction)."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (n - 1)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation percentile from a sorted list (0 <= p <= 100)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_values[-1]
    frac = rank - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


# ---------------------------------------------------------------------------
# Public calculation functions
# ---------------------------------------------------------------------------


def calculate_geometric_cv(values: list[float]) -> float:
    """Compute geometric coefficient of variation (GCV%).

    GCV% = sqrt(exp(var(log(values))) - 1) * 100

    Parameters
    ----------
    values : list[float]
        Positive numeric values (e.g., PK parameters).

    Returns
    -------
    float
        GCV in percent.

    Raises
    ------
    ValueError
        If values is empty or contains non-positive numbers.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive for geometric CV calculation")
    log_vals = [math.log(v) for v in values]
    var_log = _variance(log_vals)
    gcv = math.sqrt(max(0.0, math.exp(var_log) - 1.0)) * 100.0
    return gcv


def anova_variability(
    values: list[float],
    group_ids: list[str | int],
) -> dict[str, float]:
    """One-way ANOVA decomposition on log-scale values.

    Parameters
    ----------
    values : list[float]
        Positive observations.
    group_ids : list[str | int]
        Group label for each observation (same length as values).

    Returns
    -------
    dict with keys:
        between_group_cv : float — between-group CV% on natural scale
        within_group_cv  : float — within-group (residual) CV% on natural scale
        icc              : float — intraclass correlation coefficient [0, 1]

    Raises
    ------
    ValueError
        If values and group_ids have different lengths, or fewer than 2 groups.
    """
    if len(values) != len(group_ids):
        raise ValueError("values and group_ids must have the same length")
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive")

    # Work on log scale
    log_vals = [math.log(v) for v in values]

    # Group observations
    groups: dict[str | int, list[float]] = {}
    for gid, lv in zip(group_ids, log_vals, strict=True):
        groups.setdefault(gid, []).append(lv)

    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError("At least 2 distinct groups are required for ANOVA")

    n_total = len(log_vals)
    grand_mean = _mean(log_vals)

    # Between-group sum of squares
    ss_between = sum(len(g_vals) * (_mean(g_vals) - grand_mean) ** 2 for g_vals in groups.values())
    df_between = n_groups - 1

    # Within-group sum of squares
    ss_within = sum(sum((x - _mean(g_vals)) ** 2 for x in g_vals) for g_vals in groups.values())
    df_within = n_total - n_groups

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # Average group size (for ICC estimation)
    n0 = n_total / n_groups  # simplified; exact formula uses harmonic mean

    # Variance components
    var_between = max(0.0, (ms_between - ms_within) / n0)
    var_within = ms_within

    # Convert log-scale variance components to CV% (approximate via delta method)
    # For log-normal: CV% ≈ sqrt(exp(sigma^2) - 1) * 100
    between_cv = math.sqrt(max(0.0, math.exp(var_between) - 1.0)) * 100.0
    within_cv = math.sqrt(max(0.0, math.exp(var_within) - 1.0)) * 100.0

    total_var = var_between + var_within
    icc = var_between / total_var if total_var > 0 else 0.0

    return {
        "between_group_cv": between_cv,
        "within_group_cv": within_cv,
        "icc": icc,
    }


# ---------------------------------------------------------------------------
# Main decomposition function
# ---------------------------------------------------------------------------


def decompose_variability(
    cmax_values: list[float],
    auc_values: list[float],
    group_ids: list[str | int] | None = None,
) -> VariabilityResult:
    """Decompose total PK variability into between- and within-subject components.

    Parameters
    ----------
    cmax_values : list[float]
        Observed Cmax values (positive, one per subject/occasion).
    auc_values : list[float]
        Observed AUC values (positive, one per subject/occasion).
    group_ids : list or None
        Subject/group identifiers. If provided, enables ANOVA decomposition
        to separate between-subject from within-subject variability.

    Returns
    -------
    VariabilityResult

    Raises
    ------
    ValueError
        If inputs are invalid (empty, non-positive, mismatched lengths).
    """
    # Validate inputs
    if not cmax_values:
        raise ValueError("cmax_values must not be empty")
    if not auc_values:
        raise ValueError("auc_values must not be empty")
    if len(cmax_values) != len(auc_values):
        raise ValueError("cmax_values and auc_values must have the same length")
    if any(v <= 0 for v in cmax_values):
        raise ValueError("All cmax_values must be positive")
    if any(v <= 0 for v in auc_values):
        raise ValueError("All auc_values must be positive")
    if group_ids is not None and len(group_ids) != len(cmax_values):
        raise ValueError("group_ids must have the same length as cmax_values")

    n = len(cmax_values)

    # Geometric CV
    cmax_gcv = calculate_geometric_cv(cmax_values)
    auc_gcv = calculate_geometric_cv(auc_values)

    # Percentiles
    sorted_cmax = sorted(cmax_values)
    sorted_auc = sorted(auc_values)

    cmax_p5 = _percentile(sorted_cmax, 5.0)
    cmax_p50 = _percentile(sorted_cmax, 50.0)
    cmax_p95 = _percentile(sorted_cmax, 95.0)

    auc_p5 = _percentile(sorted_auc, 5.0)
    auc_p50 = _percentile(sorted_auc, 50.0)
    auc_p95 = _percentile(sorted_auc, 95.0)

    # Variability decomposition
    if group_ids is not None:
        anova_res = anova_variability(cmax_values, group_ids)
        between_cv = anova_res["between_group_cv"]
        within_cv = anova_res["within_group_cv"]
        icc = anova_res["icc"]
        notes = (
            f"n={n}; Cmax GCV={cmax_gcv:.1f}%, AUC GCV={auc_gcv:.1f}%; "
            f"BSV (between-group) Cmax CV={between_cv:.1f}%, "
            f"WSV (within-group) Cmax CV={within_cv:.1f}%; "
            f"ICC={icc:.3f}"
        )
    else:
        between_cv = cmax_gcv
        within_cv = 0.0
        notes = (
            f"n={n}; Cmax GCV={cmax_gcv:.1f}%, AUC GCV={auc_gcv:.1f}%; "
            f"No group_ids provided — total variability attributed to between-subject"
        )

    return VariabilityResult(
        n_subjects=n,
        cmax_gcv_pct=round(cmax_gcv, 3),
        auc_gcv_pct=round(auc_gcv, 3),
        cmax_p5=round(cmax_p5, 4),
        cmax_p50=round(cmax_p50, 4),
        cmax_p95=round(cmax_p95, 4),
        auc_p5=round(auc_p5, 4),
        auc_p50=round(auc_p50, 4),
        auc_p95=round(auc_p95, 4),
        between_subject_cv_pct=round(between_cv, 3),
        within_subject_cv_pct=round(within_cv, 3),
        notes=notes,
    )
