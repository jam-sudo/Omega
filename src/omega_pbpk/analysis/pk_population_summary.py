"""Phase 636 — Population PK summary statistics.

Statistical summary of PK parameters across a virtual population.
Computes mean, median, SD, CV%, percentiles, geometric mean, and BSV.
Pure Python only — no numpy, no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationPKSummary:
    """Summary statistics for a single PK parameter across a population.

    Attributes
    ----------
    n_subjects : Number of subjects.
    parameter_name : Name of the PK parameter (e.g. 'cmax', 'auc').
    mean : Arithmetic mean.
    median : Median (50th percentile).
    sd : Sample standard deviation (ddof=1).
    cv_pct : Coefficient of variation (%).
    p5 : 5th percentile.
    p25 : 25th percentile.
    p75 : 75th percentile.
    p95 : 95th percentile.
    min_val : Minimum value.
    max_val : Maximum value.
    geometric_mean : Geometric mean; 0.0 if any value <= 0.
    notes : Summary notes.
    """

    n_subjects: int
    parameter_name: str
    mean: float
    median: float
    sd: float
    cv_pct: float
    p5: float
    p25: float
    p75: float
    p95: float
    min_val: float
    max_val: float
    geometric_mean: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear interpolation percentile from a sorted list.

    Uses the "exclusive" (Hazen) formula: idx = pct/100 * (n+1) - 1,
    clamped to [0, n-1].  This ensures p95 equals the maximum for
    small lists (n=2) while correctly interpolating medians.

    Parameters
    ----------
    sorted_values : Sorted list of floats.
    pct : Percentile in [0, 100].

    Returns
    -------
    float : Interpolated percentile value.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Exclusive formula: idx = pct/100 * (n+1) - 1, clamped to [0, n-1]
    idx = pct / 100.0 * (n + 1) - 1.0
    idx = max(0.0, min(float(n - 1), idx))
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper or upper >= n:
        return sorted_values[min(lower, n - 1)]
    frac = idx - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


def _compute_stats(values: list[float], parameter_name: str) -> PopulationPKSummary:
    """Internal: compute all statistics for a list of values."""
    n = len(values)
    if n < 1:
        raise ValueError(f"Cannot summarize empty values for parameter '{parameter_name}'")
    if n < 2:
        raise ValueError(
            f"At least 2 subjects required to compute SD for parameter '{parameter_name}'"
        )

    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(variance)
    cv_pct = (sd / abs(mean_val) * 100.0) if mean_val != 0.0 else 0.0

    sorted_vals = sorted(values)
    median = _percentile(sorted_vals, 50.0)
    p5 = _percentile(sorted_vals, 5.0)
    p25 = _percentile(sorted_vals, 25.0)
    p75 = _percentile(sorted_vals, 75.0)
    p95 = _percentile(sorted_vals, 95.0)

    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Geometric mean: only if all values > 0
    if all(v > 0 for v in values):
        geometric_mean = math.exp(sum(math.log(v) for v in values) / n)
    else:
        geometric_mean = 0.0

    notes = f"n={n}; mean={mean_val:.3g}; CV%={cv_pct:.1f}%; range=[{min_val:.3g}, {max_val:.3g}]"

    return PopulationPKSummary(
        n_subjects=n,
        parameter_name=parameter_name,
        mean=mean_val,
        median=median,
        sd=sd,
        cv_pct=cv_pct,
        p5=p5,
        p25=p25,
        p75=p75,
        p95=p95,
        min_val=min_val,
        max_val=max_val,
        geometric_mean=geometric_mean,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_pk_parameter(values: list[float], parameter_name: str) -> PopulationPKSummary:
    """Compute all statistics from a list of individual parameter values.

    Parameters
    ----------
    values : List of individual PK parameter values (one per subject).
    parameter_name : Name of the parameter (used in output and notes).

    Returns
    -------
    PopulationPKSummary

    Raises
    ------
    ValueError : If values is empty or has fewer than 2 elements.
    """
    if not values:
        raise ValueError(f"values must not be empty for parameter '{parameter_name}'")
    return _compute_stats(list(values), parameter_name)


def summarize_population_pk(
    individual_results: list[dict],
    parameters: list[str] | None = None,
) -> dict[str, PopulationPKSummary]:
    """Return population PK summaries for each parameter.

    Parameters
    ----------
    individual_results : List of dicts with keys like 'cmax', 'auc', 'tmax', 't_half'.
    parameters : Which parameters to summarize (default: all keys present in all dicts).

    Returns
    -------
    dict mapping parameter_name -> PopulationPKSummary

    Raises
    ------
    ValueError : If individual_results is empty.
    """
    if not individual_results:
        raise ValueError("individual_results must not be empty")

    # Determine parameters to summarize
    if parameters is None:
        # Use all keys that appear in every dict
        all_keys: set[str] = set(individual_results[0].keys())
        for d in individual_results[1:]:
            all_keys &= set(d.keys())
        parameters = sorted(all_keys)

    result: dict[str, PopulationPKSummary] = {}
    for param in parameters:
        vals = [float(d[param]) for d in individual_results if param in d]
        if vals:
            result[param] = summarize_pk_parameter(vals, param)

    return result


def compute_bsv(values: list[float]) -> dict:
    """Compute between-subject variability metrics.

    Parameters
    ----------
    values : List of individual PK parameter values.

    Returns
    -------
    dict with keys:
        - omega_sq : Variance of log-transformed values (omega²).
        - cv_pct : CV% derived from omega² (sqrt(exp(omega²)-1)*100).
        - iqr : Interquartile range (p75 - p25).

    Raises
    ------
    ValueError : If values is empty or has fewer than 2 elements.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) < 2:
        raise ValueError("At least 2 values required to compute BSV")

    n = len(values)
    sorted_vals = sorted(values)

    # Omega² = variance of log-values (only if all positive)
    if all(v > 0 for v in values):
        log_vals = [math.log(v) for v in values]
        log_mean = sum(log_vals) / n
        omega_sq = sum((lv - log_mean) ** 2 for lv in log_vals) / (n - 1)
        cv_pct = math.sqrt(math.exp(omega_sq) - 1.0) * 100.0
    else:
        # Fallback: use arithmetic variance
        mean_val = sum(values) / n
        omega_sq = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        cv_pct = (math.sqrt(omega_sq) / abs(mean_val) * 100.0) if mean_val != 0.0 else 0.0

    p25 = _percentile(sorted_vals, 25.0)
    p75 = _percentile(sorted_vals, 75.0)
    iqr = p75 - p25

    return {
        "omega_sq": omega_sq,
        "cv_pct": cv_pct,
        "iqr": iqr,
    }


__all__ = [
    "PopulationPKSummary",
    "summarize_pk_parameter",
    "summarize_population_pk",
    "compute_bsv",
]
