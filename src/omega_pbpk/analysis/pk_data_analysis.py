"""Phase 437 — PK Data Analysis.

Analyze observed PK data from clinical studies or preclinical experiments.
Provides descriptive statistics, outlier detection, and correlation analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKDataSummary:
    """Summary statistics for a PK parameter across subjects."""

    pk_param: str
    n: int
    mean: float
    median: float
    sd: float
    cv_pct: float
    min_val: float
    max_val: float
    geometric_mean: float
    geometric_cv_pct: float
    notes: str


def _extract_values(subjects: list[dict], pk_param: str) -> list[float]:
    """Extract and validate numeric values for pk_param from subjects list."""
    if not subjects:
        raise ValueError("subjects list must not be empty")
    if not pk_param:
        raise ValueError("pk_param must be a non-empty string")

    values: list[float] = []
    for i, s in enumerate(subjects):
        if pk_param not in s:
            raise ValueError(f"pk_param '{pk_param}' not found in subject at index {i}")
        v = float(s[pk_param])
        values.append(v)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _sd(values: list[float], mean: float) -> float:
    """Sample standard deviation (ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def summarize_pk_data(subjects: list[dict], pk_param: str) -> PKDataSummary:
    """Compute descriptive statistics for a PK parameter across subjects.

    Parameters
    ----------
    subjects:
        List of subject dicts, each containing PK parameters as numeric values.
        E.g. [{"id": 1, "cl": 5.2, "vd": 50, "t_half": 6.6, "auc": 120}, ...]
    pk_param:
        The parameter name to summarize (e.g. "cl", "auc").

    Returns
    -------
    PKDataSummary
        Descriptive statistics including arithmetic and geometric summaries.
    """
    values = _extract_values(subjects, pk_param)
    n = len(values)

    if any(v <= 0 for v in values):
        raise ValueError(f"All values for '{pk_param}' must be > 0 to compute geometric statistics")

    mn = _mean(values)
    med = _median(values)
    sd = _sd(values, mn)
    cv_pct = (sd / mn * 100.0) if mn > 0 else 0.0
    min_val = min(values)
    max_val = max(values)

    # Geometric mean = exp(mean of log-values)
    log_values = [math.log(v) for v in values]
    log_mean = _mean(log_values)
    geom_mean = math.exp(log_mean)

    # Geometric CV% = (exp(sd_log) - 1) * 100
    sd_log = _sd(log_values, log_mean)
    geom_cv_pct = (math.exp(sd_log) - 1.0) * 100.0

    notes = (
        f"n={n} subjects; arithmetic mean={mn:.4g}, geometric mean={geom_mean:.4g}; "
        f"CV%={cv_pct:.1f}%, geometric CV%={geom_cv_pct:.1f}%."
    )

    return PKDataSummary(
        pk_param=pk_param,
        n=n,
        mean=mn,
        median=med,
        sd=sd,
        cv_pct=cv_pct,
        min_val=min_val,
        max_val=max_val,
        geometric_mean=geom_mean,
        geometric_cv_pct=geom_cv_pct,
        notes=notes,
    )


def detect_outliers(values: list[float], method: str = "iqr") -> list[int]:
    """Detect outlier indices in a list of values.

    Parameters
    ----------
    values:
        List of numeric values to assess.
    method:
        "iqr" — uses IQR × 1.5 Tukey fences.
        "zscore" — flags |z| > 3.

    Returns
    -------
    list[int]
        Indices of outlier values.
    """
    if not values:
        raise ValueError("values list must not be empty")
    valid_methods = {"iqr", "zscore"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    n = len(values)
    outlier_indices: list[int] = []

    if method == "iqr":
        sorted_vals = sorted(values)
        # Lower quartile: median of lower half (exclusive of median for odd n)
        mid = n // 2
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid + (n % 2) :]

        q1 = _median(lower_half) if lower_half else sorted_vals[0]
        q3 = _median(upper_half) if upper_half else sorted_vals[-1]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        for i, v in enumerate(values):
            if v < lower_fence or v > upper_fence:
                outlier_indices.append(i)

    else:  # zscore
        if n < 2:
            return []
        mn = _mean(values)
        sd = _sd(values, mn)
        if sd == 0.0:
            return []
        for i, v in enumerate(values):
            z = abs((v - mn) / sd)
            if z > 3.0:
                outlier_indices.append(i)

    return outlier_indices


def correlation_analysis(subjects: list[dict], param_x: str, param_y: str) -> dict:
    """Compute Pearson correlation between two PK parameters.

    Parameters
    ----------
    subjects:
        List of subject dicts containing both param_x and param_y.
    param_x:
        Name of the x-axis parameter.
    param_y:
        Name of the y-axis parameter.

    Returns
    -------
    dict with keys: r, r_squared, slope, intercept, n
    """
    if not subjects:
        raise ValueError("subjects list must not be empty")
    if not param_x:
        raise ValueError("param_x must be a non-empty string")
    if not param_y:
        raise ValueError("param_y must be a non-empty string")

    x_vals = _extract_values(subjects, param_x)
    y_vals = _extract_values(subjects, param_y)

    n = len(x_vals)
    if n < 2:
        raise ValueError("At least 2 subjects are required for correlation analysis")

    mx = _mean(x_vals)
    my = _mean(y_vals)

    # Pearson r (manual formula)
    num = sum((x_vals[i] - mx) * (y_vals[i] - my) for i in range(n))
    denom_x = math.sqrt(sum((x_vals[i] - mx) ** 2 for i in range(n)))
    denom_y = math.sqrt(sum((y_vals[i] - my) ** 2 for i in range(n)))

    if denom_x < 1e-15 or denom_y < 1e-15:
        r = 0.0
    else:
        r = num / (denom_x * denom_y)

    # Clamp to [-1, 1] for floating-point safety
    r = max(-1.0, min(1.0, r))
    r_squared = r**2

    # Least-squares slope and intercept
    if denom_x**2 < 1e-30:
        slope = 0.0
        intercept = my
    else:
        slope = num / (denom_x**2)
        intercept = my - slope * mx

    return {
        "r": r,
        "r_squared": r_squared,
        "slope": slope,
        "intercept": intercept,
        "n": n,
    }
