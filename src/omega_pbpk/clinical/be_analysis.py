"""Bioequivalence statistical analysis: TOST, power curves, sample size estimation.

Implements FDA TOST (Two One-Sided Tests) approach for BE assessment
with 80-125% acceptance limits on geometric mean ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEResult:
    metric: str
    n_reference: int
    n_test: int
    gm_ratio: float
    ci_lower_90: float
    ci_upper_90: float
    is_bioequivalent: bool
    power_at_observed_n: float
    cv_pct: float
    notes: str


@dataclass(frozen=True)
class PowerCurveResult:
    n_values: list[int]
    powers: list[float]
    cv_pct: float
    ratio: float


def _t_critical(df: int) -> float:
    """Approximate t-critical value for alpha=0.05 (one-sided) using normal approx for large df."""
    if df <= 0:
        return 1.645
    # Approximation: t ~ z + (z^3 + z) / (4*df) for moderate df
    z = 1.645
    if df >= 30:
        return z + (z**3 + z) / (4 * df)
    # For smaller df, use a lookup-based rough approximation
    # t_0.05 values for df 1-30
    t_table = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        21: 1.721,
        22: 1.717,
        23: 1.714,
        24: 1.711,
        25: 1.708,
        26: 1.706,
        27: 1.703,
        28: 1.701,
        29: 1.699,
        30: 1.697,
    }
    return t_table.get(df, z)


def _compute_power(n: int, cv_pct: float, ratio: float = 1.0) -> float:
    """Approximate power for 2x2 crossover BE study.

    Power = P(reject both one-sided tests) using normal approximation.
    """
    if n < 2:
        return 0.0
    cv = cv_pct / 100.0
    sigma_w = math.sqrt(math.log(1.0 + cv**2))
    se = sigma_w * math.sqrt(2.0 / n)

    log_ratio = math.log(ratio)
    log_lower = math.log(0.80)
    log_upper = math.log(1.25)

    z_crit = 1.645

    # Power for lower bound test
    z_lower = (log_ratio - log_lower) / se - z_crit
    # Power for upper bound test
    z_upper = (log_upper - log_ratio) / se - z_crit

    # Use normal CDF approximation
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    power = _norm_cdf(z_lower) * _norm_cdf(z_upper) if min(z_lower, z_upper) > -10 else 0.0
    return min(max(power, 0.0), 1.0)


def analyze_be(
    reference_aucs: list[float],
    test_aucs: list[float],
    metric: str = "AUC",
) -> BEResult:
    """Perform TOST bioequivalence analysis on paired PK data.

    Parameters
    ----------
    reference_aucs : Per-subject reference values.
    test_aucs : Per-subject test values.
    metric : Label for the PK metric (e.g., "AUC", "Cmax").

    Returns
    -------
    BEResult

    Raises
    ------
    ValueError
        If fewer than 2 subjects or any value <= 0.
    """
    if len(reference_aucs) < 2 or len(test_aucs) < 2:
        raise ValueError("At least 2 subjects required per group.")

    ref = np.asarray(reference_aucs, dtype=float)
    test = np.asarray(test_aucs, dtype=float)

    if np.any(ref <= 0) or np.any(test <= 0):
        raise ValueError("All PK values must be positive (> 0).")

    n_ref = len(ref)
    n_test = len(test)

    # Log-transform
    log_ref = np.log(ref)
    log_test = np.log(test)

    mean_diff = float(np.mean(log_test)) - float(np.mean(log_ref))

    # SE of difference
    var_ref = float(np.var(log_ref, ddof=1))
    var_test = float(np.var(log_test, ddof=1))
    se = math.sqrt(var_test / n_test + var_ref / n_ref)

    # Degrees of freedom (Welch-Satterthwaite)
    if se > 0:
        df_num = (var_test / n_test + var_ref / n_ref) ** 2
        df_den = (var_test / n_test) ** 2 / (n_test - 1) + (var_ref / n_ref) ** 2 / (n_ref - 1)
        df = max(1, int(df_num / df_den)) if df_den > 0 else max(n_ref + n_test - 2, 1)
    else:
        df = max(n_ref + n_test - 2, 1)

    t_crit = _t_critical(df)

    ci_lower = math.exp(mean_diff - t_crit * se) if se > 0 else math.exp(mean_diff)
    ci_upper = math.exp(mean_diff + t_crit * se) if se > 0 else math.exp(mean_diff)

    gm_ratio = math.exp(mean_diff)

    is_be = (0.80 <= ci_lower) and (ci_upper <= 1.25)

    # CV from pooled within-subject variability
    # For parallel design, use pooled SD
    log_all = np.concatenate([log_ref, log_test])
    pooled_var = float(np.var(log_all, ddof=1))
    cv_pct = (math.exp(math.sqrt(pooled_var)) - 1) * 100.0 if pooled_var > 0 else 0.0

    # Power at observed n
    n_total = n_ref + n_test
    power = _compute_power(n_total // 2, cv_pct, gm_ratio)

    notes = (
        f"BE analysis ({metric}): n_ref={n_ref}, n_test={n_test}. "
        f"GMR={gm_ratio:.4f}, 90% CI=[{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"BE={'PASS' if is_be else 'FAIL'}. CV={cv_pct:.1f}%."
    )

    return BEResult(
        metric=metric,
        n_reference=n_ref,
        n_test=n_test,
        gm_ratio=gm_ratio,
        ci_lower_90=ci_lower,
        ci_upper_90=ci_upper,
        is_bioequivalent=is_be,
        power_at_observed_n=power,
        cv_pct=cv_pct,
        notes=notes,
    )


def calculate_power_curve(
    n_range: list[int],
    cv_pct: float,
    ratio: float = 1.0,
) -> PowerCurveResult:
    """Calculate statistical power across sample sizes.

    Parameters
    ----------
    n_range : List of per-group sample sizes to evaluate.
    cv_pct : Within-subject CV (%).
    ratio : True geometric mean ratio (test/reference).

    Returns
    -------
    PowerCurveResult

    Raises
    ------
    ValueError
        If n_range is empty or cv_pct <= 0.
    """
    if not n_range:
        raise ValueError("n_range must not be empty.")
    if cv_pct <= 0:
        raise ValueError("cv_pct must be > 0.")

    powers = [_compute_power(n, cv_pct, ratio) for n in n_range]

    return PowerCurveResult(
        n_values=list(n_range),
        powers=powers,
        cv_pct=cv_pct,
        ratio=ratio,
    )


def sample_size_for_power(
    cv_pct: float,
    power_target: float = 0.80,
    ratio: float = 1.0,
) -> int:
    """Find minimum per-group sample size to achieve target power.

    Parameters
    ----------
    cv_pct : Within-subject CV (%).
    power_target : Target power (0-1). Default 0.80.
    ratio : True geometric mean ratio.

    Returns
    -------
    int : Minimum per-group sample size.

    Raises
    ------
    ValueError
        If cv_pct <= 0 or power_target not in (0, 1).
    """
    if cv_pct <= 0:
        raise ValueError("cv_pct must be > 0.")
    if not (0.0 < power_target < 1.0):
        raise ValueError("power_target must be between 0 and 1 (exclusive).")

    for n in range(2, 10000):
        if _compute_power(n, cv_pct, ratio) >= power_target:
            return n
    return 10000


__all__ = [
    "BEResult",
    "PowerCurveResult",
    "analyze_be",
    "calculate_power_curve",
    "sample_size_for_power",
]
