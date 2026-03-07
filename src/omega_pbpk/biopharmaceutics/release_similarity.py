"""
Phase 521 — Drug Release Profile Similarity

FDA f1/f2 dissolution profile similarity metrics per FDA guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DissolutionComparisonResult:
    """Result of a dissolution profile comparison."""

    f1: float
    f2: float
    similar: bool  # f2 >= 50
    acceptable_f1: bool  # f1 <= 15
    reference_weibull: dict
    test_weibull: dict
    notes: str


def _validate_pct_list(values: list[float], name: str) -> None:
    """Validate that a list of percentage values is within [0, 100]."""
    for v in values:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"{name} values must be in [0, 100]; got {v}")


def f2_similarity(reference_pct: list[float], test_pct: list[float]) -> float:
    """
    Calculate FDA f2 similarity factor.

    f2 = 50 * log10((1 + (1/n) * sum((Rt - Tt)^2))^(-0.5) * 100)

    Parameters
    ----------
    reference_pct:
        Percent dissolved for reference product at each timepoint.
    test_pct:
        Percent dissolved for test product at each timepoint.

    Returns
    -------
    float
        f2 value; f2 >= 50 indicates similar profiles.
    """
    if len(reference_pct) != len(test_pct):
        raise ValueError(
            f"reference_pct and test_pct must have the same length; "
            f"got {len(reference_pct)} and {len(test_pct)}"
        )
    _validate_pct_list(reference_pct, "reference_pct")
    _validate_pct_list(test_pct, "test_pct")

    n = len(reference_pct)
    if n == 0:
        raise ValueError("Input lists must not be empty.")

    sum_sq = sum((r - t) ** 2 for r, t in zip(reference_pct, test_pct))
    inner = 1.0 + sum_sq / n
    return 50.0 * math.log10(inner ** (-0.5) * 100.0)


def f1_difference(reference_pct: list[float], test_pct: list[float]) -> float:
    """
    Calculate FDA f1 difference factor.

    f1 = (sum|Rt - Tt|) / sum(Rt) * 100

    Parameters
    ----------
    reference_pct:
        Percent dissolved for reference product at each timepoint.
    test_pct:
        Percent dissolved for test product at each timepoint.

    Returns
    -------
    float
        f1 value; f1 <= 15 indicates acceptable similarity.
    """
    if len(reference_pct) != len(test_pct):
        raise ValueError(
            f"reference_pct and test_pct must have the same length; "
            f"got {len(reference_pct)} and {len(test_pct)}"
        )
    _validate_pct_list(reference_pct, "reference_pct")
    _validate_pct_list(test_pct, "test_pct")

    n = len(reference_pct)
    if n == 0:
        raise ValueError("Input lists must not be empty.")

    sum_abs_diff = sum(abs(r - t) for r, t in zip(reference_pct, test_pct))
    sum_ref = sum(reference_pct)

    if sum_ref == 0.0:
        return 0.0

    return sum_abs_diff / sum_ref * 100.0


def weibull_fit(times: list[float], fractions_dissolved: list[float]) -> dict:
    """
    Fit Weibull dissolution model to observed data via grid search.

    Weibull: F(t) = 1 - exp(-(t/alpha)^beta)

    Uses a 20x20 grid search over alpha in [0.1, max_t*2] and beta in [0.1, 5.0].

    Parameters
    ----------
    times:
        Time points (hours or minutes).
    fractions_dissolved:
        Fraction (0-1) dissolved at each timepoint.

    Returns
    -------
    dict with keys: alpha, beta, r_squared, td50
    """
    if len(times) != len(fractions_dissolved):
        raise ValueError("times and fractions_dissolved must have the same length.")
    if len(times) == 0:
        raise ValueError("Input lists must not be empty.")

    max_t = max(times) if times else 1.0

    alpha_candidates = [0.1 + (max_t * 2 - 0.1) * i / 19 for i in range(20)]
    beta_candidates = [0.1 + (5.0 - 0.1) * i / 19 for i in range(20)]

    best_sse = float("inf")
    best_alpha = alpha_candidates[0]
    best_beta = beta_candidates[0]

    for alpha in alpha_candidates:
        for beta in beta_candidates:
            sse = 0.0
            for t, f_obs in zip(times, fractions_dissolved):
                try:
                    f_pred = 1.0 - math.exp(-((t / alpha) ** beta))
                except (OverflowError, ZeroDivisionError):
                    f_pred = 1.0
                sse += (f_obs - f_pred) ** 2
            if sse < best_sse:
                best_sse = sse
                best_alpha = alpha
                best_beta = beta

    # Compute R-squared
    mean_f = sum(fractions_dissolved) / len(fractions_dissolved)
    ss_tot = sum((f - mean_f) ** 2 for f in fractions_dissolved)

    if ss_tot == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - best_sse / ss_tot

    # t50: time at which 50% dissolved → F(td50) = 0.5
    # 0.5 = 1 - exp(-(td50/alpha)^beta) → td50 = alpha * (ln(2))^(1/beta)
    td50 = best_alpha * (math.log(2.0) ** (1.0 / best_beta))

    return {
        "alpha": best_alpha,
        "beta": best_beta,
        "r_squared": r_squared,
        "td50": td50,
    }


def _interpolate(x_query: float, x_vals: list[float], y_vals: list[float]) -> float:
    """Linear interpolation; clamps to boundary values outside range."""
    if x_query <= x_vals[0]:
        return y_vals[0]
    if x_query >= x_vals[-1]:
        return y_vals[-1]
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x_query <= x_vals[i + 1]:
            t = (x_query - x_vals[i]) / (x_vals[i + 1] - x_vals[i])
            return y_vals[i] + t * (y_vals[i + 1] - y_vals[i])
    return y_vals[-1]


def compare_dissolution_profiles(
    reference_times: list[float],
    reference_pct: list[float],
    test_times: list[float],
    test_pct: list[float],
    specification_times: list[float] | None = None,
    specification_min_pct: list[float] | None = None,
) -> DissolutionComparisonResult:
    """
    Compare two dissolution profiles with f1/f2 and Weibull fitting.

    Test profile is interpolated onto reference timepoints before comparison.

    Parameters
    ----------
    reference_times:
        Time points for the reference profile.
    reference_pct:
        Percent dissolved for reference at each timepoint.
    test_times:
        Time points for the test profile.
    test_pct:
        Percent dissolved for test at each timepoint.
    specification_times:
        Optional specification timepoints.
    specification_min_pct:
        Optional minimum % dissolved at each specification timepoint.

    Returns
    -------
    DissolutionComparisonResult
    """
    _validate_pct_list(reference_pct, "reference_pct")
    _validate_pct_list(test_pct, "test_pct")

    # Interpolate test profile onto reference timepoints
    test_pct_interp = [_interpolate(t, test_times, test_pct) for t in reference_times]

    f1 = f1_difference(reference_pct, test_pct_interp)
    f2 = f2_similarity(reference_pct, test_pct_interp)

    similar = f2 >= 50.0
    acceptable_f1 = f1 <= 15.0

    # Weibull fitting (using fractions 0-1)
    ref_fractions = [v / 100.0 for v in reference_pct]
    tst_fractions = [v / 100.0 for v in test_pct]

    ref_weibull = weibull_fit(reference_times, ref_fractions)
    tst_weibull = weibull_fit(test_times, tst_fractions)

    notes_parts: list[str] = []
    if similar:
        notes_parts.append("Profiles are similar (f2 >= 50).")
    else:
        notes_parts.append("Profiles are NOT similar (f2 < 50).")

    if acceptable_f1:
        notes_parts.append("f1 is acceptable (<= 15).")
    else:
        notes_parts.append("f1 exceeds acceptable limit (> 15).")

    # Spec check
    if specification_times is not None and specification_min_pct is not None:
        spec_pass = True
        for spec_t, spec_min in zip(specification_times, specification_min_pct):
            test_val = _interpolate(spec_t, test_times, test_pct)
            if test_val < spec_min:
                spec_pass = False
                notes_parts.append(
                    f"Test profile fails spec at t={spec_t}: "
                    f"{test_val:.1f}% < {spec_min:.1f}% required."
                )
        if spec_pass:
            notes_parts.append("Test profile meets all specifications.")

    return DissolutionComparisonResult(
        f1=f1,
        f2=f2,
        similar=similar,
        acceptable_f1=acceptable_f1,
        reference_weibull=ref_weibull,
        test_weibull=tst_weibull,
        notes=" ".join(notes_parts),
    )
