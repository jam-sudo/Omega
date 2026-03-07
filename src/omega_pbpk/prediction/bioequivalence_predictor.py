"""Bioequivalence (BE) outcome prediction from PK parameters.

Predicts 90% confidence intervals for AUC and Cmax ratios (test/reference),
evaluates FDA 80–125% BE criterion, and estimates the required sample size.

Reference: FDA Guidance for Industry — Statistical Approaches to Establishing
Bioequivalence (2001); FDA BE Guidance (2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BEPredictionResult:
    """Result of bioequivalence prediction from summary PK parameters."""

    ref_auc: float
    test_auc: float
    ref_cmax: float
    test_cmax: float
    gmr_auc: float
    gmr_cmax: float
    ci_auc_lower: float
    ci_auc_upper: float
    ci_cmax_lower: float
    ci_cmax_upper: float
    be_auc_pass: bool
    be_cmax_pass: bool
    overall_be_pass: bool
    n_subjects: int
    intra_cv_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _t_quantile_approx(df: int) -> float:
    """Approximate one-sided t_{df, 0.05} (upper 5th percentile).

    Uses the simple formula:
        t ≈ 1.645 + 0.02 * max(0, 30 - df)

    This matches 1.645 (normal approx) for large df and increases for small df.
    """
    return 1.645 + 0.02 * max(0, 30 - df)


def _ci_bounds(gmr: float, intra_cv_frac: float, n: int) -> tuple[float, float]:
    """Compute 90% CI bounds for a geometric mean ratio.

    The log-scale CI half-width is:
        hw = t_{n-2, 0.05} * sigma / sqrt(n/2)
    where sigma = intra_cv_frac (within-subject CV as a fraction).

    Returns (lower, upper) on the ratio scale.
    """
    df = max(n - 2, 1)
    t = _t_quantile_approx(df)
    hw = t * intra_cv_frac / math.sqrt(n / 2.0)
    log_gmr = math.log(gmr)
    lower = math.exp(log_gmr - hw)
    upper = math.exp(log_gmr + hw)
    return lower, upper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_be_outcome(
    ref_auc: float,
    test_auc: float,
    ref_cmax: float,
    test_cmax: float,
    n_subjects: int,
    intra_cv_pct: float,
    be_limits: tuple[float, float] = (0.80, 1.25),
) -> BEPredictionResult:
    """Predict bioequivalence outcome from summary PK parameters.

    Parameters
    ----------
    ref_auc:
        Reference formulation AUC (any consistent units, e.g. mg·h/L).
    test_auc:
        Test formulation AUC (same units as ref_auc).
    ref_cmax:
        Reference formulation Cmax (mg/L or consistent units).
    test_cmax:
        Test formulation Cmax (same units as ref_cmax).
    n_subjects:
        Number of subjects in the crossover study.
    intra_cv_pct:
        Intra-subject coefficient of variation in percent (e.g. 20 for 20%).
    be_limits:
        (lower, upper) acceptance limits for the 90% CI. Default (0.80, 1.25).

    Returns
    -------
    BEPredictionResult
    """
    # --- Input validation ---
    if ref_auc <= 0:
        raise ValueError("ref_auc must be > 0")
    if test_auc <= 0:
        raise ValueError("test_auc must be > 0")
    if ref_cmax <= 0:
        raise ValueError("ref_cmax must be > 0")
    if test_cmax <= 0:
        raise ValueError("test_cmax must be > 0")
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")
    if intra_cv_pct <= 0:
        raise ValueError("intra_cv_pct must be > 0")
    if len(be_limits) != 2:
        raise ValueError("be_limits must be a 2-tuple (lower, upper)")
    lo_lim, hi_lim = be_limits
    if lo_lim <= 0 or hi_lim <= lo_lim:
        raise ValueError("be_limits must satisfy 0 < lower < upper")

    intra_cv_frac = intra_cv_pct / 100.0

    # Geometric mean ratios
    gmr_auc = test_auc / ref_auc
    gmr_cmax = test_cmax / ref_cmax

    # 90% CI
    ci_auc_lower, ci_auc_upper = _ci_bounds(gmr_auc, intra_cv_frac, n_subjects)
    ci_cmax_lower, ci_cmax_upper = _ci_bounds(gmr_cmax, intra_cv_frac, n_subjects)

    # BE pass criteria
    be_auc_pass = (ci_auc_lower >= lo_lim) and (ci_auc_upper <= hi_lim)
    be_cmax_pass = (ci_cmax_lower >= lo_lim) and (ci_cmax_upper <= hi_lim)
    overall_be_pass = be_auc_pass and be_cmax_pass

    # Notes
    notes_list: list[str] = []
    if abs(gmr_auc - 1.0) > 0.20:
        notes_list.append(f"GMR_AUC={gmr_auc:.3f} deviates >20% from unity")
    if abs(gmr_cmax - 1.0) > 0.20:
        notes_list.append(f"GMR_Cmax={gmr_cmax:.3f} deviates >20% from unity")
    if intra_cv_pct > 30:
        notes_list.append(
            f"High intra-subject CV ({intra_cv_pct}%); consider reference-scaled approach"
        )
    if not overall_be_pass:
        notes_list.append("Study failed BE criteria; consider dose adjustment or larger N")
    notes = "; ".join(notes_list) if notes_list else "BE criteria met"

    return BEPredictionResult(
        ref_auc=ref_auc,
        test_auc=test_auc,
        ref_cmax=ref_cmax,
        test_cmax=test_cmax,
        gmr_auc=gmr_auc,
        gmr_cmax=gmr_cmax,
        ci_auc_lower=ci_auc_lower,
        ci_auc_upper=ci_auc_upper,
        ci_cmax_lower=ci_cmax_lower,
        ci_cmax_upper=ci_cmax_upper,
        be_auc_pass=be_auc_pass,
        be_cmax_pass=be_cmax_pass,
        overall_be_pass=overall_be_pass,
        n_subjects=n_subjects,
        intra_cv_pct=intra_cv_pct,
        notes=notes,
    )


def sample_size_for_be(
    target_gmr: float,
    intra_cv_pct: float,
    power: float = 0.80,
    be_limits: tuple[float, float] = (0.80, 1.25),
) -> int:
    """Estimate sample size required to achieve a given BE power.

    Uses an iterative approach: for each candidate N (12, 14, ..., 200),
    compute the approximate probability that the 90% CI falls within the
    BE limits and return the first N that meets the power requirement.

    The normal approximation for power is:
        power ≈ Phi((log(upper_limit) - |log(GMR)|) / sigma_n - z_alpha)
               + Phi((log(lower_limit) + |log(GMR)|) / sigma_n - z_alpha)  (two one-sided)

    where sigma_n = intra_cv_frac / sqrt(n/2), z_alpha = 1.645.

    Parameters
    ----------
    target_gmr:
        Expected geometric mean ratio (test/reference), e.g. 0.95.
    intra_cv_pct:
        Intra-subject coefficient of variation in percent.
    power:
        Desired minimum power (probability) to demonstrate BE. Default 0.80.
    be_limits:
        BE acceptance limits (lower, upper). Default (0.80, 1.25).

    Returns
    -------
    int
        Estimated minimum number of subjects (even, minimum 12).
    """
    if target_gmr <= 0:
        raise ValueError("target_gmr must be > 0")
    if intra_cv_pct <= 0:
        raise ValueError("intra_cv_pct must be > 0")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1)")
    if len(be_limits) != 2:
        raise ValueError("be_limits must be a 2-tuple (lower, upper)")
    lo_lim, hi_lim = be_limits
    if lo_lim <= 0 or hi_lim <= lo_lim:
        raise ValueError("be_limits must satisfy 0 < lower < upper")

    intra_cv_frac = intra_cv_pct / 100.0
    log_gmr = math.log(target_gmr)
    log_lo = math.log(lo_lim)
    log_hi = math.log(hi_lim)
    z_alpha = 1.645  # one-sided alpha = 0.05

    def _phi(x: float) -> float:
        """Standard normal CDF via math.erfc approximation."""
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def _power_at_n(n: int) -> float:
        """Approximate TOST power for sample size n."""
        sigma_n = intra_cv_frac / math.sqrt(n / 2.0)
        if sigma_n <= 0:
            return 0.0
        # Two one-sided tests
        p1 = _phi((log_hi - log_gmr) / sigma_n - z_alpha)
        p2 = _phi((-log_lo + log_gmr) / sigma_n - z_alpha)
        # TOST power ≈ p1 + p2 - 1 (when both individually > 0.5)
        tost_power = p1 + p2 - 1.0
        return max(0.0, min(1.0, tost_power))

    # Iterate over candidate sample sizes (even numbers, minimum 12)
    for n in range(12, 202, 2):
        if _power_at_n(n) >= power:
            return n

    # If no N in range meets power, return 200 as maximum
    return 200


__all__ = [
    "BEPredictionResult",
    "predict_be_outcome",
    "sample_size_for_be",
]
