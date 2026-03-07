"""Formulation Bioequivalence Predictor — Phase 288.

Predict bioequivalence outcomes for different formulations using statistical
TOST (Two One-Sided Tests) methodology without running a full crossover trial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import t as t_dist

__all__ = [
    "BEPredictionResult",
    "predict_be_outcome",
    "formulation_sensitivity",
    "required_sample_size",
]

# BE acceptance criteria
_BE_LOWER: float = 0.80
_BE_UPPER: float = 1.25


@dataclass(frozen=True)
class BEPredictionResult:
    """Bioequivalence prediction for a test vs reference formulation."""

    gmr_auc: float
    gmr_cmax: float
    ci90_auc: tuple[float, float]
    ci90_cmax: tuple[float, float]
    be_auc: bool
    be_cmax: bool
    overall_be: bool
    n_subjects: int
    notes: str


def predict_be_outcome(
    ref_auc: float,
    ref_cmax: float,
    test_auc: float,
    test_cmax: float,
    ref_auc_cv: float = 0.20,
    ref_cmax_cv: float = 0.20,
    n_subjects: int = 12,
) -> BEPredictionResult:
    """Predict bioequivalence outcome for a test formulation vs reference.

    Uses the log-scale TOST 90% CI approach:
        ln(GMR) ± t(0.10, df=n-1) * CV_ln / sqrt(n)

    where CV_ln = sqrt(ln(1 + CV^2)).

    Parameters
    ----------
    ref_auc:
        Reference formulation AUC (any consistent units).
    ref_cmax:
        Reference formulation Cmax.
    test_auc:
        Test formulation AUC.
    test_cmax:
        Test formulation Cmax.
    ref_auc_cv:
        Within-subject CV for AUC (proportion, e.g. 0.20 = 20%).
    ref_cmax_cv:
        Within-subject CV for Cmax.
    n_subjects:
        Number of subjects in the study.

    Returns
    -------
    BEPredictionResult
    """
    if ref_auc <= 0:
        raise ValueError("ref_auc must be > 0")
    if ref_cmax <= 0:
        raise ValueError("ref_cmax must be > 0")
    if test_auc <= 0:
        raise ValueError("test_auc must be > 0")
    if test_cmax <= 0:
        raise ValueError("test_cmax must be > 0")
    if ref_auc_cv <= 0:
        raise ValueError("ref_auc_cv must be > 0")
    if ref_cmax_cv <= 0:
        raise ValueError("ref_cmax_cv must be > 0")
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")

    gmr_auc = test_auc / ref_auc
    gmr_cmax = test_cmax / ref_cmax

    df = n_subjects - 1
    # t critical value at alpha=0.10 (one-sided), for 90% CI
    t_crit = float(t_dist.ppf(0.90, df))

    def _compute_ci(
        gmr: float, cv: float
    ) -> tuple[tuple[float, float], bool]:
        sigma_ln = math.sqrt(math.log(1.0 + cv**2))
        half_width = t_crit * sigma_ln / math.sqrt(n_subjects)
        ln_gmr = math.log(gmr)
        ci_lo = math.exp(ln_gmr - half_width)
        ci_hi = math.exp(ln_gmr + half_width)
        within_be = ci_lo >= _BE_LOWER and ci_hi <= _BE_UPPER
        return (ci_lo, ci_hi), within_be

    ci_auc, be_auc = _compute_ci(gmr_auc, ref_auc_cv)
    ci_cmax, be_cmax = _compute_ci(gmr_cmax, ref_cmax_cv)
    overall_be = be_auc and be_cmax

    verdict = "BIOEQUIVALENT" if overall_be else "NOT BIOEQUIVALENT"
    notes = (
        f"{verdict}. "
        f"GMR AUC={gmr_auc:.3f} 90%CI=[{ci_auc[0]:.3f},{ci_auc[1]:.3f}] "
        f"({'PASS' if be_auc else 'FAIL'}); "
        f"GMR Cmax={gmr_cmax:.3f} 90%CI=[{ci_cmax[0]:.3f},{ci_cmax[1]:.3f}] "
        f"({'PASS' if be_cmax else 'FAIL'}). "
        f"n={n_subjects}."
    )

    return BEPredictionResult(
        gmr_auc=gmr_auc,
        gmr_cmax=gmr_cmax,
        ci90_auc=ci_auc,
        ci90_cmax=ci_cmax,
        be_auc=be_auc,
        be_cmax=be_cmax,
        overall_be=overall_be,
        n_subjects=n_subjects,
        notes=notes,
    )


def formulation_sensitivity(
    ref_params: dict,
    test_variants: list[dict],
) -> list[BEPredictionResult]:
    """Screen multiple test formulation variants against a reference.

    Parameters
    ----------
    ref_params:
        Reference formulation parameters with keys:
        ``ref_auc``, ``ref_cmax``, and optionally
        ``ref_auc_cv`` (default 0.20), ``ref_cmax_cv`` (default 0.20),
        ``n_subjects`` (default 12).
    test_variants:
        List of dicts, each with keys ``test_auc`` and ``test_cmax``,
        and optionally the same CV/n overrides.

    Returns
    -------
    list[BEPredictionResult]
        One result per test variant.
    """
    ref_auc = float(ref_params["ref_auc"])
    ref_cmax = float(ref_params["ref_cmax"])
    default_cv_auc = float(ref_params.get("ref_auc_cv", 0.20))
    default_cv_cmax = float(ref_params.get("ref_cmax_cv", 0.20))
    default_n = int(ref_params.get("n_subjects", 12))

    results: list[BEPredictionResult] = []
    for variant in test_variants:
        test_auc = float(variant["test_auc"])
        test_cmax = float(variant["test_cmax"])
        cv_auc = float(variant.get("ref_auc_cv", default_cv_auc))
        cv_cmax = float(variant.get("ref_cmax_cv", default_cv_cmax))
        n = int(variant.get("n_subjects", default_n))

        result = predict_be_outcome(
            ref_auc=ref_auc,
            ref_cmax=ref_cmax,
            test_auc=test_auc,
            test_cmax=test_cmax,
            ref_auc_cv=cv_auc,
            ref_cmax_cv=cv_cmax,
            n_subjects=n,
        )
        results.append(result)

    return results


def required_sample_size(
    gmr: float,
    cv: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Estimate sample size needed for a BE study using TOST power approximation.

    Uses the Owen Q-function approximation:
        n ≈ 2 * ((z_alpha + z_beta)^2 * 2 * CV_ln^2) / (ln(GMR))^2

    where z_alpha = z(alpha), z_beta = z(1-power), CV_ln = sqrt(ln(1 + CV^2)).

    Parameters
    ----------
    gmr:
        Assumed geometric mean ratio (test/reference, e.g. 0.95).
    cv:
        Within-subject coefficient of variation (proportion).
    power:
        Desired power (0 < power < 1), default 0.80.
    alpha:
        One-sided significance level, default 0.05.

    Returns
    -------
    int
        Estimated sample size clamped to [2, 200].
    """
    if gmr <= 0:
        raise ValueError("gmr must be > 0")
    if cv <= 0:
        raise ValueError("cv must be > 0")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1)")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")

    from scipy.stats import norm

    z_alpha = float(norm.ppf(1.0 - alpha))
    z_beta = float(norm.ppf(power))

    sigma_ln = math.sqrt(math.log(1.0 + cv**2))

    # Effect size: distance from GMR to nearest BE boundary on log scale
    delta = min(
        abs(math.log(gmr) - math.log(_BE_LOWER)),
        abs(math.log(_BE_UPPER) - math.log(gmr)),
    )

    if delta <= 0:
        return 200

    # Owen approximation: n = 2 * (z_alpha + z_beta)^2 * sigma_ln^2 / delta^2
    n_float = 2.0 * (z_alpha + z_beta) ** 2 * sigma_ln**2 / delta**2
    # Round up to nearest even integer (crossover design)
    n = int(math.ceil(n_float))
    if n % 2 != 0:
        n += 1

    return max(2, min(200, n))
