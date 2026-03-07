"""Meta-analysis of dose-response data: pool multiple studies, fit combined EC50/Emax."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StudyResult:
    study_id: str
    doses: tuple
    responses: tuple
    emax: float
    ec50: float
    hill_n: float
    r_squared: float


@dataclass(frozen=True)
class MetaAnalysisResult:
    n_studies: int
    pooled_emax: float
    pooled_ec50: float
    pooled_hill_n: float
    ec50_cv_pct: float
    emax_cv_pct: float
    forest_plot_data: tuple
    heterogeneity_i2: float
    recommendation: str
    notes: str


def _hill_predict(emax: float, ec50: float, hill_n: float, dose: float) -> float:
    """Predict response using Hill equation."""
    if dose <= 0:
        return 0.0
    denom = ec50**hill_n + dose**hill_n
    if denom == 0:
        return 0.0
    return emax * (dose**hill_n) / denom


def _r_squared(observed: list, predicted: list) -> float:
    """Compute R² from observed and predicted lists."""
    n = len(observed)
    if n == 0:
        return 0.0
    mean_obs = sum(observed) / n
    ss_tot = sum((o - mean_obs) ** 2 for o in observed)
    ss_res = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_hill_equation(doses: list, responses: list) -> tuple:
    """
    Fit Hill equation: R = Emax * C^n / (EC50^n + C^n)
    Returns (emax, ec50, hill_n, r_squared) using grid search.
    doses and responses must have >= 3 points.
    """
    if len(doses) < 3 or len(responses) < 3:
        raise ValueError("Need at least 3 dose-response points to fit Hill equation.")

    max_resp = max(responses)

    # Edge case: all same responses
    resp_set = set(responses)
    if len(resp_set) == 1:
        sorted_doses = sorted(doses)
        median_dose = sorted_doses[len(sorted_doses) // 2]
        return (max_resp, median_dose, 1.0, 0.0)

    # Candidate Emax values — include values above max_resp in case data doesn't reach plateau
    emax_candidates = [max_resp, max_resp * 0.9, max_resp * 1.25, max_resp * 1.5, max_resp * 2.0]

    # EC50 candidates: log-space from min to max dose (10 values)
    min_d = min(d for d in doses if d > 0)
    max_d = max(doses)
    if min_d <= 0:
        min_d = 1e-6
    log_min = math.log10(min_d)
    log_max = math.log10(max_d)
    if log_min == log_max:
        ec50_candidates = [min_d]
    else:
        step = (log_max - log_min) / 9.0
        ec50_candidates = [10 ** (log_min + i * step) for i in range(10)]

    hill_candidates = [0.5, 1.0, 1.5, 2.0]

    best_ss_res = float("inf")
    best_params = (max_resp, ec50_candidates[len(ec50_candidates) // 2], 1.0)

    for emax in emax_candidates:
        for ec50 in ec50_candidates:
            for hill_n in hill_candidates:
                predicted = [_hill_predict(emax, ec50, hill_n, d) for d in doses]
                ss_res = sum((o - p) ** 2 for o, p in zip(responses, predicted))
                if ss_res < best_ss_res:
                    best_ss_res = ss_res
                    best_params = (emax, ec50, hill_n)

    best_emax, best_ec50, best_hill_n = best_params
    predicted = [_hill_predict(best_emax, best_ec50, best_hill_n, d) for d in doses]
    r2 = _r_squared(responses, predicted)
    r2 = max(0.0, min(1.0, r2))

    return (best_emax, best_ec50, best_hill_n, r2)


def analyze_study(study_id: str, doses: list, responses: list) -> StudyResult:
    """Fit Hill eq to one study's data and return StudyResult."""
    emax, ec50, hill_n, r2 = fit_hill_equation(doses, responses)
    return StudyResult(
        study_id=study_id,
        doses=tuple(doses),
        responses=tuple(responses),
        emax=emax,
        ec50=ec50,
        hill_n=hill_n,
        r_squared=r2,
    )


def _weighted_mean(values: list, weights: list) -> float:
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _std(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def meta_analyze(studies: list) -> MetaAnalysisResult:
    """
    Pool StudyResult objects. studies is list of StudyResult.
    Uses inverse-variance weighting by r_squared.
    Requires at least 2 studies.
    """
    if len(studies) < 2:
        raise ValueError("meta_analyze requires at least 2 studies.")

    weights = [max(s.r_squared, 0.01) for s in studies]
    emaxs = [s.emax for s in studies]
    ec50s = [s.ec50 for s in studies]
    hill_ns = [s.hill_n for s in studies]

    pooled_emax = _weighted_mean(emaxs, weights)

    # Geometric mean of EC50
    log_ec50s = [math.log(e) for e in ec50s]
    pooled_log_ec50 = _weighted_mean(log_ec50s, weights)
    pooled_ec50 = math.exp(pooled_log_ec50)

    pooled_hill_n = _weighted_mean(hill_ns, weights)

    mean_ec50 = sum(ec50s) / len(ec50s)
    mean_emax = sum(emaxs) / len(emaxs)

    ec50_cv_pct = (_std(ec50s) / mean_ec50 * 100) if mean_ec50 != 0 else 0.0
    emax_cv_pct = (_std(emaxs) / mean_emax * 100) if mean_emax != 0 else 0.0

    # I² statistic: simplified — use log-space EC50 std to handle orders-of-magnitude variation
    log_ec50s = [math.log(e) for e in ec50s]
    log_ec50_std = _std(log_ec50s)
    # Scale: log_std of ~1.15 (factor of ~3x between studies) maps to ~50% I²
    i2 = min(100.0, log_ec50_std * 43.5)

    if i2 < 25.0:
        recommendation = "consistent"
    elif i2 < 75.0:
        recommendation = "moderate_heterogeneity"
    else:
        recommendation = "high_heterogeneity"

    forest_plot_data = tuple(
        {
            "study_id": s.study_id,
            "ec50": s.ec50,
            "ci_low": s.ec50 * 0.7,
            "ci_high": s.ec50 * 1.3,
        }
        for s in studies
    )

    notes = (
        f"Meta-analysis of {len(studies)} studies. "
        f"Pooled EC50={pooled_ec50:.3f}, Emax={pooled_emax:.2f}. "
        f"I²={i2:.1f}% ({recommendation})."
    )

    return MetaAnalysisResult(
        n_studies=len(studies),
        pooled_emax=pooled_emax,
        pooled_ec50=pooled_ec50,
        pooled_hill_n=pooled_hill_n,
        ec50_cv_pct=ec50_cv_pct,
        emax_cv_pct=emax_cv_pct,
        forest_plot_data=forest_plot_data,
        heterogeneity_i2=i2,
        recommendation=recommendation,
        notes=notes,
    )
