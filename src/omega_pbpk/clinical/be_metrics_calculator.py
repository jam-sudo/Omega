"""
Phase 900 — Drug Bioequivalence Metric Calculator

Calculates regulatory bioequivalence metrics (AUC ratio, Cmax ratio, 90% CI)
from PK simulation data following FDA/EMA guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BEMetricsResult:
    """Result of bioequivalence metric calculation."""

    drug_name: str
    test_formulation: str
    reference_formulation: str
    test_auc_t: float
    ref_auc_t: float
    test_auc_inf: float
    ref_auc_inf: float
    test_cmax: float
    ref_cmax: float
    test_tmax_h: float
    ref_tmax_h: float
    auc_ratio: float
    cmax_ratio: float
    auc_90ci_lower: float
    auc_90ci_upper: float
    cmax_90ci_lower: float
    cmax_90ci_upper: float
    be_conclusion: str
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _estimate_ke(times: list[float], concs: list[float]) -> float:
    """
    Estimate terminal elimination rate constant from the last 3 points
    using log-linear regression.

    Returns ke > 0 on success, or -1.0 if estimation fails.
    """
    c_last3 = concs[-3:]
    t_last3 = times[-3:]

    # All three concentrations must be positive
    if any(c <= 0.0 for c in c_last3):
        return -1.0

    # ke from first and last of the three points
    delta_t = t_last3[2] - t_last3[0]
    if delta_t <= 0.0:
        return -1.0

    ke = (math.log(c_last3[0]) - math.log(c_last3[2])) / delta_t
    return ke if ke > 0.0 else -1.0


def _compute_pk(
    times: list[float],
    concs: list[float],
) -> tuple[float, float, float, float]:
    """
    Compute AUCt, AUCinf, Cmax, Tmax from a PK profile.

    Returns (auc_t, auc_inf, cmax, tmax_h).
    """
    auc_t = _trapezoidal_auc(times, concs)
    cmax = max(concs)
    tmax_h = times[concs.index(cmax)]

    ke = _estimate_ke(times, concs)
    c_last = concs[-1]

    if ke > 0.0 and c_last > 0.0:
        auc_inf = auc_t + c_last / ke
    else:
        auc_inf = auc_t

    return auc_t, auc_inf, cmax, tmax_h


def calculate_be_metrics(
    drug_name: str,
    test_formulation: str,
    reference_formulation: str,
    test_times_h: list[float],
    test_conc_mg_L: list[float],
    ref_times_h: list[float],
    ref_conc_mg_L: list[float],
    cv_intra_pct: float = 15.0,
) -> BEMetricsResult:
    """
    Calculate regulatory bioequivalence metrics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    test_formulation:
        Label for the test formulation.
    reference_formulation:
        Label for the reference formulation.
    test_times_h:
        Sampling time points (hours) for the test formulation.
    test_conc_mg_L:
        Plasma concentrations (mg/L) for the test formulation.
    ref_times_h:
        Sampling time points (hours) for the reference formulation.
    ref_conc_mg_L:
        Plasma concentrations (mg/L) for the reference formulation.
    cv_intra_pct:
        Intrasubject CV (%) for 90% CI estimation (default 15%).

    Returns
    -------
    BEMetricsResult
    """
    # --- Validation ---
    if len(test_times_h) != len(test_conc_mg_L):
        raise ValueError("test_times_h and test_conc_mg_L must have the same length")
    if len(ref_times_h) != len(ref_conc_mg_L):
        raise ValueError("ref_times_h and ref_conc_mg_L must have the same length")
    if len(test_times_h) < 3:
        raise ValueError("test_times_h and test_conc_mg_L must have at least 3 data points")
    if len(ref_times_h) < 3:
        raise ValueError("ref_times_h and ref_conc_mg_L must have at least 3 data points")

    # --- PK computation ---
    test_auc_t, test_auc_inf, test_cmax, test_tmax_h = _compute_pk(test_times_h, test_conc_mg_L)
    ref_auc_t, ref_auc_inf, ref_cmax, ref_tmax_h = _compute_pk(ref_times_h, ref_conc_mg_L)

    # --- Ratios ---
    auc_ratio = test_auc_inf / ref_auc_inf if ref_auc_inf > 0.0 else 0.0
    cmax_ratio = test_cmax / ref_cmax if ref_cmax > 0.0 else 0.0

    # --- 90% CI estimation (approximate, CV-based) ---
    half_width = 1.645 * cv_intra_pct / 100.0
    auc_90ci_lower = auc_ratio * (1.0 - half_width)
    auc_90ci_upper = auc_ratio * (1.0 + half_width)
    cmax_90ci_lower = cmax_ratio * (1.0 - half_width)
    cmax_90ci_upper = cmax_ratio * (1.0 + half_width)

    # --- BE conclusion ---
    auc_in_range = 0.8 <= auc_90ci_lower and auc_90ci_upper <= 1.25
    cmax_in_range = 0.8 <= cmax_90ci_lower and cmax_90ci_upper <= 1.25
    clearly_not_be = auc_ratio < 0.5 or auc_ratio > 2.0 or cmax_ratio < 0.5 or cmax_ratio > 2.0

    if auc_in_range and cmax_in_range:
        be_conclusion = "BE demonstrated"
    elif clearly_not_be:
        be_conclusion = "BE not demonstrated"
    else:
        be_conclusion = "Inconclusive — additional data needed"

    # --- Notes ---
    if be_conclusion == "BE demonstrated":
        notes = "Test and reference formulations meet regulatory BE criteria (80-125%)"
    elif be_conclusion == "BE not demonstrated":
        notes = "Significant PK difference — reformulation or additional studies required"
    else:
        notes = "Borderline BE — consider crossover study with larger n"

    return BEMetricsResult(
        drug_name=drug_name,
        test_formulation=test_formulation,
        reference_formulation=reference_formulation,
        test_auc_t=test_auc_t,
        ref_auc_t=ref_auc_t,
        test_auc_inf=test_auc_inf,
        ref_auc_inf=ref_auc_inf,
        test_cmax=test_cmax,
        ref_cmax=ref_cmax,
        test_tmax_h=test_tmax_h,
        ref_tmax_h=ref_tmax_h,
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        auc_90ci_lower=auc_90ci_lower,
        auc_90ci_upper=auc_90ci_upper,
        cmax_90ci_lower=cmax_90ci_lower,
        cmax_90ci_upper=cmax_90ci_upper,
        be_conclusion=be_conclusion,
        notes=notes,
    )


def compare_formulation_be(
    drug_name: str,
    reference_times: list[float],
    reference_conc: list[float],
    test_formulations: list[dict],
) -> list[BEMetricsResult]:
    """
    Compare multiple test formulations against a common reference.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    reference_times:
        Sampling time points (hours) for the reference formulation.
    reference_conc:
        Plasma concentrations (mg/L) for the reference formulation.
    test_formulations:
        List of dicts with keys: "name", "times_h", "conc_mg_L".

    Returns
    -------
    List of BEMetricsResult sorted by auc_ratio descending.
    """
    results: list[BEMetricsResult] = []
    for tf in test_formulations:
        result = calculate_be_metrics(
            drug_name=drug_name,
            test_formulation=tf["name"],
            reference_formulation="Reference",
            test_times_h=tf["times_h"],
            test_conc_mg_L=tf["conc_mg_L"],
            ref_times_h=reference_times,
            ref_conc_mg_L=reference_conc,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_ratio, reverse=True)
    return results
