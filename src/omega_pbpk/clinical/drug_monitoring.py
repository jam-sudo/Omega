"""Therapeutic drug monitoring (TDM) dashboard utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "TDMResult",
    "assess_tdm",
    "tdm_dashboard",
    "population_tdm_summary",
]


@dataclass(frozen=True)
class TDMResult:
    drug_name: str
    measured_conc_mg_L: List[float]
    measured_times_h: List[float]
    therapeutic_min_mg_L: float
    therapeutic_max_mg_L: float
    n_above: int
    n_below: int
    n_within: int
    pct_within: float
    time_above_min_h: float
    time_below_min_h: float
    recommended_dose_adjustment: str
    dose_adjustment_factor: float
    monitoring_interval_h: float
    alert_level: str


def assess_tdm(
    drug_name: str,
    measured_conc_mg_L: List[float],
    measured_times_h: List[float],
    therapeutic_min_mg_L: float,
    therapeutic_max_mg_L: float,
    target_pct_within: float = 80.0,
) -> TDMResult:
    """Assess therapeutic drug monitoring data and generate dosing recommendations."""
    if therapeutic_min_mg_L >= therapeutic_max_mg_L:
        raise ValueError(
            f"therapeutic_min_mg_L ({therapeutic_min_mg_L}) must be less than "
            f"therapeutic_max_mg_L ({therapeutic_max_mg_L})"
        )
    if len(measured_conc_mg_L) < 2:
        raise ValueError("At least 2 concentration measurements are required.")
    if len(measured_conc_mg_L) != len(measured_times_h):
        raise ValueError(
            "measured_conc_mg_L and measured_times_h must have the same length."
        )

    conc = np.array(measured_conc_mg_L, dtype=float)
    times = np.array(measured_times_h, dtype=float)

    t_min = therapeutic_min_mg_L
    t_max = therapeutic_max_mg_L

    n_above = int(np.sum(conc > t_max))
    n_below = int(np.sum(conc < t_min))
    n_within = int(np.sum((conc >= t_min) & (conc <= t_max)))
    n_total = len(conc)
    pct_within = n_within / n_total * 100.0

    # Trapezoidal integration of indicator functions
    above_min_indicator = (conc > t_min).astype(float)
    below_min_indicator = (conc < t_min).astype(float)
    time_above_min_h = float(np.trapz(above_min_indicator, times))
    time_below_min_h = float(np.trapz(below_min_indicator, times))

    median_conc = float(np.median(conc))

    if median_conc < t_min:
        recommended_dose_adjustment = "increase"
        raw_factor = (t_min / median_conc) * 0.8
        dose_adjustment_factor = float(min(2.0, raw_factor))
    elif median_conc > t_max:
        recommended_dose_adjustment = "decrease"
        raw_factor = (t_max / median_conc) * 1.2
        dose_adjustment_factor = float(max(0.5, raw_factor))
    else:
        recommended_dose_adjustment = "maintain"
        dose_adjustment_factor = 1.0

    # Alert level
    any_critically_high = bool(np.any(conc > 3.0 * t_max))
    if pct_within < 50.0 or any_critically_high:
        alert_level = "critical"
    elif pct_within < target_pct_within:
        alert_level = "caution"
    else:
        alert_level = "normal"

    monitoring_interval_h = 24.0 if recommended_dose_adjustment == "maintain" else 12.0

    return TDMResult(
        drug_name=drug_name,
        measured_conc_mg_L=list(measured_conc_mg_L),
        measured_times_h=list(measured_times_h),
        therapeutic_min_mg_L=therapeutic_min_mg_L,
        therapeutic_max_mg_L=therapeutic_max_mg_L,
        n_above=n_above,
        n_below=n_below,
        n_within=n_within,
        pct_within=pct_within,
        time_above_min_h=time_above_min_h,
        time_below_min_h=time_below_min_h,
        recommended_dose_adjustment=recommended_dose_adjustment,
        dose_adjustment_factor=dose_adjustment_factor,
        monitoring_interval_h=monitoring_interval_h,
        alert_level=alert_level,
    )


def tdm_dashboard(
    drug_name: str,
    measured_conc_mg_L: List[float],
    measured_times_h: List[float],
    therapeutic_min_mg_L: float,
    therapeutic_max_mg_L: float,
) -> dict:
    """Generate a TDM dashboard summary with assessment, statistics, and recommendations."""
    result = assess_tdm(
        drug_name=drug_name,
        measured_conc_mg_L=measured_conc_mg_L,
        measured_times_h=measured_times_h,
        therapeutic_min_mg_L=therapeutic_min_mg_L,
        therapeutic_max_mg_L=therapeutic_max_mg_L,
    )

    conc = np.array(measured_conc_mg_L, dtype=float)
    mean_conc = float(np.mean(conc))
    median_conc = float(np.median(conc))
    std_conc = float(np.std(conc, ddof=1)) if len(conc) > 1 else 0.0
    cv_pct = (std_conc / mean_conc * 100.0) if mean_conc != 0.0 else 0.0

    statistics = {
        "mean": mean_conc,
        "median": median_conc,
        "CV_pct": cv_pct,
        "min_observed": float(np.min(conc)),
        "max_observed": float(np.max(conc)),
    }

    recommendations: List[str] = []
    adjustment = result.recommended_dose_adjustment
    factor = result.dose_adjustment_factor

    if adjustment == "increase":
        pct_change = (factor - 1.0) * 100.0
        recommendations.append(
            f"Increase dose by {pct_change:.0f}% (adjustment factor: {factor:.2f})."
        )
    elif adjustment == "decrease":
        pct_change = (1.0 - factor) * 100.0
        recommendations.append(
            f"Decrease dose by {pct_change:.0f}% (adjustment factor: {factor:.2f})."
        )
    else:
        recommendations.append("Continue current regimen.")

    if result.alert_level == "critical":
        recommendations.append(
            "CRITICAL: Concentrations are dangerously out of range. Immediate clinical review required."
        )
    elif result.alert_level == "caution":
        recommendations.append(
            f"CAUTION: Only {result.pct_within:.1f}% of concentrations are within the therapeutic range."
        )

    recommendations.append(
        f"Next monitoring recommended in {result.monitoring_interval_h:.0f} hours."
    )

    assessment_dict = {
        "drug_name": result.drug_name,
        "measured_conc_mg_L": result.measured_conc_mg_L,
        "measured_times_h": result.measured_times_h,
        "therapeutic_min_mg_L": result.therapeutic_min_mg_L,
        "therapeutic_max_mg_L": result.therapeutic_max_mg_L,
        "n_above": result.n_above,
        "n_below": result.n_below,
        "n_within": result.n_within,
        "pct_within": result.pct_within,
        "time_above_min_h": result.time_above_min_h,
        "time_below_min_h": result.time_below_min_h,
        "recommended_dose_adjustment": result.recommended_dose_adjustment,
        "dose_adjustment_factor": result.dose_adjustment_factor,
        "monitoring_interval_h": result.monitoring_interval_h,
        "alert_level": result.alert_level,
    }

    return {
        "assessment": assessment_dict,
        "statistics": statistics,
        "recommendations": recommendations,
    }


def population_tdm_summary(patient_results: List[TDMResult]) -> dict:
    """Summarize TDM results across a patient population."""
    n_patients = len(patient_results)

    if n_patients == 0:
        return {
            "n_patients": 0,
            "pct_patients_within_target": 0.0,
            "n_critical": 0,
            "n_caution": 0,
            "n_normal": 0,
            "median_pct_within": 0.0,
            "dose_adjustment_counts": {"increase": 0, "decrease": 0, "maintain": 0},
        }

    n_critical = sum(1 for r in patient_results if r.alert_level == "critical")
    n_caution = sum(1 for r in patient_results if r.alert_level == "caution")
    n_normal = sum(1 for r in patient_results if r.alert_level == "normal")

    # Patients within target: those whose pct_within >= their implied target (use 80% as default)
    n_within_target = sum(1 for r in patient_results if r.alert_level == "normal")
    pct_patients_within_target = n_within_target / n_patients * 100.0

    pct_within_values = np.array([r.pct_within for r in patient_results], dtype=float)
    median_pct_within = float(np.median(pct_within_values))

    dose_adjustment_counts = {
        "increase": sum(1 for r in patient_results if r.recommended_dose_adjustment == "increase"),
        "decrease": sum(1 for r in patient_results if r.recommended_dose_adjustment == "decrease"),
        "maintain": sum(1 for r in patient_results if r.recommended_dose_adjustment == "maintain"),
    }

    return {
        "n_patients": n_patients,
        "pct_patients_within_target": pct_patients_within_target,
        "n_critical": n_critical,
        "n_caution": n_caution,
        "n_normal": n_normal,
        "median_pct_within": median_pct_within,
        "dose_adjustment_counts": dose_adjustment_counts,
    }
