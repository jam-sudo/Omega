"""Therapeutic Drug Monitoring (TDM) interpretation and dose recommendation reports."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TDMTroughResult:
    drug_name: str
    measured_trough_mg_L: float
    lower_target: float
    upper_target: float
    status: str
    confidence_interval: tuple[float, float]
    interpretation: str
    dose_recommendation: str


def interpret_trough(
    drug_name: str,
    measured_trough_mg_L: float,
    therapeutic_range_mg_L: tuple[float, float],
    assay_cv_pct: float = 10.0,
) -> TDMTroughResult:
    """Assess whether a measured trough concentration is within the therapeutic range.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    measured_trough_mg_L:
        Measured trough concentration (mg/L).
    therapeutic_range_mg_L:
        (lower_target, upper_target) in mg/L.
    assay_cv_pct:
        Assay coefficient of variation (%). Default 10%.

    Returns
    -------
    TDMTroughResult
    """
    if len(therapeutic_range_mg_L) != 2:
        raise ValueError("therapeutic_range_mg_L must be a (lower, upper) tuple")
    lower_target, upper_target = therapeutic_range_mg_L
    if lower_target >= upper_target:
        raise ValueError(
            f"lower_target ({lower_target}) must be less than upper_target ({upper_target})"
        )
    if measured_trough_mg_L <= 0:
        raise ValueError(f"measured_trough_mg_L must be positive, got {measured_trough_mg_L}")

    # 95% confidence interval: measured ± 2 * CV * measured
    half_width = 2.0 * (assay_cv_pct / 100.0) * measured_trough_mg_L
    ci_lower = measured_trough_mg_L - half_width
    ci_upper = measured_trough_mg_L + half_width

    # Status classification
    if ci_upper < lower_target:
        status = "subtherapeutic"
        interpretation = (
            f"Trough is below the therapeutic range [{lower_target}–{upper_target} mg/L]. "
            "Plasma levels are insufficient for therapeutic effect."
        )
        dose_recommendation = "Increase dose to achieve target trough concentrations."
    elif ci_lower > upper_target:
        status = "supratherapeutic"
        interpretation = (
            f"Trough exceeds the therapeutic range [{lower_target}–{upper_target} mg/L]. "
            "Risk of toxicity."
        )
        dose_recommendation = (
            "Reduce dose or extend dosing interval to lower trough concentrations."
        )
    else:
        # In range, but check borderline
        if measured_trough_mg_L >= lower_target and measured_trough_mg_L <= upper_target:
            # CI overlaps lower boundary?
            if ci_lower < lower_target:
                status = "borderline_low"
                interpretation = (
                    f"Trough is within range but confidence interval overlaps the lower boundary "
                    f"({lower_target} mg/L). Close monitoring recommended."
                )
                dose_recommendation = "Consider slight dose increase; monitor closely."
            elif ci_upper > upper_target:
                status = "borderline_high"
                interpretation = (
                    f"Trough is within range but confidence interval overlaps the upper boundary "
                    f"({upper_target} mg/L). Monitor for signs of toxicity."
                )
                dose_recommendation = (
                    "Monitor for toxicity; consider slight dose reduction if side effects occur."
                )
            else:
                status = "therapeutic"
                interpretation = (
                    f"Trough is within the therapeutic range [{lower_target}–{upper_target} mg/L]. "
                    "No dose adjustment required."
                )
                dose_recommendation = "Continue current regimen."
        else:
            # Measured is outside range but CI overlaps (borderline cases)
            if measured_trough_mg_L < lower_target:
                status = "borderline_low"
                interpretation = (
                    f"Trough is slightly below range [{lower_target}–{upper_target} mg/L], "
                    "but assay uncertainty allows for borderline therapeutic levels."
                )
                dose_recommendation = "Consider slight dose increase; repeat TDM after next dose."
            else:
                status = "borderline_high"
                interpretation = (
                    f"Trough is slightly above range [{lower_target}–{upper_target} mg/L], "
                    "but assay uncertainty allows for borderline therapeutic levels."
                )
                dose_recommendation = "Monitor for toxicity; repeat TDM after next dose."

    return TDMTroughResult(
        drug_name=drug_name,
        measured_trough_mg_L=measured_trough_mg_L,
        lower_target=lower_target,
        upper_target=upper_target,
        status=status,
        confidence_interval=(ci_lower, ci_upper),
        interpretation=interpretation,
        dose_recommendation=dose_recommendation,
    )


def _round_dose(raw_dose: float) -> float:
    """Round dose to nearest 50 mg (or 25 mg if < 200 mg)."""
    if raw_dose < 200.0:
        return round(raw_dose / 25.0) * 25.0
    return round(raw_dose / 50.0) * 50.0


def bayesian_dose_recommendation(
    measured_trough_mg_L: float,
    current_dose_mg: float,
    target_trough_mg_L: float,
    t_half_h: float,
    interval_h: float,
) -> dict:
    """Recommend a new dose using linear proportionality (MAP-Bayesian simplified).

    Parameters
    ----------
    measured_trough_mg_L:
        Current measured trough (mg/L).
    current_dose_mg:
        Current dose (mg).
    target_trough_mg_L:
        Desired trough target (mg/L).
    t_half_h:
        Drug half-life (h).
    interval_h:
        Dosing interval (h).

    Returns
    -------
    dict with new_dose_mg, dose_change_pct, time_to_new_ss_h, next_trough_check_h, notes.
    """
    if measured_trough_mg_L <= 0:
        raise ValueError(f"measured_trough_mg_L must be positive, got {measured_trough_mg_L}")
    if current_dose_mg <= 0:
        raise ValueError(f"current_dose_mg must be positive, got {current_dose_mg}")
    if target_trough_mg_L <= 0:
        raise ValueError(f"target_trough_mg_L must be positive, got {target_trough_mg_L}")

    raw_new_dose = current_dose_mg * (target_trough_mg_L / measured_trough_mg_L)
    new_dose_mg = _round_dose(raw_new_dose)
    dose_change_pct = (new_dose_mg - current_dose_mg) / current_dose_mg * 100.0
    time_to_new_ss_h = 4.0 * t_half_h
    next_trough_check_h = time_to_new_ss_h + interval_h

    if abs(dose_change_pct) < 5:
        notes = "Minimal dose adjustment needed; current dose is close to target."
    elif dose_change_pct > 0:
        notes = f"Dose increase recommended from {current_dose_mg:.0f} mg to {new_dose_mg:.0f} mg."
    else:
        notes = f"Dose reduction recommended from {current_dose_mg:.0f} mg to {new_dose_mg:.0f} mg."

    return {
        "new_dose_mg": new_dose_mg,
        "dose_change_pct": dose_change_pct,
        "time_to_new_ss_h": time_to_new_ss_h,
        "next_trough_check_h": next_trough_check_h,
        "notes": notes,
    }


def generate_tdm_report(
    drug_name: str,
    patient_id: str,
    measured_trough_mg_L: float,
    current_dose_mg: float,
    interval_h: float,
    therapeutic_range_mg_L: tuple[float, float],
    t_half_h: float,
    target_trough_mg_L: float | None = None,
    assay_cv_pct: float = 10.0,
) -> dict:
    """Generate a complete TDM interpretation and dose recommendation report.

    Parameters
    ----------
    drug_name:
        Drug name.
    patient_id:
        Patient identifier.
    measured_trough_mg_L:
        Measured trough concentration (mg/L).
    current_dose_mg:
        Current dose (mg).
    interval_h:
        Dosing interval (h).
    therapeutic_range_mg_L:
        (lower, upper) therapeutic range (mg/L).
    t_half_h:
        Drug half-life (h).
    target_trough_mg_L:
        Target trough; defaults to midpoint of therapeutic range.
    assay_cv_pct:
        Assay CV (%). Default 10%.

    Returns
    -------
    dict with patient_id, drug_name, interpretation, recommendation, summary_text.
    """
    lower_target, upper_target = therapeutic_range_mg_L
    if target_trough_mg_L is None:
        target_trough_mg_L = (lower_target + upper_target) / 2.0

    interpretation = interpret_trough(
        drug_name=drug_name,
        measured_trough_mg_L=measured_trough_mg_L,
        therapeutic_range_mg_L=therapeutic_range_mg_L,
        assay_cv_pct=assay_cv_pct,
    )

    recommendation = bayesian_dose_recommendation(
        measured_trough_mg_L=measured_trough_mg_L,
        current_dose_mg=current_dose_mg,
        target_trough_mg_L=target_trough_mg_L,
        t_half_h=t_half_h,
        interval_h=interval_h,
    )

    summary_text = (
        f"TDM Report for {drug_name} — Patient {patient_id}: "
        f"Measured trough = {measured_trough_mg_L} mg/L "
        f"(range {lower_target}–{upper_target} mg/L), status = {interpretation.status}. "
        f"{interpretation.dose_recommendation} "
        f"Recommended new dose: {recommendation['new_dose_mg']:.0f} mg every {interval_h:.0f} h. "
        f"Re-check trough in {recommendation['next_trough_check_h']:.0f} h."
    )

    return {
        "patient_id": patient_id,
        "drug_name": drug_name,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "summary_text": summary_text,
    }


def population_trough_percentile(
    measured_trough_mg_L: float,
    population_mean_mg_L: float,
    population_cv_pct: float,
) -> dict:
    """Estimate where a patient's trough falls in the population distribution.

    Assumes log-normal distribution for the population trough.

    Parameters
    ----------
    measured_trough_mg_L:
        Individual measured trough (mg/L).
    population_mean_mg_L:
        Population geometric mean trough (mg/L).
    population_cv_pct:
        Population coefficient of variation (%).

    Returns
    -------
    dict with percentile, z_score, interpretation.
    """
    if measured_trough_mg_L <= 0:
        raise ValueError(f"measured_trough_mg_L must be positive, got {measured_trough_mg_L}")
    if population_mean_mg_L <= 0:
        raise ValueError(f"population_mean_mg_L must be positive, got {population_mean_mg_L}")
    if population_cv_pct <= 0:
        raise ValueError(f"population_cv_pct must be positive, got {population_cv_pct}")

    cv = population_cv_pct / 100.0
    # Log-normal sigma: sigma^2 = ln(1 + CV^2)
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    # Log-normal mean: mu = ln(mean) - 0.5 * sigma^2
    mu = math.log(population_mean_mg_L) - 0.5 * sigma * sigma

    z = (math.log(measured_trough_mg_L) - mu) / sigma
    # Normal CDF via erf: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    percentile = 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if percentile < 25.0:
        interpretation = "below average"
    elif percentile > 75.0:
        interpretation = "above average"
    else:
        interpretation = "average"

    return {
        "percentile": percentile,
        "z_score": z,
        "interpretation": interpretation,
    }
