"""Drug level monitoring and therapeutic range assessment — clinical TDM decision support."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DrugLevelAssessment",
    "assess_drug_level",
    "monitor_steady_state",
    "generate_dose_recommendation",
]


@dataclass(frozen=True)
class DrugLevelAssessment:
    """Result from a drug level / TDM assessment."""

    compound_name: str
    measured_conc_mg_L: float
    therapeutic_min: float
    therapeutic_max: float
    status: str
    in_therapeutic_range: bool
    percent_of_min: float
    percent_of_max: float
    dose_adjustment_factor: float
    dose_recommendation: str
    back_calculated_dose_mg: float
    notes: str


def _determine_status(
    measured: float,
    t_min: float,
    t_max: float,
) -> str:
    """Return therapeutic status string."""
    if measured > 2.0 * t_max:
        return "toxic"
    elif measured > t_max:
        return "supratherapeutic"
    elif measured < t_min:
        return "subtherapeutic"
    else:
        return "therapeutic"


def assess_drug_level(
    compound_name: str,
    measured_conc_mg_L: float,
    therapeutic_min_mg_L: float,
    therapeutic_max_mg_L: float,
    sample_time_h: float,
    dose_mg: float,
    t_half_h: float | None = None,
    vd_L: float | None = None,
    last_dose_time_h: float | None = None,
) -> DrugLevelAssessment:
    """Assess whether a measured drug level is within the therapeutic range.

    Parameters
    ----------
    compound_name:
        Name of the drug.
    measured_conc_mg_L:
        Measured plasma concentration (mg/L).
    therapeutic_min_mg_L:
        Lower bound of therapeutic range (mg/L).
    therapeutic_max_mg_L:
        Upper bound of therapeutic range (mg/L).
    sample_time_h:
        Time since last dose when sample was taken (hours).
    dose_mg:
        Administered dose (mg).
    t_half_h:
        Elimination half-life (hours). Optional.
    vd_L:
        Volume of distribution (L). Optional.
    last_dose_time_h:
        Time since last dose (hours). Optional (defaults to sample_time_h).

    Returns
    -------
    DrugLevelAssessment
    """
    if measured_conc_mg_L < 0:
        raise ValueError(f"measured_conc_mg_L must be >= 0, got {measured_conc_mg_L}")
    if therapeutic_min_mg_L <= 0 or therapeutic_max_mg_L <= 0:
        raise ValueError("therapeutic_min and therapeutic_max must be > 0")
    if therapeutic_min_mg_L >= therapeutic_max_mg_L:
        raise ValueError(
            f"therapeutic_min ({therapeutic_min_mg_L}) must be < therapeutic_max ({therapeutic_max_mg_L})"
        )

    t_min = therapeutic_min_mg_L
    t_max = therapeutic_max_mg_L
    t_mid = (t_min + t_max) / 2.0

    in_range = t_min <= measured_conc_mg_L <= t_max
    status = _determine_status(measured_conc_mg_L, t_min, t_max)

    percent_of_min = (measured_conc_mg_L / t_min) * 100.0
    percent_of_max = (measured_conc_mg_L / t_max) * 100.0

    # Dose adjustment factor: target mid-range / measured, clamped [0.5, 2.0]
    if measured_conc_mg_L > 0:
        raw_factor = t_mid / measured_conc_mg_L
    else:
        raw_factor = 2.0  # maximum increase when measured is zero

    dose_adjustment_factor = max(0.5, min(2.0, raw_factor))

    # Dose recommendation
    if status == "therapeutic":
        recommendation = f"Maintain current dose. Level {measured_conc_mg_L:.2f} mg/L is within therapeutic range ({t_min}-{t_max} mg/L)."
    elif status == "subtherapeutic":
        new_dose = dose_mg * dose_adjustment_factor
        recommendation = (
            f"Increase dose. Level {measured_conc_mg_L:.2f} mg/L is below therapeutic minimum ({t_min} mg/L). "
            f"Consider increasing dose by factor {dose_adjustment_factor:.2f} (suggest {new_dose:.0f} mg)."
        )
    elif status == "supratherapeutic":
        new_dose = dose_mg * dose_adjustment_factor
        recommendation = (
            f"Reduce dose. Level {measured_conc_mg_L:.2f} mg/L exceeds therapeutic maximum ({t_max} mg/L). "
            f"Consider reducing dose by factor {dose_adjustment_factor:.2f} (suggest {new_dose:.0f} mg)."
        )
    else:  # toxic
        recommendation = (
            f"HOLD DOSE — TOXIC LEVEL. Measured {measured_conc_mg_L:.2f} mg/L exceeds 2x therapeutic maximum "
            f"({2 * t_max:.2f} mg/L). Seek immediate clinical evaluation."
        )

    # Back-calculate dose using simple 1-compartment model if Vd and t_half provided
    back_calculated_dose_mg = 0.0
    t_elapsed = last_dose_time_h if last_dose_time_h is not None else sample_time_h
    if vd_L is not None and t_half_h is not None and t_half_h > 0 and t_elapsed >= 0:
        ke = math.log(2.0) / t_half_h
        # C = D / Vd * exp(-ke * t)  =>  D = C * Vd / exp(-ke * t) = C * Vd * exp(ke * t)
        back_calculated_dose_mg = measured_conc_mg_L * vd_L * math.exp(ke * t_elapsed)

    # Notes
    note_parts = [f"Status: {status}. Sample taken {sample_time_h:.1f}h post-dose."]
    if t_half_h is not None:
        note_parts.append(f"t½ = {t_half_h:.1f} h.")
    if vd_L is not None:
        note_parts.append(f"Vd = {vd_L:.1f} L.")
    if back_calculated_dose_mg > 0:
        note_parts.append(f"Back-calculated dose estimate: {back_calculated_dose_mg:.1f} mg.")
    notes = " ".join(note_parts)

    return DrugLevelAssessment(
        compound_name=compound_name,
        measured_conc_mg_L=measured_conc_mg_L,
        therapeutic_min=t_min,
        therapeutic_max=t_max,
        status=status,
        in_therapeutic_range=in_range,
        percent_of_min=round(percent_of_min, 4),
        percent_of_max=round(percent_of_max, 4),
        dose_adjustment_factor=round(dose_adjustment_factor, 6),
        dose_recommendation=recommendation,
        back_calculated_dose_mg=round(back_calculated_dose_mg, 4),
        notes=notes,
    )


def monitor_steady_state(
    compound_name: str,
    trough_conc_mg_L: float,
    peak_conc_mg_L: float,
    therapeutic_trough_min: float,
    therapeutic_trough_max: float,
    therapeutic_peak_min: float,
    therapeutic_peak_max: float,
) -> dict:
    """Monitor both trough and peak concentrations at steady state.

    Parameters
    ----------
    compound_name:
        Name of the drug.
    trough_conc_mg_L:
        Measured trough (pre-dose) concentration (mg/L).
    peak_conc_mg_L:
        Measured peak concentration (mg/L).
    therapeutic_trough_min/max:
        Therapeutic range for trough concentrations (mg/L).
    therapeutic_peak_min/max:
        Therapeutic range for peak concentrations (mg/L).

    Returns
    -------
    dict with trough_status, peak_status, overall_status and details.
    """
    trough_status = _determine_status(
        trough_conc_mg_L, therapeutic_trough_min, therapeutic_trough_max
    )
    peak_status = _determine_status(peak_conc_mg_L, therapeutic_peak_min, therapeutic_peak_max)

    trough_in_range = therapeutic_trough_min <= trough_conc_mg_L <= therapeutic_trough_max
    peak_in_range = therapeutic_peak_min <= peak_conc_mg_L <= therapeutic_peak_max

    if trough_in_range and peak_in_range:
        overall_status = "optimal"
    elif not trough_in_range and not peak_in_range:
        overall_status = "adjust_interval"
    elif not trough_in_range:
        overall_status = "adjust_dose"
    else:
        overall_status = "adjust_interval"

    return {
        "compound_name": compound_name,
        "trough_conc_mg_L": trough_conc_mg_L,
        "peak_conc_mg_L": peak_conc_mg_L,
        "trough_status": trough_status,
        "peak_status": peak_status,
        "trough_in_range": trough_in_range,
        "peak_in_range": peak_in_range,
        "overall_status": overall_status,
        "therapeutic_trough_range": (therapeutic_trough_min, therapeutic_trough_max),
        "therapeutic_peak_range": (therapeutic_peak_min, therapeutic_peak_max),
        "recommendation": (
            f"Trough {trough_status}, peak {peak_status}. Overall: {overall_status}."
        ),
    }


def generate_dose_recommendation(assessment: DrugLevelAssessment) -> str:
    """Generate a clinical dose recommendation string from an assessment.

    Parameters
    ----------
    assessment:
        DrugLevelAssessment result from assess_drug_level.

    Returns
    -------
    str
        Human-readable clinical recommendation.
    """
    return assessment.dose_recommendation
