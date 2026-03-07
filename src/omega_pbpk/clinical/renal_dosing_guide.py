"""Renal impairment dose adjustment guide.

FDA and EMA framework for drug dosing based on eGFR/CrCl using the
Dettli equation for dose adjustment in renal impairment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenalDoseResult:
    """Result of a renal dose adjustment calculation."""

    drug_name: str
    egfr_mL_per_min: float
    ckd_stage: str
    renal_function_category: str
    baseline_dose_mg: float
    recommended_dose_mg: float
    dose_interval_h: float
    dose_adjustment_factor: float
    fe_renal: float
    cl_renal_mL_per_min: float
    cl_hepatic_mL_per_min: float
    contraindicated: bool
    monitoring_required: bool
    notes: str


def ckd_stage_from_egfr(egfr: float) -> str:
    """CKD staging per KDIGO guidelines.

    G1  >= 90
    G2   60-89
    G3a  45-59
    G3b  30-44
    G4   15-29
    G5  < 15

    Parameters
    ----------
    egfr:
        Estimated GFR in mL/min/1.73 m².

    Returns
    -------
    str
        CKD stage label.
    """
    if egfr >= 90:
        return "G1"
    elif egfr >= 60:
        return "G2"
    elif egfr >= 45:
        return "G3a"
    elif egfr >= 30:
        return "G3b"
    elif egfr >= 15:
        return "G4"
    else:
        return "G5"


def renal_function_category(egfr: float) -> str:
    """FDA renal function categories based on eGFR.

    normal  >= 90
    mild     60-89
    moderate 30-59
    severe   15-29
    esrd    < 15

    Parameters
    ----------
    egfr:
        Estimated GFR in mL/min.

    Returns
    -------
    str
        Renal function category.
    """
    if egfr >= 90:
        return "normal"
    elif egfr >= 60:
        return "mild"
    elif egfr >= 30:
        return "moderate"
    elif egfr >= 15:
        return "severe"
    else:
        return "esrd"


def calculate_renal_dose(
    drug_name: str,
    baseline_dose_mg: float,
    baseline_interval_h: float,
    fe_renal: float,
    egfr_mL_per_min: float,
    normal_egfr_mL_per_min: float = 100.0,
    contraindicated_below_egfr: float = 0.0,
    dose_rounding_mg: float = 0.0,
) -> RenalDoseResult:
    """Calculate renally-adjusted dose using the Dettli equation.

    The Dettli equation adjusts dose based on the fraction of drug eliminated
    renally (fe_renal) and the patient's renal function relative to normal.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Standard dose in patients with normal renal function (mg).
    baseline_interval_h:
        Standard dosing interval (h).
    fe_renal:
        Fraction of drug eliminated renally (0 to 1).
    egfr_mL_per_min:
        Patient's eGFR or CrCl (mL/min). Must be >= 0.
    normal_egfr_mL_per_min:
        Normal eGFR reference value (default 100 mL/min).
    contraindicated_below_egfr:
        eGFR threshold below which drug is contraindicated (0 = never).
    dose_rounding_mg:
        Round recommended dose to nearest X mg (0 = no rounding).

    Returns
    -------
    RenalDoseResult
    """
    # Validation
    if baseline_dose_mg <= 0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")
    if baseline_interval_h <= 0:
        raise ValueError(f"baseline_interval_h must be > 0, got {baseline_interval_h}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")
    if egfr_mL_per_min < 0:
        raise ValueError(f"egfr_mL_per_min must be >= 0, got {egfr_mL_per_min}")
    if normal_egfr_mL_per_min <= 0:
        raise ValueError(f"normal_egfr_mL_per_min must be > 0, got {normal_egfr_mL_per_min}")

    # Dettli equation
    rf = min(1.0, egfr_mL_per_min / normal_egfr_mL_per_min)  # renal fraction, capped at 1
    q0 = 1.0 - fe_renal  # non-renal (hepatic) fraction
    dose_factor = q0 + fe_renal * rf

    recommended_dose = baseline_dose_mg * dose_factor

    # Dose rounding
    if dose_rounding_mg > 0:
        recommended_dose = round(recommended_dose / dose_rounding_mg) * dose_rounding_mg

    # Clearance estimates
    cl_renal = egfr_mL_per_min * fe_renal
    # Slight hepatic reduction in severe impairment (hepatorenal considerations)
    cl_hepatic = normal_egfr_mL_per_min * (1.0 - fe_renal) * (1.0 - 0.2 * (1.0 - rf))

    # CKD stage and category
    stage = ckd_stage_from_egfr(egfr_mL_per_min)
    category = renal_function_category(egfr_mL_per_min)

    # Contraindicated flag
    contraindicated = (
        contraindicated_below_egfr > 0 and egfr_mL_per_min < contraindicated_below_egfr
    )

    # Monitoring required if stage G3a or worse (egfr < 60), or high renal elimination
    monitoring_required = egfr_mL_per_min < 60 or fe_renal > 0.5

    notes_parts = [
        f"CKD {stage} ({category}), eGFR={egfr_mL_per_min:.1f} mL/min.",
        f"Dettli dose factor={dose_factor:.3f} (Q0={q0:.2f}, rf={rf:.3f}).",
    ]
    if contraindicated:
        notes_parts.append("CONTRAINDICATED at this eGFR level.")
    if monitoring_required:
        notes_parts.append("Therapeutic drug monitoring recommended.")
    if dose_rounding_mg > 0:
        notes_parts.append(f"Dose rounded to nearest {dose_rounding_mg} mg.")

    return RenalDoseResult(
        drug_name=drug_name,
        egfr_mL_per_min=egfr_mL_per_min,
        ckd_stage=stage,
        renal_function_category=category,
        baseline_dose_mg=baseline_dose_mg,
        recommended_dose_mg=recommended_dose,
        dose_interval_h=baseline_interval_h,
        dose_adjustment_factor=dose_factor,
        fe_renal=fe_renal,
        cl_renal_mL_per_min=cl_renal,
        cl_hepatic_mL_per_min=cl_hepatic,
        contraindicated=contraindicated,
        monitoring_required=monitoring_required,
        notes=" ".join(notes_parts),
    )


def screen_renal_impairment(
    drug_name: str,
    baseline_dose_mg: float,
    baseline_interval_h: float,
    fe_renal: float,
    egfr_values: list,
) -> list:
    """Return list of RenalDoseResult for each eGFR in egfr_values.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Standard dose in patients with normal renal function (mg).
    baseline_interval_h:
        Standard dosing interval (h).
    fe_renal:
        Fraction of drug eliminated renally (0 to 1).
    egfr_values:
        List of eGFR values (mL/min) to evaluate.

    Returns
    -------
    list of RenalDoseResult, one per eGFR value
    """
    return [
        calculate_renal_dose(
            drug_name=drug_name,
            baseline_dose_mg=baseline_dose_mg,
            baseline_interval_h=baseline_interval_h,
            fe_renal=fe_renal,
            egfr_mL_per_min=float(egfr),
        )
        for egfr in egfr_values
    ]
