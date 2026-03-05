"""Dose individualization — covariates-based dose adjustment for renal/hepatic impairment."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PatientCovariates:
    """Patient-specific covariates for dose individualization."""
    body_weight_kg: float = 70.0
    age_years: float = 40.0
    sex: str = "male"              # "male" or "female"
    serum_creatinine_mg_dL: float = 1.0
    egfr_mL_min: float | None = None  # if None, estimated from CrCl
    child_pugh_score: int = 5         # 5-6=A, 7-9=B, 10-15=C
    bilirubin_mg_dL: float = 1.0
    albumin_g_dL: float = 4.0
    alt_U_L: float = 30.0             # alanine aminotransferase


@dataclass(frozen=True)
class IndividualizedDoseResult:
    """Result of dose individualization."""
    patient_id: str
    standard_dose_mg: float
    individualized_dose_mg: float
    dose_interval_h: float
    dose_adjustment_factor: float    # individualized / standard
    renal_adjustment: float          # renal component of adjustment
    hepatic_adjustment: float        # hepatic component of adjustment
    weight_adjustment: float         # allometric/weight component
    egfr_mL_min: float
    crcl_mL_min: float
    renal_impairment_grade: str      # "normal", "mild", "moderate", "severe", "ESRD"
    hepatic_impairment_grade: str    # "normal", "mild" (CP-A), "moderate" (CP-B), "severe" (CP-C)
    predicted_auc_ratio: float       # expected AUC relative to standard patient
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cockcroft_gault(weight_kg: float, age_years: float, sex: str,
                     scr_mg_dL: float) -> float:
    """Cockcroft-Gault equation for CrCl (mL/min)."""
    crcl = (140 - age_years) * weight_kg / (72 * scr_mg_dL)
    if sex.lower() in ("female", "f"):
        crcl *= 0.85
    return max(crcl, 5.0)


def _egfr_ckd_epi(age_years: float, sex: str, scr_mg_dL: float) -> float:
    """CKD-EPI 2021 eGFR equation (mL/min/1.73m²)."""
    kappa = 0.7 if sex.lower() in ("female", "f") else 0.9
    alpha = -0.241 if sex.lower() in ("female", "f") else -0.302
    sex_factor = 1.012 if sex.lower() in ("female", "f") else 1.0
    scr_kappa = scr_mg_dL / kappa
    egfr = 142 * min(scr_kappa, 1.0) ** alpha * max(scr_kappa, 1.0) ** (-1.200) \
           * 0.9938 ** age_years * sex_factor
    return max(egfr, 5.0)


def _renal_impairment_grade(egfr: float) -> str:
    if egfr >= 90:
        return "normal"
    elif egfr >= 60:
        return "mild"
    elif egfr >= 30:
        return "moderate"
    elif egfr >= 15:
        return "severe"
    else:
        return "ESRD"


def _child_pugh_grade(score: int) -> str:
    if score <= 6:
        return "normal"
    elif score <= 9:
        return "mild"    # Child-Pugh A
    elif score <= 12:
        return "moderate"  # Child-Pugh B
    else:
        return "severe"   # Child-Pugh C


def _renal_dose_adjustment(
    egfr: float,
    fraction_renal: float,
    method: str = "fraction_unchanged",
) -> float:
    """Compute renal dose adjustment factor.

    Parameters
    ----------
    egfr : float
        eGFR (mL/min).
    fraction_renal : float
        Fraction of drug eliminated renally (0–1).
    method : str
        "fraction_unchanged": based on fe * (GFR/normal_GFR) scaling
        "proportional": dose ∝ eGFR/90
    """
    normal_gfr = 90.0
    gfr_ratio = min(egfr / normal_gfr, 1.0)

    if method == "proportional":
        return gfr_ratio
    else:
        # Adjustment factor = 1 / (1 - fe + fe * GFR_ratio)
        # Maintains same AUC target
        denom = (1.0 - fraction_renal) + fraction_renal * gfr_ratio
        return max(denom, 0.05)


def _hepatic_dose_adjustment(cp_score: int, fraction_hepatic: float) -> float:
    """Compute hepatic dose adjustment factor based on Child-Pugh score."""
    # Reduction in hepatic CL by CP grade (FDA guidance approximations)
    cl_reduction = {
        "normal": 1.0,
        "mild": 0.75,     # CP-A: ~25% CL reduction
        "moderate": 0.50,  # CP-B: ~50% CL reduction
        "severe": 0.25,    # CP-C: ~75% CL reduction
    }
    grade = _child_pugh_grade(cp_score)
    hepatic_cl_factor = cl_reduction[grade]

    # Combined CL = hepatic_fraction * CL_hepatic + (1-hepatic_fraction) * CL_renal
    # Dose adjustment to maintain same exposure
    combined_cl_ratio = fraction_hepatic * hepatic_cl_factor + (1.0 - fraction_hepatic)
    return combined_cl_ratio


def _weight_adjustment(weight_kg: float, reference_kg: float = 70.0,
                        exponent: float = 0.75) -> float:
    """Allometric weight-based dose adjustment."""
    return (weight_kg / reference_kg) ** exponent


def individualize_dose(
    patient_id: str,
    patient: PatientCovariates,
    standard_dose_mg: float,
    dose_interval_h: float = 24.0,
    fraction_renal: float = 0.5,
    fraction_hepatic: float = 0.5,
    use_weight_adjustment: bool = False,
    renal_method: str = "fraction_unchanged",
    reference_weight_kg: float = 70.0,
) -> IndividualizedDoseResult:
    """Compute individualized dose based on patient covariates.

    Parameters
    ----------
    standard_dose_mg : float
        Reference dose for a standard 70 kg adult with normal organ function.
    fraction_renal : float
        Fraction of drug eliminated via renal route.
    fraction_hepatic : float
        Fraction of drug eliminated via hepatic metabolism.
    use_weight_adjustment : bool
        Apply allometric weight scaling.
    renal_method : str
        Renal adjustment method: "fraction_unchanged" or "proportional".
    """
    warnings: list[str] = []
    notes: list[str] = []

    # Compute renal function
    crcl = _cockcroft_gault(patient.body_weight_kg, patient.age_years,
                             patient.sex, patient.serum_creatinine_mg_dL)
    if patient.egfr_mL_min is not None:
        egfr = patient.egfr_mL_min
    else:
        egfr = _egfr_ckd_epi(patient.age_years, patient.sex,
                               patient.serum_creatinine_mg_dL)

    renal_grade = _renal_impairment_grade(egfr)
    hepatic_grade = _child_pugh_grade(patient.child_pugh_score)

    # Renal adjustment
    renal_adj = _renal_dose_adjustment(egfr, fraction_renal, renal_method)

    # Hepatic adjustment
    hepatic_adj = _hepatic_dose_adjustment(patient.child_pugh_score, fraction_hepatic)

    # Weight adjustment
    if use_weight_adjustment:
        weight_adj = _weight_adjustment(patient.body_weight_kg, reference_weight_kg)
    else:
        weight_adj = 1.0

    # Combined adjustment
    # CL ∝ renal_adj * hepatic_adj → dose adjustment = renal_adj * hepatic_adj
    combined_adj = renal_adj * hepatic_adj * weight_adj

    # Predicted AUC ratio (inverse of CL ratio)
    predicted_auc_ratio = 1.0 / combined_adj if combined_adj > 0 else math.nan

    individualized_dose = standard_dose_mg * combined_adj

    # Cap adjustment
    if combined_adj > 2.0:
        warnings.append("Dose adjustment >2x standard — verify clinical appropriateness")
    if combined_adj < 0.1:
        warnings.append("Dose adjustment <10% of standard — consider alternative drug")
        individualized_dose = standard_dose_mg * 0.1

    # Clinical notes
    if renal_grade != "normal":
        notes.append(f"Renal impairment ({renal_grade}): eGFR={egfr:.0f} mL/min, CrCl={crcl:.0f} mL/min")
    if hepatic_grade not in ("normal", "mild"):
        notes.append(f"Hepatic impairment (Child-Pugh {hepatic_grade}): score={patient.child_pugh_score}")
    if patient.age_years >= 65:
        warnings.append("Elderly patient (≥65y): monitor closely; consider conservative starting dose")
    if patient.age_years < 18:
        warnings.append("Pediatric patient: weight-based dosing recommended")

    # Round to practical dose (nearest 5 mg)
    individualized_dose_rounded = round(individualized_dose / 5) * 5
    if abs(individualized_dose_rounded - individualized_dose) > 0.1:
        notes.append(f"Rounded from {individualized_dose:.1f} mg to {individualized_dose_rounded} mg")
    individualized_dose = max(individualized_dose_rounded, 5.0)

    return IndividualizedDoseResult(
        patient_id=patient_id,
        standard_dose_mg=standard_dose_mg,
        individualized_dose_mg=individualized_dose,
        dose_interval_h=dose_interval_h,
        dose_adjustment_factor=individualized_dose / standard_dose_mg,
        renal_adjustment=renal_adj,
        hepatic_adjustment=hepatic_adj,
        weight_adjustment=weight_adj,
        egfr_mL_min=egfr,
        crcl_mL_min=crcl,
        renal_impairment_grade=renal_grade,
        hepatic_impairment_grade=hepatic_grade,
        predicted_auc_ratio=predicted_auc_ratio,
        warnings=warnings,
        notes=notes,
    )


def batch_individualize(
    patients: list[tuple[str, PatientCovariates]],
    standard_dose_mg: float,
    dose_interval_h: float = 24.0,
    fraction_renal: float = 0.5,
    fraction_hepatic: float = 0.5,
) -> list[IndividualizedDoseResult]:
    """Individualize dose for a cohort of patients."""
    return [
        individualize_dose(
            patient_id=pid,
            patient=pat,
            standard_dose_mg=standard_dose_mg,
            dose_interval_h=dose_interval_h,
            fraction_renal=fraction_renal,
            fraction_hepatic=fraction_hepatic,
        )
        for pid, pat in patients
    ]


__all__ = [
    "PatientCovariates",
    "IndividualizedDoseResult",
    "individualize_dose",
    "batch_individualize",
]
