"""Renal dosing adjustments for patients with chronic kidney disease."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RenalDosingResult",
    "adjust_renal_dose",
    "renal_dose_from_patient",
]


@dataclass
class RenalDosingResult:
    drug_name: str
    crcl_mL_per_min: float
    ckd_stage: str
    dose_adjustment_factor: float
    adjusted_dose_mg: float
    adjusted_interval_h: float
    primary_method: str
    recommendation: str
    requires_dialysis_supplementation: bool


def _ckd_stage(crcl: float) -> str:
    """Classify CKD stage from creatinine clearance (mL/min)."""
    if crcl >= 90:
        return "normal"
    if crcl >= 60:
        return "mild"
    if crcl >= 30:
        return "moderate"
    if crcl >= 15:
        return "severe"
    return "ESRD"


def _cockcroft_gault(
    age_years: float,
    weight_kg: float,
    serum_creatinine_mg_dL: float,
    sex: str = "male",
) -> float:
    """Estimate creatinine clearance (mL/min) using the Cockcroft-Gault equation."""
    crcl = (140.0 - age_years) * weight_kg / (72.0 * serum_creatinine_mg_dL)
    if sex.lower() in ("female", "f"):
        crcl *= 0.85
    return float(max(crcl, 1.0))


def _primary_method(factor: float, original_interval: float, adjusted_interval: float) -> str:
    """Determine primary dose-adjustment method used."""
    dose_reduced = factor < 1.0
    interval_extended = adjusted_interval > original_interval + 1e-9
    if dose_reduced and interval_extended:
        return "both"
    if interval_extended:
        return "interval_extension"
    return "dose_reduction"


def _build_recommendation(
    drug_name: str,
    ckd_stage: str,
    factor: float,
    adjusted_dose: float,
    adjusted_interval: float,
    requires_dialysis_supplementation: bool,
    method: str,
) -> str:
    lines = [
        f"{drug_name}: CKD stage '{ckd_stage}' — dose adjustment factor {factor:.2f}.",
        f"Adjusted dose: {adjusted_dose:.2f} mg every {adjusted_interval:.1f} h "
        f"(method: {method.replace('_', ' ')}).",
    ]
    if requires_dialysis_supplementation:
        lines.append(
            "Patient is on dialysis with high renal elimination — "
            "supplemental dosing after each dialysis session is recommended."
        )
    if ckd_stage == "normal":
        lines.append("No dose adjustment required.")
    return " ".join(lines)


def adjust_renal_dose(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    crcl_mL_per_min: float,
    fraction_renal: float = 0.5,
    target_auc_ratio: float = 1.0,
) -> RenalDosingResult:
    """Compute renally adjusted dose from a known creatinine clearance.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Standard (reference) dose in mg.
    dosing_interval_h:
        Standard dosing interval in hours.
    crcl_mL_per_min:
        Creatinine clearance in mL/min.
    fraction_renal:
        Fraction of total clearance attributable to renal elimination (0–1).
    target_auc_ratio:
        Reserved for future extension; currently unused.

    Returns
    -------
    RenalDosingResult

    Raises
    ------
    ValueError
        If ``dose_mg <= 0`` or ``crcl_mL_per_min < 0``.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if crcl_mL_per_min < 0:
        raise ValueError(f"crcl_mL_per_min must be >= 0, got {crcl_mL_per_min}")

    stage = _ckd_stage(crcl_mL_per_min)

    raw_factor = 1.0 - fraction_renal * (1.0 - crcl_mL_per_min / 100.0)
    factor = float(np.clip(raw_factor, 0.1, 1.0))

    adjusted_dose = dose_mg * factor
    adjusted_interval = float(min(dosing_interval_h / factor, 48.0))

    requires_dialysis_supplementation = stage == "ESRD" and fraction_renal > 0.5

    method = _primary_method(factor, dosing_interval_h, adjusted_interval)
    recommendation = _build_recommendation(
        drug_name,
        stage,
        factor,
        adjusted_dose,
        adjusted_interval,
        requires_dialysis_supplementation,
        method,
    )

    return RenalDosingResult(
        drug_name=drug_name,
        crcl_mL_per_min=float(crcl_mL_per_min),
        ckd_stage=stage,
        dose_adjustment_factor=factor,
        adjusted_dose_mg=adjusted_dose,
        adjusted_interval_h=adjusted_interval,
        primary_method=method,
        recommendation=recommendation,
        requires_dialysis_supplementation=requires_dialysis_supplementation,
    )


def renal_dose_from_patient(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    age_years: float,
    weight_kg: float,
    serum_creatinine_mg_dL: float,
    sex: str = "male",
    fraction_renal: float = 0.5,
) -> RenalDosingResult:
    """Compute renally adjusted dose from patient demographics using Cockcroft-Gault.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Standard (reference) dose in mg.
    dosing_interval_h:
        Standard dosing interval in hours.
    age_years:
        Patient age in years.
    weight_kg:
        Patient weight in kilograms.
    serum_creatinine_mg_dL:
        Serum creatinine in mg/dL.
    sex:
        'male' or 'female'.
    fraction_renal:
        Fraction of total clearance attributable to renal elimination (0–1).

    Returns
    -------
    RenalDosingResult
    """
    crcl = _cockcroft_gault(age_years, weight_kg, serum_creatinine_mg_dL, sex)
    return adjust_renal_dose(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        crcl_mL_per_min=crcl,
        fraction_renal=fraction_renal,
    )
