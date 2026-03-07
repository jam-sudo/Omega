"""Renal dosing adjustments for patients with chronic kidney disease.

Phase 407 additions: ``adjust_dose_renal``, ``cockroft_gault``,
``PhaseRenalDosingResult``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RenalDosingResult",
    "adjust_renal_dose",
    "renal_dose_from_patient",
    # Phase 407 additions
    "PhaseRenalDosingResult",
    "adjust_dose_renal",
    "cockroft_gault",
]

# ---------------------------------------------------------------------------
# GFR reference (Phase 407)
# ---------------------------------------------------------------------------

_GFR_NORMAL_REFERENCE = 90.0  # mL/min
_MIN_DOSE_FACTOR = 0.1
_MAX_DOSE_FACTOR = 1.0
_KE_NORMAL_DEFAULT = 0.1  # h⁻¹

_GFR_CATEGORIES: list[tuple[float, float, str]] = [
    (90.0, float("inf"), "normal"),
    (60.0, 90.0, "mild"),
    (30.0, 60.0, "moderate"),
    (15.0, 30.0, "severe"),
    (0.0, 15.0, "ESRD"),
]


# ---------------------------------------------------------------------------
# Original result dataclass (kept for backward compat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 407 result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PhaseRenalDosingResult:
    """Result of GFR-based renal dose adjustment (Phase 407)."""

    drug_name: str
    normal_dose_mg: float
    normal_interval_h: float
    gfr_mL_per_min: float
    f_renal_excretion: float
    gfr_category: str
    dose_factor: float
    adjusted_dose_mg: float
    adjusted_interval_h: float
    method: str
    t_half_adjusted_h: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers — original
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Original public functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 407 helpers
# ---------------------------------------------------------------------------


def _gfr_category(gfr: float) -> str:
    """Return GFR category string."""
    for lo, hi, label in _GFR_CATEGORIES:
        if lo <= gfr < hi:
            return label
    return "ESRD"


def _compute_dose_factor(gfr_mL_per_min: float, f_renal_excretion: float) -> float:
    """Compute dose adjustment factor (clamped to [0.1, 1.0])."""
    raw = 1.0 - f_renal_excretion * (1.0 - gfr_mL_per_min / _GFR_NORMAL_REFERENCE)
    return max(_MIN_DOSE_FACTOR, min(_MAX_DOSE_FACTOR, raw))


# ---------------------------------------------------------------------------
# Phase 407 public functions
# ---------------------------------------------------------------------------


def cockroft_gault(
    scr_mg_dL: float,
    age: float,
    weight_kg: float,
    sex: str,
) -> float:
    """Estimate creatinine clearance (mL/min) via Cockcroft-Gault equation.

    Parameters
    ----------
    scr_mg_dL:
        Serum creatinine in mg/dL.
    age:
        Patient age in years.
    weight_kg:
        Patient body weight in kg.
    sex:
        "male" or "female".

    Returns
    -------
    float
        Estimated CrCl in mL/min.
    """
    if scr_mg_dL <= 0:
        raise ValueError(f"scr_mg_dL must be positive, got {scr_mg_dL}")
    if age <= 0:
        raise ValueError(f"age must be positive, got {age}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be positive, got {weight_kg}")
    sex_lower = sex.lower()
    if sex_lower not in ("male", "female", "m", "f"):
        raise ValueError(f"sex must be 'male' or 'female', got '{sex}'")

    crcl = (140.0 - age) * weight_kg / (72.0 * scr_mg_dL)
    if sex_lower in ("female", "f"):
        crcl *= 0.85
    return crcl


def adjust_dose_renal(
    drug_name: str,
    normal_dose_mg: float,
    normal_interval_h: float,
    gfr_mL_per_min: float,
    f_renal_excretion: float,
    method: str = "dose_reduction",
) -> PhaseRenalDosingResult:
    """Adjust dose for renal impairment using GFR-based methods.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    normal_dose_mg:
        Normal (full renal function) dose in mg.
    normal_interval_h:
        Normal dosing interval in hours.
    gfr_mL_per_min:
        Patient's GFR in mL/min. Values ≥90 indicate normal renal function.
    f_renal_excretion:
        Fraction of drug eliminated renally (0–1).
    method:
        Adjustment method: "dose_reduction", "interval_extension", or "both".

    Returns
    -------
    PhaseRenalDosingResult
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if normal_dose_mg <= 0:
        raise ValueError(f"normal_dose_mg must be positive, got {normal_dose_mg}")
    if normal_interval_h <= 0:
        raise ValueError(f"normal_interval_h must be positive, got {normal_interval_h}")
    if gfr_mL_per_min < 0:
        raise ValueError(f"gfr_mL_per_min must be >=0, got {gfr_mL_per_min}")
    if not (0.0 <= f_renal_excretion <= 1.0):
        raise ValueError(f"f_renal_excretion must be in [0, 1], got {f_renal_excretion}")
    valid_methods = ("dose_reduction", "interval_extension", "both")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    category = _gfr_category(gfr_mL_per_min)
    dose_factor = _compute_dose_factor(gfr_mL_per_min, f_renal_excretion)

    notes: list[str] = []

    if method == "dose_reduction":
        adjusted_dose_mg = normal_dose_mg * dose_factor
        adjusted_interval_h = normal_interval_h
        notes.append(
            f"Dose reduced by factor {dose_factor:.3f}; interval unchanged at "
            f"{normal_interval_h} h."
        )
    elif method == "interval_extension":
        adjusted_dose_mg = normal_dose_mg
        adjusted_interval_h = normal_interval_h / dose_factor
        notes.append(
            f"Interval extended to {adjusted_interval_h:.1f} h; dose unchanged at "
            f"{normal_dose_mg} mg."
        )
    else:  # "both"
        sqrt_factor = math.sqrt(dose_factor)
        adjusted_dose_mg = normal_dose_mg * sqrt_factor
        adjusted_interval_h = normal_interval_h / sqrt_factor
        notes.append(
            f"Dose reduced by sqrt(dose_factor)={sqrt_factor:.3f} and interval "
            f"extended to {adjusted_interval_h:.1f} h."
        )

    # Estimated t_half: t_half = 0.693 / (ke_normal / dose_factor)
    ke_adjusted = _KE_NORMAL_DEFAULT / dose_factor
    t_half_adjusted_h = math.log(2) / ke_adjusted

    if category == "ESRD":
        notes.append("ESRD: consider dialysis clearance and supplemental dosing.")
    if f_renal_excretion >= 0.8:
        notes.append("High renal excretion fraction (>=80%) — renal impairment has major impact.")
    if dose_factor <= _MIN_DOSE_FACTOR:
        notes.append("Dose factor clamped to minimum (0.1). Use with caution.")

    return PhaseRenalDosingResult(
        drug_name=drug_name,
        normal_dose_mg=normal_dose_mg,
        normal_interval_h=normal_interval_h,
        gfr_mL_per_min=gfr_mL_per_min,
        f_renal_excretion=f_renal_excretion,
        gfr_category=category,
        dose_factor=dose_factor,
        adjusted_dose_mg=adjusted_dose_mg,
        adjusted_interval_h=adjusted_interval_h,
        method=method,
        t_half_adjusted_h=t_half_adjusted_h,
        notes=notes,
    )
