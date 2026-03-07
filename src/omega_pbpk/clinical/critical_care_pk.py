"""Phase 617 — Critical care PK modeling.

PK modeling in critically ill patients: altered physiology, organ dysfunction,
augmented renal clearance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticalCarePKResult:
    drug_name: str
    patient_weight_kg: float
    condition: str  # "sepsis","ards","hepatic_failure","icu_general"
    gfr_mL_per_min: float
    albumin_g_dL: float
    cl_scaling_factor: float  # CL relative to healthy
    vd_scaling_factor: float  # Vd relative to healthy
    adjusted_cl_L_per_h: float
    adjusted_vd_L: float
    adjusted_t_half_h: float
    augmented_renal_clearance: bool  # ARC: GFR > 130
    recommended_dose_mg: float
    recommended_interval_h: float
    loading_dose_mg: float
    monitoring_recommendation: str
    notes: str


_VALID_CONDITIONS = {"sepsis", "ards", "hepatic_failure", "icu_general"}


def _sepsis_cl_factor(gfr_mL_per_min: float) -> float:
    """Compute CL factor for sepsis based on GFR."""
    if gfr_mL_per_min > 130:
        return 1.4
    elif gfr_mL_per_min < 30:
        return 0.4
    else:
        return 0.8 + 0.6 * (gfr_mL_per_min - 30) / 100


def adjust_for_critical_illness(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
    baseline_interval_h: float,
    condition: str,  # "sepsis","ards","hepatic_failure","icu_general"
    patient_weight_kg: float = 70.0,
    gfr_mL_per_min: float = 90.0,
    albumin_g_dL: float = 4.0,
    fe_renal: float = 0.5,  # renal fraction of elimination
) -> CriticalCarePKResult:
    """Adjust PK parameters for critical illness conditions."""
    # Validation
    if condition not in _VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {_VALID_CONDITIONS}, got '{condition}'")
    if patient_weight_kg <= 0:
        raise ValueError(f"patient_weight_kg must be > 0, got {patient_weight_kg}")
    if gfr_mL_per_min < 0:
        raise ValueError(f"gfr_mL_per_min must be >= 0, got {gfr_mL_per_min}")
    if albumin_g_dL <= 0:
        raise ValueError(f"albumin_g_dL must be > 0, got {albumin_g_dL}")
    if not (0.0 <= fe_renal <= 1.0):
        raise ValueError(f"fe_renal must be in [0, 1], got {fe_renal}")

    arc = gfr_mL_per_min > 130

    # Determine base scaling factors by condition
    if condition == "sepsis":
        vd_factor = 1.4 + 0.3 * (1.0 if patient_weight_kg > 100 else 0.0)
        cl_factor = _sepsis_cl_factor(gfr_mL_per_min)
    elif condition == "ards":
        vd_factor = 1.25
        cl_factor = 0.8
    elif condition == "hepatic_failure":
        vd_factor = 1.4
        cl_factor = 0.4
    else:  # icu_general
        vd_factor = 1.2
        cl_factor = 0.9

    # ARC boost
    if arc and fe_renal > 0.3:
        cl_factor *= 1.3

    # Albumin effect on CL (low albumin → higher fu → higher CL)
    if albumin_g_dL < 3.5:
        fu_correction = 4.0 / max(albumin_g_dL, 0.5)
        cl_factor *= min(fu_correction, 2.0)

    # Derived quantities
    adjusted_cl = baseline_cl_L_per_h * cl_factor
    adjusted_vd = baseline_vd_L * vd_factor
    adjusted_t_half = math.log(2) * adjusted_vd / adjusted_cl

    recommended_dose = baseline_dose_mg * cl_factor
    loading_dose = baseline_dose_mg * vd_factor
    recommended_interval = baseline_interval_h  # keep interval, adjust dose

    # Monitoring recommendation
    if condition == "sepsis":
        monitoring = "Frequent TDM (every 24-48 h); monitor renal/hepatic function daily"
    elif condition == "ards":
        monitoring = "TDM every 48 h; assess fluid balance and hepatic function"
    elif condition == "hepatic_failure":
        monitoring = "TDM every 24 h; Child-Pugh assessment; watch for accumulation"
    else:
        monitoring = "TDM every 48-72 h; routine organ function monitoring"

    if arc:
        monitoring += "; ARC detected — verify drug levels to avoid sub-therapeutic dosing"

    # Notes
    notes_parts = [
        f"Condition: {condition}; GFR={gfr_mL_per_min:.0f} mL/min; albumin={albumin_g_dL:.1f} g/dL"
    ]
    if arc:
        notes_parts.append("Augmented renal clearance (GFR>130) detected")
    if albumin_g_dL < 3.5:
        notes_parts.append(
            f"Hypoalbuminemia: fu correction applied (factor {4.0 / max(albumin_g_dL, 0.5):.2f})"
        )
    notes = ". ".join(notes_parts)

    return CriticalCarePKResult(
        drug_name=drug_name,
        patient_weight_kg=patient_weight_kg,
        condition=condition,
        gfr_mL_per_min=gfr_mL_per_min,
        albumin_g_dL=albumin_g_dL,
        cl_scaling_factor=cl_factor,
        vd_scaling_factor=vd_factor,
        adjusted_cl_L_per_h=adjusted_cl,
        adjusted_vd_L=adjusted_vd,
        adjusted_t_half_h=adjusted_t_half,
        augmented_renal_clearance=arc,
        recommended_dose_mg=recommended_dose,
        recommended_interval_h=recommended_interval,
        loading_dose_mg=loading_dose,
        monitoring_recommendation=monitoring,
        notes=notes,
    )


def sepsis_pk_adjustment(
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    weight_kg: float = 70.0,
    gfr_mL_per_min: float = 90.0,
    albumin_g_dL: float = 3.0,
) -> dict:
    """Return CL and Vd scaling factors specific to sepsis physiology."""
    arc = gfr_mL_per_min > 130
    vd_factor = 1.4 + 0.3 * (1.0 if weight_kg > 100 else 0.0)
    cl_factor = _sepsis_cl_factor(gfr_mL_per_min)

    # Albumin effect
    albumin_effect = 1.0
    if albumin_g_dL < 3.5:
        albumin_effect = min(4.0 / max(albumin_g_dL, 0.5), 2.0)
        cl_factor *= albumin_effect

    return {
        "cl_factor": cl_factor,
        "vd_factor": vd_factor,
        "arc": arc,
        "albumin_effect": albumin_effect,
    }
