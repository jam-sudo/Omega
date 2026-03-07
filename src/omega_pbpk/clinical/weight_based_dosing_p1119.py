"""Phase 1119 — Weight-based and allometric dose calculation for oncology and biologics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightBasedDoseResult:
    drug_name: str
    method: str  # "mg_per_kg", "allometric", "bsa_based"
    patient_weight_kg: float
    patient_bsa_m2: float  # 0.0 if not BSA method
    calculated_dose_mg: float
    final_dose_mg: float  # after applying min/max cap
    dose_capped: bool
    cap_reason: str  # "max_exceeded", "min_not_reached", "none"
    dose_intensity_pct: float  # final/calculated * 100
    notes: str


def _apply_cap(
    calculated_dose_mg: float,
    max_dose_mg: float,
    min_dose_mg: float,
) -> tuple[float, bool, str]:
    """Apply min/max dose capping. Returns (final_dose, capped, reason)."""
    if calculated_dose_mg > max_dose_mg:
        return max_dose_mg, True, "max_exceeded"
    if calculated_dose_mg < min_dose_mg:
        return min_dose_mg, True, "min_not_reached"
    return calculated_dose_mg, False, "none"


def calculate_weight_based_dose(
    drug_name: str,
    dose_per_kg_mg: float,
    weight_kg: float,
    max_dose_mg: float = float("inf"),
    min_dose_mg: float = 0.0,
) -> WeightBasedDoseResult:
    """Calculate weight-based dose (mg/kg * weight)."""
    if dose_per_kg_mg <= 0:
        raise ValueError("dose_per_kg_mg must be positive")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")

    calculated = dose_per_kg_mg * weight_kg
    final, capped, reason = _apply_cap(calculated, max_dose_mg, min_dose_mg)
    intensity = (final / calculated) * 100.0 if calculated > 0 else 100.0

    notes = f"Dose: {dose_per_kg_mg} mg/kg × {weight_kg} kg = {calculated:.2f} mg"
    if capped:
        notes += f"; capped to {final:.2f} mg ({reason})"

    return WeightBasedDoseResult(
        drug_name=drug_name,
        method="mg_per_kg",
        patient_weight_kg=weight_kg,
        patient_bsa_m2=0.0,
        calculated_dose_mg=calculated,
        final_dose_mg=final,
        dose_capped=capped,
        cap_reason=reason,
        dose_intensity_pct=intensity,
        notes=notes,
    )


def allometric_dose_adjustment(
    drug_name: str,
    reference_dose_mg: float,
    reference_weight_kg: float,
    patient_weight_kg: float,
    exponent: float = 0.75,
) -> WeightBasedDoseResult:
    """Allometric dose scaling: dose = ref_dose * (patient_wt / ref_wt) ** exponent."""
    if reference_weight_kg <= 0:
        raise ValueError("reference_weight_kg must be positive")
    if patient_weight_kg <= 0:
        raise ValueError("patient_weight_kg must be positive")

    ratio = patient_weight_kg / reference_weight_kg
    calculated = reference_dose_mg * (ratio**exponent)
    # No explicit min/max cap for allometric (default inf/0)
    final, capped, reason = _apply_cap(calculated, float("inf"), 0.0)
    intensity = (final / calculated) * 100.0 if calculated > 0 else 100.0

    notes = (
        f"Allometric: {reference_dose_mg} mg × "
        f"({patient_weight_kg}/{reference_weight_kg})^{exponent} = {calculated:.2f} mg"
    )

    return WeightBasedDoseResult(
        drug_name=drug_name,
        method="allometric",
        patient_weight_kg=patient_weight_kg,
        patient_bsa_m2=0.0,
        calculated_dose_mg=calculated,
        final_dose_mg=final,
        dose_capped=capped,
        cap_reason=reason,
        dose_intensity_pct=intensity,
        notes=notes,
    )


def body_surface_area_dose(
    drug_name: str,
    dose_per_m2_mg: float,
    weight_kg: float,
    height_cm: float,
    max_dose_mg: float = float("inf"),
    min_dose_mg: float = 0.0,
) -> WeightBasedDoseResult:
    """BSA-based dosing using Mosteller formula: BSA = sqrt(weight*height/3600)."""
    if dose_per_m2_mg <= 0:
        raise ValueError("dose_per_m2_mg must be positive")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")

    bsa = math.sqrt(weight_kg * height_cm / 3600.0)
    calculated = dose_per_m2_mg * bsa
    final, capped, reason = _apply_cap(calculated, max_dose_mg, min_dose_mg)
    intensity = (final / calculated) * 100.0 if calculated > 0 else 100.0

    notes = (
        f"BSA (Mosteller): sqrt({weight_kg}×{height_cm}/3600) = {bsa:.4f} m²; "
        f"dose = {dose_per_m2_mg} mg/m² × {bsa:.4f} m² = {calculated:.2f} mg"
    )
    if capped:
        notes += f"; capped to {final:.2f} mg ({reason})"

    return WeightBasedDoseResult(
        drug_name=drug_name,
        method="bsa_based",
        patient_weight_kg=weight_kg,
        patient_bsa_m2=bsa,
        calculated_dose_mg=calculated,
        final_dose_mg=final,
        dose_capped=capped,
        cap_reason=reason,
        dose_intensity_pct=intensity,
        notes=notes,
    )
