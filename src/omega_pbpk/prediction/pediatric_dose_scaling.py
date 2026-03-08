"""Phase 229: Pediatric dose scaling using allometric and maturation functions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PediatricDoseResult:
    """Result of pediatric dose scaling calculation."""

    drug_name: str
    adult_dose_mg: float
    adult_weight_kg: float
    child_weight_kg: float
    child_age_months: float
    age_group: str
    dose_metric: str
    scaled_dose_mg: float
    dose_per_kg: float
    dose_ratio: float
    maturation_factor: float
    notes: str


def _get_age_group(age_months: float) -> str:
    """Determine age group from age in months."""
    if age_months < 1:
        return "neonates"
    elif age_months < 24:
        return "infants"
    elif age_months < 60:
        return "toddlers"
    elif age_months < 144:
        return "children"
    else:
        return "adolescents"


def _estimate_height_m(age_months: float) -> float:
    """Estimate height in metres for young children."""
    return 0.35 + 0.065 * age_months


def _bsa_m2(height_m: float, weight_kg: float) -> float:
    """BSA in m^2 using Du Bois-inspired formula sqrt(H*W/3600)."""
    return math.sqrt(height_m * weight_kg / 3600.0)


def _maturation_factor(age_months: float, tm50: float = 47.0, hill: float = 3.4) -> float:
    """CYP3A4 maturation function (sigmoid-Hill)."""
    if age_months <= 0:
        return 0.0
    return (age_months**hill) / (tm50**hill + age_months**hill)


def scale_pediatric_dose(
    drug_name: str,
    adult_dose_mg: float,
    adult_weight_kg: float,
    child_weight_kg: float,
    child_age_months: float,
    dose_metric: str = "allometric",
    adult_age_months: float = 360.0,  # 30 years as reference
) -> PediatricDoseResult:
    """Scale adult dose to pediatric patient.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    adult_dose_mg:
        Adult dose in mg.
    adult_weight_kg:
        Adult body weight in kg.
    child_weight_kg:
        Child body weight in kg.
    child_age_months:
        Child age in months.
    dose_metric:
        Scaling method: "allometric", "bsa", "mg_per_kg", or "maturation_function".
    adult_age_months:
        Reference adult age in months (default 360 = 30 yr).

    Returns
    -------
    PediatricDoseResult
    """
    # Input validation
    if adult_dose_mg <= 0:
        raise ValueError("adult_dose_mg must be > 0")
    if child_weight_kg <= 0:
        raise ValueError("child_weight_kg must be > 0")
    if adult_weight_kg <= 0:
        raise ValueError("adult_weight_kg must be > 0")

    age_group = _get_age_group(child_age_months)
    fm_child = _maturation_factor(child_age_months)
    fm_adult = _maturation_factor(adult_age_months)
    weight_ratio = child_weight_kg / adult_weight_kg
    notes_parts: list[str] = []

    if dose_metric == "allometric":
        scaled = adult_dose_mg * (weight_ratio**0.75)
        notes_parts.append("Allometric exponent 0.75 applied")
    elif dose_metric == "bsa":
        child_h = _estimate_height_m(child_age_months)
        adult_h = _estimate_height_m(adult_age_months)
        bsa_child = _bsa_m2(child_h, child_weight_kg)
        bsa_adult = _bsa_m2(adult_h, adult_weight_kg)
        scaled = adult_dose_mg * (bsa_child / bsa_adult)
        notes_parts.append(f"BSA ratio {bsa_child / bsa_adult:.3f}")
    elif dose_metric == "mg_per_kg":
        scaled = adult_dose_mg / adult_weight_kg * child_weight_kg
        notes_parts.append("mg/kg proportional scaling")
    elif dose_metric == "maturation_function":
        # D_child = D_adult * weight_ratio * fm_child / fm_adult
        if fm_adult <= 0:
            fm_adult = 1.0  # fallback
        scaled = adult_dose_mg * weight_ratio * fm_child / fm_adult
        notes_parts.append(f"CYP3A4 maturation: fm_child={fm_child:.3f}, fm_adult={fm_adult:.3f}")
    else:
        raise ValueError(f"Unknown dose_metric: {dose_metric!r}")

    # Cap at adult dose
    if scaled > adult_dose_mg:
        scaled = adult_dose_mg
        notes_parts.append("Capped at adult dose")

    dose_per_kg = scaled / child_weight_kg
    dose_ratio = scaled / adult_dose_mg

    return PediatricDoseResult(
        drug_name=drug_name,
        adult_dose_mg=adult_dose_mg,
        adult_weight_kg=adult_weight_kg,
        child_weight_kg=child_weight_kg,
        child_age_months=child_age_months,
        age_group=age_group,
        dose_metric=dose_metric,
        scaled_dose_mg=round(scaled, 4),
        dose_per_kg=round(dose_per_kg, 4),
        dose_ratio=round(dose_ratio, 6),
        maturation_factor=round(fm_child, 6),
        notes="; ".join(notes_parts) if notes_parts else "",
    )


def compare_scaling_methods(
    drug_name: str,
    adult_dose_mg: float,
    adult_weight_kg: float,
    child_weight_kg: float,
    child_age_months: float,
) -> list[PediatricDoseResult]:
    """Compare all four scaling methods, sorted by scaled_dose descending."""
    methods = ["allometric", "bsa", "mg_per_kg", "maturation_function"]
    results = [
        scale_pediatric_dose(
            drug_name=drug_name,
            adult_dose_mg=adult_dose_mg,
            adult_weight_kg=adult_weight_kg,
            child_weight_kg=child_weight_kg,
            child_age_months=child_age_months,
            dose_metric=m,
        )
        for m in methods
    ]
    return sorted(results, key=lambda r: r.scaled_dose_mg, reverse=True)
