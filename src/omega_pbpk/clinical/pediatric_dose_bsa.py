"""Phase 1112 — BSA-based pediatric dosing using Mosteller formula."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PediatricDoseBSAResult:
    drug_name: str
    adult_dose_mg: float
    weight_kg: float
    height_cm: float
    age_years: float
    bsa_m2: float
    adult_bsa_m2: float
    bsa_fraction: float
    recommended_dose_mg: float
    age_group: str
    dose_capped: bool
    max_single_dose_mg: float
    notes: str


def calculate_bsa(weight_kg: float, height_cm: float) -> float:
    """Calculate body surface area using Mosteller formula.

    BSA (m²) = sqrt(weight_kg * height_cm / 3600)

    Args:
        weight_kg: Body weight in kilograms.
        height_cm: Height in centimeters.

    Returns:
        BSA in m².
    """
    return math.sqrt(weight_kg * height_cm / 3600.0)


def _age_group_label(age_years: float) -> str:
    """Return age group label based on age in years."""
    if age_years < 0.083:
        return "neonate"
    elif age_years < 2:
        return "infant"
    elif age_years < 5:
        return "toddler"
    elif age_years < 12:
        return "child"
    else:
        return "adolescent"


def pediatric_dose_bsa(
    drug_name: str,
    adult_dose_mg: float,
    weight_kg: float,
    height_cm: float,
    age_years: float,
    adult_bsa_m2: float = 1.73,
) -> PediatricDoseBSAResult:
    """Calculate BSA-based pediatric dose.

    Recommended dose = adult_dose * (child_BSA / adult_BSA).
    Dose is capped at adult_dose_mg if BSA ratio > 1.

    Args:
        drug_name: Name of the drug.
        adult_dose_mg: Standard adult dose in mg.
        weight_kg: Child's weight in kg.
        height_cm: Child's height in cm.
        age_years: Child's age in years.
        adult_bsa_m2: Reference adult BSA in m² (default 1.73).

    Returns:
        PediatricDoseBSAResult with recommended dose and metadata.
    """
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be > 0, got {adult_dose_mg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if height_cm <= 0:
        raise ValueError(f"height_cm must be > 0, got {height_cm}")
    if age_years < 0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")
    if adult_bsa_m2 <= 0:
        raise ValueError(f"adult_bsa_m2 must be > 0, got {adult_bsa_m2}")

    bsa_m2 = calculate_bsa(weight_kg, height_cm)
    bsa_fraction = bsa_m2 / adult_bsa_m2
    raw_dose = adult_dose_mg * bsa_fraction

    dose_capped = bsa_fraction > 1.0
    recommended_dose_mg = min(raw_dose, adult_dose_mg)

    age_group = _age_group_label(age_years)

    notes = (
        f"BSA {bsa_m2:.3f} m² (Mosteller). "
        f"BSA fraction {bsa_fraction:.3f} vs adult {adult_bsa_m2} m². "
        f"Age group: {age_group}."
    )
    if dose_capped:
        notes += " Dose capped at adult dose."

    return PediatricDoseBSAResult(
        drug_name=drug_name,
        adult_dose_mg=adult_dose_mg,
        weight_kg=weight_kg,
        height_cm=height_cm,
        age_years=age_years,
        bsa_m2=bsa_m2,
        adult_bsa_m2=adult_bsa_m2,
        bsa_fraction=bsa_fraction,
        recommended_dose_mg=recommended_dose_mg,
        age_group=age_group,
        dose_capped=dose_capped,
        max_single_dose_mg=adult_dose_mg,
        notes=notes,
    )


def dose_by_age_group(
    drug_name: str,
    adult_dose_mg: float,
    weight_kg: float,
    height_cm: float,
    age_years: float,
) -> PediatricDoseBSAResult:
    """Calculate BSA-based pediatric dose with age group label.

    Identical to pediatric_dose_bsa using default adult_bsa_m2=1.73.
    Age group is derived from age_years.

    Args:
        drug_name: Name of the drug.
        adult_dose_mg: Standard adult dose in mg.
        weight_kg: Child's weight in kg.
        height_cm: Child's height in cm.
        age_years: Child's age in years.

    Returns:
        PediatricDoseBSAResult with age group label and recommended dose.
    """
    return pediatric_dose_bsa(
        drug_name=drug_name,
        adult_dose_mg=adult_dose_mg,
        weight_kg=weight_kg,
        height_cm=height_cm,
        age_years=age_years,
    )
