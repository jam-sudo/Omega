"""
Phase 509 — Pediatric Weight-Based Dosing

Calculate weight-based and BSA-based doses for pediatric patients with
age-specific dose limits using Clark's rule, Young's rule, and Schwartz
CrCl formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PediatricDosingResult:
    """Results of pediatric dose calculation."""

    drug_name: str
    patient_weight_kg: float
    patient_age_years: float
    patient_height_cm: float

    dose_weight_based_mg: float  # from mg/kg
    dose_bsa_based_mg: float  # from mg/m2
    dose_clark_mg: float  # Clark's rule
    dose_young_mg: float  # Young's rule

    bsa_m2: float
    recommended_dose_mg: float  # weight_based, capped by age group
    age_group: str  # neonate/infant/child/adolescent
    adult_dose_fraction: float  # recommended / adult_dose_mg

    renal_dose_adjustment: bool  # True if CrCl < 50 mL/min
    crcl_ml_per_min: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_age(age_years: float) -> str:
    """Return age group string."""
    age_months = age_years * 12.0
    if age_months < 1.0:
        return "neonate"
    if age_months < 12.0:
        return "infant"
    if age_years < 12.0:
        return "child"
    return "adolescent"


def _age_cap_fraction(age_group: str) -> float:
    """Return maximum fraction of adult dose allowed for age group."""
    caps = {
        "neonate": 0.25,
        "infant": 0.50,
        "child": 0.75,
        "adolescent": 1.00,
    }
    return caps[age_group]


def _schwartz_crcl(
    age_years: float, height_cm: float, serum_creatinine_mg_dL: float, sex: str
) -> float:
    """
    Schwartz formula for pediatric CrCl (mL/min).

    k_factor:
      0.45  — infants  (< 1 year)
      0.55  — children (1–12 years, or adolescent female)
      0.70  — adolescent male
    """
    age_months = age_years * 12.0
    if age_months < 12.0:
        k = 0.45
    elif age_years < 12.0:
        k = 0.55
    else:
        # adolescent
        k = 0.70 if sex == "male" else 0.55

    if serum_creatinine_mg_dL <= 0.0:
        return 999.9  # avoid division by zero
    return k * height_cm / serum_creatinine_mg_dL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_pediatric_dose(
    drug_name: str,
    adult_dose_mg: float,
    dose_mg_per_kg: float,
    dose_mg_per_m2: float,
    patient_weight_kg: float,
    patient_age_years: float,
    patient_height_cm: float = 100.0,
    serum_creatinine_mg_dL: float = 0.5,
    sex: str = "male",
) -> PediatricDosingResult:
    """
    Calculate pediatric doses using multiple methods.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    adult_dose_mg : float
        Standard adult dose in mg.
    dose_mg_per_kg : float
        Weight-based dose in mg/kg.
    dose_mg_per_m2 : float
        BSA-based dose in mg/m².
    patient_weight_kg : float
        Patient weight in kg (must be > 0).
    patient_age_years : float
        Patient age in years (must be >= 0).
    patient_height_cm : float
        Patient height in cm (must be > 0).
    serum_creatinine_mg_dL : float
        Serum creatinine concentration.
    sex : str
        "male" or "female".

    Returns
    -------
    PediatricDosingResult
    """
    # --- Validation ---
    if patient_weight_kg <= 0:
        raise ValueError("patient_weight_kg must be > 0")
    if patient_age_years < 0:
        raise ValueError("patient_age_years must be >= 0")
    if patient_height_cm <= 0:
        raise ValueError("patient_height_cm must be > 0")
    if sex not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")

    # --- BSA (Mosteller formula: sqrt(height * weight / 3600)) ---
    bsa_m2 = math.sqrt(patient_height_cm * patient_weight_kg / 3600.0)

    # --- Dose calculations ---
    dose_weight_based_mg = dose_mg_per_kg * patient_weight_kg
    dose_bsa_based_mg = dose_mg_per_m2 * bsa_m2

    # Clark's rule: dose = adult_dose * weight_kg / 70
    dose_clark_mg = adult_dose_mg * patient_weight_kg / 70.0

    # Young's rule: dose = adult_dose * age / (age + 12)
    # For very young children (< 1 month), age_years ≈ 0 → dose ≈ 0
    dose_young_mg = adult_dose_mg * patient_age_years / (patient_age_years + 12.0)

    # --- Age group & safety cap ---
    age_group = _classify_age(patient_age_years)
    cap_fraction = _age_cap_fraction(age_group)
    max_dose_mg = cap_fraction * adult_dose_mg
    recommended_dose_mg = min(dose_weight_based_mg, max_dose_mg)

    adult_dose_fraction = recommended_dose_mg / adult_dose_mg if adult_dose_mg > 0 else 0.0

    # --- Renal dose adjustment (Schwartz CrCl) ---
    crcl = _schwartz_crcl(patient_age_years, patient_height_cm, serum_creatinine_mg_dL, sex)
    renal_dose_adjustment = crcl < 50.0

    # --- Notes ---
    notes_parts: list[str] = []
    if recommended_dose_mg < dose_weight_based_mg:
        notes_parts.append(
            f"Weight-based dose ({dose_weight_based_mg:.1f} mg) capped to "
            f"{max_dose_mg:.1f} mg ({int(cap_fraction * 100)}% adult dose) "
            f"for {age_group}."
        )
    if renal_dose_adjustment:
        notes_parts.append(f"CrCl {crcl:.1f} mL/min: consider renal dose adjustment.")
    notes = " ".join(notes_parts) if notes_parts else "No dose capping required."

    return PediatricDosingResult(
        drug_name=drug_name,
        patient_weight_kg=patient_weight_kg,
        patient_age_years=patient_age_years,
        patient_height_cm=patient_height_cm,
        dose_weight_based_mg=dose_weight_based_mg,
        dose_bsa_based_mg=dose_bsa_based_mg,
        dose_clark_mg=dose_clark_mg,
        dose_young_mg=dose_young_mg,
        bsa_m2=bsa_m2,
        recommended_dose_mg=recommended_dose_mg,
        age_group=age_group,
        adult_dose_fraction=adult_dose_fraction,
        renal_dose_adjustment=renal_dose_adjustment,
        crcl_ml_per_min=crcl,
        notes=notes,
    )


def dose_range_by_age(
    drug_name: str,
    adult_dose_mg: float,
    dose_mg_per_kg: float,
    dose_mg_per_m2: float,
) -> list[PediatricDosingResult]:
    """
    Return dose calculations for four representative pediatric patients.

    Patients:
      - Neonate:    0.1 years, 3.5 kg,  50 cm
      - Infant:     0.5 years, 7 kg,    65 cm
      - Child:      5.0 years, 20 kg,  110 cm
      - Adolescent: 15.0 years, 60 kg, 165 cm
    """
    representative_patients = [
        # (age_years, weight_kg, height_cm, sex)
        # Neonate: ~2 weeks old (0.04 years = 0.5 months < 1 month)
        (0.04, 3.5, 50.0, "male"),
        (0.5, 7.0, 65.0, "male"),
        (5.0, 20.0, 110.0, "male"),
        (15.0, 60.0, 165.0, "male"),
    ]

    results: list[PediatricDosingResult] = []
    for age_years, weight_kg, height_cm, sex in representative_patients:
        result = calculate_pediatric_dose(
            drug_name=drug_name,
            adult_dose_mg=adult_dose_mg,
            dose_mg_per_kg=dose_mg_per_kg,
            dose_mg_per_m2=dose_mg_per_m2,
            patient_weight_kg=weight_kg,
            patient_age_years=age_years,
            patient_height_cm=height_cm,
            sex=sex,
        )
        results.append(result)

    return results
