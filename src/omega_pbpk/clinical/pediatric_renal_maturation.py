"""Pediatric renal maturation model using the Rhodin sigmoid approach.

Models postnatal GFR maturation as a function of postmenstrual age (PMA),
enabling renal clearance dose adjustment in neonates, infants, and children.

References:
    Rhodin MM et al., Pediatr Nephrol, 2009 (GFR maturation sigmoid)
    Schwartz GJ et al., JASN, 2009 (Schwartz formula for creatinine)
    Mosteller RD, N Engl J Med, 1987 (BSA calculation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PediatricRenalResult:
    """Result of pediatric renal maturation prediction.

    Attributes:
        age_weeks_pma: Postmenstrual age in weeks.
        age_weeks_postnatal: Postnatal age in weeks.
        gestational_age_weeks: Gestational age at birth in weeks.
        gfr_mL_per_min: Absolute GFR in mL/min.
        gfr_normalized_mL_per_min_per_1_73m2: GFR normalized to 1.73 m2 BSA.
        gfr_fraction_of_adult: Fraction of mature adult GFR (0-1).
        renal_cl_adjustment_factor: Multiply adult renal CL by this factor.
        creatinine_mg_dL: Predicted serum creatinine in mg/dL.
        renal_maturation_stage: One of "neonatal", "infant", "toddler", "child", "mature".
        notes: Descriptive notes about the prediction.
    """

    age_weeks_pma: float
    age_weeks_postnatal: float
    gestational_age_weeks: float
    gfr_mL_per_min: float
    gfr_normalized_mL_per_min_per_1_73m2: float
    gfr_fraction_of_adult: float
    renal_cl_adjustment_factor: float
    creatinine_mg_dL: float
    renal_maturation_stage: str
    notes: str


def _estimate_height_cm(age_weeks_postnatal: float, gestational_age_weeks: float) -> float:
    """Estimate height from postnatal and gestational age.

    Uses a simple piecewise growth model.

    Args:
        age_weeks_postnatal: Postnatal age in weeks.
        gestational_age_weeks: Gestational age at birth in weeks.

    Returns:
        Estimated height in cm.
    """
    # At birth: term ~50 cm, preterm ~30-40 cm proportional to gestational age
    birth_height_cm = max(25.0, 50.0 * (gestational_age_weeks / 40.0))
    # Growth rate: approximately 0.5 cm/week in first 2 years, then ~0.25 cm/week
    age_weeks_total = age_weeks_postnatal
    if age_weeks_total <= 104.0:  # first 2 years
        height_cm = birth_height_cm + 0.5 * age_weeks_total
    else:
        height_at_2y = birth_height_cm + 0.5 * 104.0
        height_cm = height_at_2y + 0.25 * (age_weeks_total - 104.0)
    # Cap at adult height of 170 cm
    return min(height_cm, 170.0)


def _compute_bsa_m2(height_cm: float, weight_kg: float) -> float:
    """Compute body surface area using Mosteller formula.

    BSA (m2) = sqrt(height_cm * weight_kg / 3600)

    Args:
        height_cm: Height in cm.
        weight_kg: Weight in kg.

    Returns:
        BSA in m2.
    """
    return math.sqrt(height_cm * weight_kg / 3600.0)


def _gfr_fraction_sigmoid(pma_weeks: float) -> float:
    """Sigmoid maturation function for GFR based on postmenstrual age.

    Uses logistic approximation to Rhodin model:
        GFR_fraction = 1 / (1 + exp(-0.42 * (PMA - 36.4)))

    Args:
        pma_weeks: Postmenstrual age in weeks.

    Returns:
        GFR fraction of adult value, capped at 1.0.
    """
    fraction = 1.0 / (1.0 + math.exp(-0.42 * (pma_weeks - 36.4)))
    return min(fraction, 1.0)


def _renal_maturation_stage(pma_weeks: float) -> str:
    """Classify renal maturation stage based on PMA.

    Args:
        pma_weeks: Postmenstrual age in weeks.

    Returns:
        Stage label: "neonatal", "infant", "toddler", "child", or "mature".
    """
    if pma_weeks < 36.0:
        return "neonatal"
    if pma_weeks < 52.0:
        return "infant"
    if pma_weeks < 104.0:
        return "toddler"
    if pma_weeks < 520.0:
        return "child"
    return "mature"


def predict_gfr_maturation(
    age_weeks_postnatal: float,
    gestational_age_weeks: float,
    weight_kg: float,
) -> PediatricRenalResult:
    """Predict GFR and renal function based on postnatal and gestational age.

    Uses the sigmoid (Rhodin) maturation model to compute GFR fraction of
    adult reference, absolute GFR, BSA-normalized GFR, and predicted serum
    creatinine.

    Args:
        age_weeks_postnatal: Postnatal age in weeks (>= 0).
        gestational_age_weeks: Gestational age at birth in weeks (20-45).
        weight_kg: Body weight in kg (> 0).

    Returns:
        PediatricRenalResult with all computed renal parameters.

    Raises:
        ValueError: If any input is out of physiological range.
    """
    if age_weeks_postnatal < 0.0:
        raise ValueError(f"age_weeks_postnatal must be >= 0, got {age_weeks_postnatal}")
    if gestational_age_weeks < 20.0 or gestational_age_weeks > 45.0:
        raise ValueError(f"gestational_age_weeks must be 20-45, got {gestational_age_weeks}")
    if weight_kg <= 0.0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")

    pma_weeks = gestational_age_weeks + age_weeks_postnatal

    # GFR fraction via sigmoid approximation to Rhodin model
    gfr_fraction = _gfr_fraction_sigmoid(pma_weeks)

    # Adult GFR reference: 100 mL/min/1.73m2 (sex-neutral)
    adult_gfr_normalized = 100.0  # mL/min/1.73m2

    # Normalized GFR (per 1.73 m2)
    gfr_normalized = gfr_fraction * adult_gfr_normalized

    # Absolute GFR via BSA scaling
    height_cm = _estimate_height_cm(age_weeks_postnatal, gestational_age_weeks)
    bsa_m2 = _compute_bsa_m2(height_cm, weight_kg)
    gfr_absolute = gfr_normalized * bsa_m2 / 1.73

    # Serum creatinine: Schwartz-simplified approach
    # Scr = k * height_cm / GFR  (k=0.55 for children, 0.33 for preterm infants)
    k_schwartz = 0.33 if pma_weeks < 36.0 else 0.55
    creatinine_mg_dL = k_schwartz * height_cm / (gfr_absolute + 0.001)
    # Enforce physiological bounds
    creatinine_mg_dL = max(0.1, min(creatinine_mg_dL, 10.0))

    stage = _renal_maturation_stage(pma_weeks)

    notes = (
        f"PMA={pma_weeks:.1f}w; GFR fraction={gfr_fraction:.3f}; "
        f"BSA={bsa_m2:.3f}m2; height={height_cm:.1f}cm; stage={stage}"
    )

    return PediatricRenalResult(
        age_weeks_pma=pma_weeks,
        age_weeks_postnatal=age_weeks_postnatal,
        gestational_age_weeks=gestational_age_weeks,
        gfr_mL_per_min=gfr_absolute,
        gfr_normalized_mL_per_min_per_1_73m2=gfr_normalized,
        gfr_fraction_of_adult=gfr_fraction,
        renal_cl_adjustment_factor=gfr_fraction,
        creatinine_mg_dL=creatinine_mg_dL,
        renal_maturation_stage=stage,
        notes=notes,
    )


def dose_adjustment_renal_immature(
    drug_name: str,
    adult_dose_mg: float,
    adult_gfr_mL_per_min: float,
    age_weeks_postnatal: float,
    gestational_age_weeks: float,
    weight_kg: float,
    fe_renal: float,
) -> dict:
    """Compute pediatric dose adjusted for renal immaturity.

    Uses GFR-based adjustment: dose_fraction = 1 - fe_renal * (1 - GFR_fraction)

    Args:
        drug_name: Name of the drug.
        adult_dose_mg: Standard adult dose in mg.
        adult_gfr_mL_per_min: Adult GFR reference value in mL/min (informational).
        age_weeks_postnatal: Postnatal age in weeks.
        gestational_age_weeks: Gestational age at birth in weeks.
        weight_kg: Body weight in kg.
        fe_renal: Fraction of drug eliminated renally (0-1).

    Returns:
        Dict with keys:
            "pediatric_dose_mg": Adjusted pediatric dose in mg.
            "dose_fraction": Fraction of adult dose (0-1].
            "gfr_fraction": Fraction of adult GFR.
            "gfr_mL_per_min": Absolute pediatric GFR.
            "renal_maturation_stage": Maturation stage string.
            "notes": Description of the adjustment.
    """
    pediatric_result = predict_gfr_maturation(
        age_weeks_postnatal=age_weeks_postnatal,
        gestational_age_weeks=gestational_age_weeks,
        weight_kg=weight_kg,
    )

    gfr_fraction = pediatric_result.gfr_fraction_of_adult
    # Dose fraction accounts for renally eliminated fraction being reduced
    dose_fraction = 1.0 - fe_renal * (1.0 - gfr_fraction)
    # Clamp to (0, 1]
    dose_fraction = max(0.0, min(1.0, dose_fraction))

    pediatric_dose_mg = adult_dose_mg * dose_fraction

    notes = (
        f"Drug: {drug_name}; fe_renal={fe_renal:.2f}; "
        f"GFR fraction={gfr_fraction:.3f}; dose_fraction={dose_fraction:.3f}; "
        f"adult_dose={adult_dose_mg}mg → pediatric_dose={pediatric_dose_mg:.2f}mg"
    )

    return {
        "pediatric_dose_mg": pediatric_dose_mg,
        "dose_fraction": dose_fraction,
        "gfr_fraction": gfr_fraction,
        "gfr_mL_per_min": pediatric_result.gfr_mL_per_min,
        "renal_maturation_stage": pediatric_result.renal_maturation_stage,
        "notes": notes,
    }
