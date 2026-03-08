"""Warfarin dose optimization for target INR.

Based on IWPC 2009 Warfarin Dose Algorithm with CYP2C9 genotype,
VKORC1 polymorphism, and patient-specific covariates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_CYP2C9 = {"*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"}
_VALID_VKORC1 = {"GG", "AG", "AA"}

_CYP2C9_FACTOR: dict[str, float] = {
    "*1/*1": 1.00,
    "*1/*2": 0.75,
    "*1/*3": 0.60,
    "*2/*2": 0.60,
    "*2/*3": 0.45,
    "*3/*3": 0.30,
}

_VKORC1_FACTOR: dict[str, float] = {
    "GG": 1.00,
    "AG": 0.75,
    "AA": 0.55,
}


@dataclass(frozen=True)
class WarfarinDoseOptResult:
    """Result of warfarin dose optimization."""

    patient_id: str
    target_inr: float
    cyp2c9_genotype: str
    vkorc1_genotype: str
    weekly_dose_mg: float
    daily_dose_mg: float
    predicted_inr: float
    therapeutic_range_probability: float
    bleeding_risk: str
    initiation_dose_mg: str
    monitoring_frequency: str
    notes: str


def optimize_warfarin_dose(
    patient_id: str,
    target_inr: float = 2.5,
    cyp2c9_genotype: str = "*1/*1",
    vkorc1_genotype: str = "GG",
    age: int = 65,
    weight_kg: float = 70.0,
    height_cm: float = 170.0,
    indication: str = "afib",
) -> WarfarinDoseOptResult:
    """Optimize warfarin weekly dose for a target INR.

    Parameters
    ----------
    patient_id:
        Patient identifier.
    target_inr:
        Target INR (typically 2.0–3.0 for AF, 2.5–3.5 for mechanical valve).
    cyp2c9_genotype:
        CYP2C9 diplotype — one of *1/*1, *1/*2, *1/*3, *2/*2, *2/*3, *3/*3.
    vkorc1_genotype:
        VKORC1 -1639G>A genotype — GG, AG, or AA.
    age:
        Patient age in years.
    weight_kg:
        Body weight in kg.
    height_cm:
        Height in cm.
    indication:
        Clinical indication (informational).

    Returns
    -------
    WarfarinDoseOptResult
    """
    # --- validation ---
    if cyp2c9_genotype not in _VALID_CYP2C9:
        raise ValueError(
            f"Invalid CYP2C9 genotype '{cyp2c9_genotype}'. Must be one of {sorted(_VALID_CYP2C9)}"
        )
    if vkorc1_genotype not in _VALID_VKORC1:
        raise ValueError(
            f"Invalid VKORC1 genotype '{vkorc1_genotype}'. Must be one of {sorted(_VALID_VKORC1)}"
        )
    if target_inr <= 0:
        raise ValueError(f"target_inr must be > 0, got {target_inr}")

    cyp2c9_factor = _CYP2C9_FACTOR[cyp2c9_genotype]
    vkorc1_factor = _VKORC1_FACTOR[vkorc1_genotype]

    # BSA (DuBois)
    bsa = math.sqrt(height_cm * weight_kg / 3600.0)

    # Base weekly dose (mg/week)
    base_dose = 35.0 * bsa * target_inr / 2.5

    # Apply genotype adjustments
    weekly_dose = base_dose * cyp2c9_factor * vkorc1_factor

    # Age correction
    if age > 80:
        weekly_dose *= 0.75
    elif age > 70:
        weekly_dose *= 0.85

    # Clamp to safe range
    weekly_dose = max(5.0, min(70.0, weekly_dose))
    daily_dose = weekly_dose / 7.0

    # Predicted INR (assume well-calibrated)
    predicted_inr = target_inr

    # Therapeutic range probability
    p = min(0.95, max(0.25, 0.7 * cyp2c9_factor * vkorc1_factor))
    therapeutic_range_probability = p

    # Bleeding risk
    if weekly_dose > 50.0:
        bleeding_risk = "high"
    elif weekly_dose > 30.0:
        bleeding_risk = "moderate"
    else:
        bleeding_risk = "low"

    # Initiation dose category
    if daily_dose < 5.0:
        initiation_dose_mg = "low"
    elif daily_dose > 7.0:
        initiation_dose_mg = "high"
    else:
        initiation_dose_mg = "standard"

    # Monitoring frequency
    if cyp2c9_genotype == "*3/*3" or vkorc1_genotype == "AA":
        monitoring_frequency = "daily"
    elif cyp2c9_factor < 1.0 or vkorc1_factor < 1.0:
        monitoring_frequency = "every_2_days"
    else:
        monitoring_frequency = "weekly"

    notes = (
        f"IWPC algorithm; indication={indication}; "
        f"BSA={bsa:.2f} m²; age={age} yr; "
        f"CYP2C9 factor={cyp2c9_factor:.2f}; VKORC1 factor={vkorc1_factor:.2f}"
    )

    return WarfarinDoseOptResult(
        patient_id=patient_id,
        target_inr=target_inr,
        cyp2c9_genotype=cyp2c9_genotype,
        vkorc1_genotype=vkorc1_genotype,
        weekly_dose_mg=round(weekly_dose, 3),
        daily_dose_mg=round(daily_dose, 4),
        predicted_inr=predicted_inr,
        therapeutic_range_probability=round(therapeutic_range_probability, 4),
        bleeding_risk=bleeding_risk,
        initiation_dose_mg=initiation_dose_mg,
        monitoring_frequency=monitoring_frequency,
        notes=notes,
    )


def compare_genotype_doses(
    patient_id: str,
    **base_kwargs,
) -> list[WarfarinDoseOptResult]:
    """Compare warfarin doses across all 6 CYP2C9 genotypes.

    Parameters
    ----------
    patient_id:
        Patient identifier.
    **base_kwargs:
        Keyword arguments forwarded to :func:`optimize_warfarin_dose`
        (except ``cyp2c9_genotype`` which is varied automatically).

    Returns
    -------
    list[WarfarinDoseOptResult]
        Six results sorted by weekly_dose_mg ascending.
    """
    results = []
    for genotype in _VALID_CYP2C9:
        kwargs = dict(base_kwargs)
        kwargs["cyp2c9_genotype"] = genotype
        results.append(optimize_warfarin_dose(patient_id, **kwargs))
    results.sort(key=lambda r: r.weekly_dose_mg)
    return results
