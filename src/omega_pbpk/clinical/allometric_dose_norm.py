"""Allometric dose normalization across patient populations and species.

Provides BSA-based dosing, weight-based dosing, allometric PK scaling,
pediatric dose estimation, and geriatric dose adjustment.
"""

from __future__ import annotations

from math import sqrt


def bsa_dubois(weight_kg: float, height_cm: float) -> float:
    """Body surface area using DuBois formula.

    BSA = 0.007184 * weight^0.425 * height^0.725  (m²)

    Parameters
    ----------
    weight_kg:
        Body weight in kg (must be > 0).
    height_cm:
        Height in cm (must be > 0).

    Returns
    -------
    float
        BSA in m².
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    return 0.007184 * (weight_kg**0.425) * (height_cm**0.725)


def bsa_mosteller(weight_kg: float, height_cm: float) -> float:
    """Body surface area using Mosteller formula.

    BSA = sqrt(weight * height / 3600)  (m²)

    Parameters
    ----------
    weight_kg:
        Body weight in kg (must be > 0).
    height_cm:
        Height in cm (must be > 0).

    Returns
    -------
    float
        BSA in m².
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    return sqrt(weight_kg * height_cm / 3600.0)


def dose_by_bsa(
    dose_per_m2: float,
    weight_kg: float,
    height_cm: float,
    method: str = "mosteller",
    max_bsa: float = 2.0,
) -> float:
    """Calculate dose based on body surface area.

    dose = dose_per_m2 * min(BSA, max_bsa)

    Parameters
    ----------
    dose_per_m2:
        Dose per m² of BSA (same units as return value).
    weight_kg:
        Body weight in kg.
    height_cm:
        Height in cm.
    method:
        BSA formula: 'mosteller' or 'dubois'.
    max_bsa:
        Maximum BSA to cap dose calculation (m²). Default 2.0 m².

    Returns
    -------
    float
        Calculated dose in same units as dose_per_m2.
    """
    if method == "mosteller":
        bsa = bsa_mosteller(weight_kg, height_cm)
    elif method == "dubois":
        bsa = bsa_dubois(weight_kg, height_cm)
    else:
        raise ValueError("method must be 'mosteller' or 'dubois'")
    return dose_per_m2 * min(bsa, max_bsa)


def dose_by_weight(
    dose_per_kg: float,
    weight_kg: float,
    max_dose: float | None = None,
) -> float:
    """Calculate dose based on body weight.

    dose = dose_per_kg * weight_kg

    Parameters
    ----------
    dose_per_kg:
        Dose per kg of body weight.
    weight_kg:
        Body weight in kg.
    max_dose:
        Optional maximum dose cap.

    Returns
    -------
    float
        Calculated dose.
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    dose = dose_per_kg * weight_kg
    if max_dose is not None:
        dose = min(dose, max_dose)
    return dose


def allometric_scale_cl(
    cl_ref: float,
    weight_ref_kg: float,
    weight_target_kg: float,
    exponent: float = 0.75,
) -> float:
    """Scale clearance allometrically between body weights.

    CL_target = CL_ref * (weight_target / weight_ref)^exponent

    Parameters
    ----------
    cl_ref:
        Reference clearance (any units, e.g., L/h).
    weight_ref_kg:
        Reference body weight in kg.
    weight_target_kg:
        Target body weight in kg.
    exponent:
        Allometric exponent. Default 0.75 (three-quarters power law).

    Returns
    -------
    float
        Scaled clearance in same units as cl_ref.
    """
    if weight_ref_kg <= 0:
        raise ValueError("weight_ref_kg must be > 0")
    if weight_target_kg <= 0:
        raise ValueError("weight_target_kg must be > 0")
    return cl_ref * (weight_target_kg / weight_ref_kg) ** exponent


def allometric_scale_vd(
    vd_ref: float,
    weight_ref_kg: float,
    weight_target_kg: float,
    exponent: float = 1.0,
) -> float:
    """Scale volume of distribution allometrically between body weights.

    Vd_target = Vd_ref * (weight_target / weight_ref)^exponent

    Parameters
    ----------
    vd_ref:
        Reference Vd (any units, e.g., L).
    weight_ref_kg:
        Reference body weight in kg.
    weight_target_kg:
        Target body weight in kg.
    exponent:
        Allometric exponent. Default 1.0 (linear scaling).

    Returns
    -------
    float
        Scaled Vd in same units as vd_ref.
    """
    if weight_ref_kg <= 0:
        raise ValueError("weight_ref_kg must be > 0")
    if weight_target_kg <= 0:
        raise ValueError("weight_target_kg must be > 0")
    return vd_ref * (weight_target_kg / weight_ref_kg) ** exponent


def pediatric_dose_from_adult(
    adult_dose_mg: float,
    adult_weight_kg: float,
    child_weight_kg: float,
    method: str = "weight_based",
    child_height_cm: float | None = None,
    dose_per_m2: float | None = None,
) -> dict:
    """Estimate pediatric dose from adult dose.

    Parameters
    ----------
    adult_dose_mg:
        Adult dose in mg.
    adult_weight_kg:
        Adult body weight in kg.
    child_weight_kg:
        Child body weight in kg.
    method:
        Scaling method: 'weight_based' (allometric 0.75 exponent) or 'bsa'.
    child_height_cm:
        Child height in cm (required for 'bsa' method).
    dose_per_m2:
        Dose per m² (required for 'bsa' method).

    Returns
    -------
    dict with keys:
        recommended_dose_mg, method_used, scaling_factor, notes.
    """
    if child_weight_kg <= 0:
        raise ValueError("child_weight_kg must be > 0")
    if adult_weight_kg <= 0:
        raise ValueError("adult_weight_kg must be > 0")

    if method == "weight_based":
        scaling_factor = (child_weight_kg / adult_weight_kg) ** 0.75
        recommended_dose_mg = adult_dose_mg * scaling_factor
        notes = (
            f"Allometric scaling (exponent=0.75): child {child_weight_kg} kg "
            f"vs adult {adult_weight_kg} kg"
        )
    elif method == "bsa":
        if child_height_cm is None:
            raise ValueError("child_height_cm is required for 'bsa' method")
        if dose_per_m2 is None:
            raise ValueError("dose_per_m2 is required for 'bsa' method")
        child_bsa = bsa_mosteller(child_weight_kg, child_height_cm)
        recommended_dose_mg = dose_per_m2 * child_bsa
        # Estimate adult BSA for reference (assume height 170 cm)
        adult_bsa = bsa_mosteller(adult_weight_kg, 170.0)
        scaling_factor = child_bsa / adult_bsa if adult_bsa > 0 else 0.0
        notes = f"BSA-based dosing: child BSA={child_bsa:.2f} m²; dose_per_m2={dose_per_m2} mg/m²"
    else:
        raise ValueError("method must be 'weight_based' or 'bsa'")

    return {
        "recommended_dose_mg": recommended_dose_mg,
        "method_used": method,
        "scaling_factor": scaling_factor,
        "notes": notes,
    }


def geriatric_dose_adjustment(
    adult_dose_mg: float,
    age_years: float,
    egfr: float = 90.0,
    albumin_g_dL: float = 4.0,
) -> dict:
    """Adjust dose for geriatric patients based on age, renal function, and albumin.

    Age-based CL decline: 1% per year over age 65.
    eGFR correction: proportional reduction if eGFR < 90 mL/min/1.73m².
    Albumin correction: proportional reduction if albumin < 4.0 g/dL.

    Parameters
    ----------
    adult_dose_mg:
        Standard adult dose in mg.
    age_years:
        Patient age in years.
    egfr:
        Estimated GFR in mL/min/1.73m². Default 90 (normal).
    albumin_g_dL:
        Serum albumin in g/dL. Default 4.0 (normal).

    Returns
    -------
    dict with keys:
        adjusted_dose_mg, combined_factor, age_factor, egfr_factor, notes.
    """
    if age_years < 0:
        raise ValueError("age_years must be >= 0")
    if egfr < 0:
        raise ValueError("egfr must be >= 0")

    # Age-based factor: 1% per year over 65
    age_factor = 1.0 - max(0.0, (age_years - 65.0) * 0.01)
    age_factor = max(0.0, age_factor)  # floor at 0

    # eGFR correction
    egfr_factor = min(1.0, egfr / 90.0)

    # Albumin correction (simplified for highly protein-bound drugs)
    alb_factor = min(1.0, albumin_g_dL / 4.0)

    # Combined factor (age + renal; albumin noted separately)
    combined_factor = age_factor * egfr_factor

    adjusted_dose_mg = adult_dose_mg * combined_factor

    parts = []
    if age_years > 65:
        parts.append(f"age {age_years:.0f}y → age_factor={age_factor:.2f}")
    if egfr < 90:
        parts.append(f"eGFR={egfr:.0f} → egfr_factor={egfr_factor:.2f}")
    if albumin_g_dL < 4.0:
        parts.append(
            f"albumin={albumin_g_dL:.1f} g/dL → alb_factor={alb_factor:.2f} (consider dose reduction)"
        )
    notes = "; ".join(parts) if parts else "No adjustment needed"

    return {
        "adjusted_dose_mg": adjusted_dose_mg,
        "combined_factor": combined_factor,
        "age_factor": age_factor,
        "egfr_factor": egfr_factor,
        "notes": notes,
    }
