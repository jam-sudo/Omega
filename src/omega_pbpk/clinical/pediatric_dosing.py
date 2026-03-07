"""Pediatric dose scaling — allometric + maturation function + weight/BSA-based rules."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    # Legacy API (kept for backward compatibility)
    "PediatricDosingResult",
    "pediatric_dose",
    "pediatric_dose_range",
    "allometric_dose_scaling",
    # New API (Phase 228)
    "PediatricDoseResult",
    "bsa_dubois",
    "calculate_pediatric_dose",
    "dose_frequency_adjustment",
]

# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------

# CYP3A4 ontogeny: (age_years, fraction_adult_activity)
CYP3A4_ONTOGENY: list[tuple[float, float]] = [
    (0.0, 0.05),
    (0.25, 0.35),
    (0.5, 0.50),
    (1.0, 0.60),
    (2.0, 0.75),
    (5.0, 0.85),
    (10.0, 0.95),
    (18.0, 1.0),
]

# GFR ontogeny: (age_years, mL/min/1.73m²)
GFR_ONTOGENY: list[tuple[float, float]] = [
    (0.0, 20.0),
    (0.1, 40.0),
    (0.5, 60.0),
    (1.0, 90.0),
    (2.0, 110.0),
    (5.0, 120.0),
    (10.0, 125.0),
    (18.0, 130.0),
]


def _interpolate_ontogeny(age_years: float, table: list[tuple[float, float]]) -> float:
    """Linear interpolation on (age, value) table. Clamp to table bounds."""
    if age_years <= table[0][0]:
        return table[0][1]
    if age_years >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        a0, v0 = table[i]
        a1, v1 = table[i + 1]
        if a0 <= age_years <= a1:
            frac = (age_years - a0) / (a1 - a0)
            return v0 + frac * (v1 - v0)
    return table[-1][1]


def allometric_dose_scaling(
    adult_dose_mg: float,
    adult_weight_kg: float,
    child_weight_kg: float,
    exponent: float = 0.75,
) -> float:
    """Scale adult dose to child using allometric power law."""
    return adult_dose_mg * (child_weight_kg / adult_weight_kg) ** exponent


@dataclass(frozen=True)
class PediatricDosingResult:
    patient_age_years: float
    patient_weight_kg: float
    adult_dose_mg: float
    allometric_dose_mg: float
    cyp3a4_adjusted_dose_mg: float
    renal_adjusted_dose_mg: float
    combined_adjusted_dose_mg: float
    cyp3a4_fraction_adult: float
    gfr_estimated: float
    dose_per_kg: float
    warnings: list[str]


def pediatric_dose(
    adult_dose_mg: float,
    adult_weight_kg: float = 70.0,
    child_age_years: float = 5.0,
    child_weight_kg: float | None = None,
    fraction_cyp3a4: float = 0.0,
    fraction_renal: float = 0.0,
    min_dose_mg: float = 1.0,
) -> PediatricDosingResult:
    """Compute pediatric dose with allometric scaling and maturation adjustments."""
    if child_weight_kg is None:
        child_weight_kg = max(3.0, min(40.0, 3.0 + 2.0 * child_age_years))

    allometric = allometric_dose_scaling(adult_dose_mg, adult_weight_kg, child_weight_kg)

    cyp3a4_f = _interpolate_ontogeny(child_age_years, CYP3A4_ONTOGENY)
    gfr = _interpolate_ontogeny(child_age_years, GFR_ONTOGENY)

    adj_cyp = 1.0 - fraction_cyp3a4 * (1.0 - cyp3a4_f)
    adj_renal = 1.0 - fraction_renal * (1.0 - gfr / 130.0)

    cyp_dose = allometric * adj_cyp
    renal_dose = allometric * adj_renal
    combined = allometric * adj_cyp * adj_renal
    combined = max(combined, min_dose_mg)

    warnings: list[str] = []
    if child_age_years < 1.0:
        warnings.append("Neonate/infant — extra caution required")
    if combined < 0.1 * adult_dose_mg:
        warnings.append("Large dose reduction: verify with clinical pharmacist")

    return PediatricDosingResult(
        patient_age_years=child_age_years,
        patient_weight_kg=child_weight_kg,
        adult_dose_mg=adult_dose_mg,
        allometric_dose_mg=allometric,
        cyp3a4_adjusted_dose_mg=cyp_dose,
        renal_adjusted_dose_mg=renal_dose,
        combined_adjusted_dose_mg=combined,
        cyp3a4_fraction_adult=cyp3a4_f,
        gfr_estimated=gfr,
        dose_per_kg=combined / child_weight_kg,
        warnings=warnings,
    )


def pediatric_dose_range(
    adult_dose_mg: float,
    fraction_cyp3a4: float = 0.0,
    fraction_renal: float = 0.0,
) -> list[dict]:
    """Return dosing info across standard pediatric ages."""
    ages = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    results: list[dict] = []
    for age in ages:
        r = pediatric_dose(
            adult_dose_mg=adult_dose_mg,
            fraction_cyp3a4=fraction_cyp3a4,
            fraction_renal=fraction_renal,
            child_age_years=age,
        )
        results.append(
            {
                "age_years": age,
                "weight_kg": r.patient_weight_kg,
                "dose_mg": r.combined_adjusted_dose_mg,
                "dose_per_kg": r.dose_per_kg,
            }
        )
    return results


# ---------------------------------------------------------------------------
# New API (Phase 228)
# ---------------------------------------------------------------------------

# Maturation half-life for clearance (years) — simplified Hill model
# CL maturation: CL_mat = CL_adult * age^n / (TM50^n + age^n)
_CL_MAT_TM50 = 0.25  # half-mature age in years (approximately 3 months post-birth)
_CL_MAT_HILL = 3.4  # Hill exponent


def bsa_dubois(weight_kg: float, height_cm: float) -> float:
    """Compute body surface area using DuBois formula.

    BSA = 0.007184 * weight_kg^0.425 * height_cm^0.725  (m²)

    Parameters
    ----------
    weight_kg:
        Body weight in kilograms (must be > 0).
    height_cm:
        Height in centimetres (must be > 0).

    Returns
    -------
    float
        BSA in m².
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive.")
    if height_cm <= 0:
        raise ValueError("height_cm must be positive.")
    return 0.007184 * (weight_kg**0.425) * (height_cm**0.725)


@dataclass
class PediatricDoseResult:
    """Result from weight/BSA-based pediatric dose calculation."""

    drug_name: str
    method: str
    age_years: float | None
    weight_kg: float | None
    height_cm: float | None
    bsa_m2: float | None
    adult_dose_mg: float
    calculated_dose_mg: float
    adjusted_dose_mg: float
    dose_per_kg: float | None
    was_clamped: bool
    cautions: list[str]
    notes: str


def calculate_pediatric_dose(
    drug_name: str,
    adult_dose_mg: float,
    adult_bsa_m2: float = 1.73,
    weight_kg: float | None = None,
    height_cm: float | None = None,
    age_years: float | None = None,
    method: str = "bsa",
    max_dose_mg: float | None = None,
    min_dose_mg: float | None = None,
) -> PediatricDoseResult:
    """Calculate weight/BSA-based pediatric dose.

    Parameters
    ----------
    drug_name:
        Drug name (informational).
    adult_dose_mg:
        Adult dose in mg.
    adult_bsa_m2:
        Reference adult BSA in m² (default 1.73).
    weight_kg:
        Child weight in kg (required for bsa, weight, clark methods).
    height_cm:
        Child height in cm (required for bsa method).
    age_years:
        Child age in years (required for young method).
    method:
        One of "bsa", "weight", "young", "clark".
    max_dose_mg:
        Upper clamp on dose.
    min_dose_mg:
        Lower clamp on dose.

    Returns
    -------
    PediatricDoseResult
    """
    if adult_dose_mg <= 0:
        raise ValueError("adult_dose_mg must be positive.")

    valid_methods = {"bsa", "weight", "young", "clark"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}. Got: {method!r}")

    # Validate required parameters per method
    if method == "bsa":
        if weight_kg is None:
            raise ValueError("weight_kg is required for method='bsa'.")
        if height_cm is None:
            raise ValueError("height_cm is required for method='bsa'.")
        if weight_kg <= 0:
            raise ValueError("weight_kg must be positive.")
        if height_cm <= 0:
            raise ValueError("height_cm must be positive.")
    elif method == "weight":
        if weight_kg is None:
            raise ValueError("weight_kg is required for method='weight'.")
        if weight_kg <= 0:
            raise ValueError("weight_kg must be positive.")
    elif method == "young":
        if age_years is None:
            raise ValueError("age_years is required for method='young'.")
        if age_years <= 0:
            raise ValueError("age_years must be positive.")
    elif method == "clark":
        if weight_kg is None:
            raise ValueError("weight_kg is required for method='clark'.")
        if weight_kg <= 0:
            raise ValueError("weight_kg must be positive.")

    # Compute BSA if relevant
    bsa_m2: float | None = None
    if method == "bsa" and weight_kg is not None and height_cm is not None:
        bsa_m2 = bsa_dubois(weight_kg, height_cm)

    # Calculate dose
    if method == "bsa":
        calculated_dose = adult_dose_mg * (bsa_m2 / adult_bsa_m2)  # type: ignore[operator]
        notes = f"BSA method: {bsa_m2:.3f} m² / {adult_bsa_m2} m² (reference)"
    elif method == "weight":
        mg_per_kg_adult = adult_dose_mg / 70.0  # use 70 kg reference adult
        calculated_dose = mg_per_kg_adult * weight_kg  # type: ignore[operator]
        notes = f"Weight method: {mg_per_kg_adult:.3f} mg/kg × {weight_kg} kg"
    elif method == "young":
        calculated_dose = adult_dose_mg * age_years / (age_years + 12.0)  # type: ignore[operator]
        notes = f"Young's rule: age {age_years} / (age + 12)"
    else:  # clark
        calculated_dose = adult_dose_mg * weight_kg / 68.0  # type: ignore[operator]
        notes = f"Clark's rule: {weight_kg} kg / 68 kg"

    # Clamp
    adjusted_dose = calculated_dose
    was_clamped = False
    if max_dose_mg is not None and adjusted_dose > max_dose_mg:
        adjusted_dose = max_dose_mg
        was_clamped = True
    if min_dose_mg is not None and adjusted_dose < min_dose_mg:
        adjusted_dose = min_dose_mg
        was_clamped = True

    # Dose per kg
    dose_per_kg: float | None = None
    if weight_kg is not None and weight_kg > 0:
        dose_per_kg = adjusted_dose / weight_kg

    # Age-specific cautions
    cautions: list[str] = []
    if age_years is not None:
        if age_years < 2.0:
            cautions.append(
                "Very young patient (age < 2 years): use with extreme caution, "
                "verify dose with neonatal/infant specialist."
            )
        if age_years < 1.0:
            cautions.append(
                "Infant (age < 1 year): immature organ function, consider reduced dosing interval."
            )
        if age_years < 0.25:
            cautions.append("Neonate: highly variable PK; therapeutic drug monitoring recommended.")
    if weight_kg is not None and weight_kg < 5.0:
        cautions.append("Low body weight (< 5 kg): precise weight-based dosing critical.")

    return PediatricDoseResult(
        drug_name=drug_name,
        method=method,
        age_years=age_years,
        weight_kg=weight_kg,
        height_cm=height_cm,
        bsa_m2=bsa_m2,
        adult_dose_mg=adult_dose_mg,
        calculated_dose_mg=calculated_dose,
        adjusted_dose_mg=adjusted_dose,
        dose_per_kg=dose_per_kg,
        was_clamped=was_clamped,
        cautions=cautions,
        notes=notes,
    )


def dose_frequency_adjustment(
    age_years: float,
    adult_frequency_h: float,
    maturation_factor: float = 1.0,
) -> float:
    """Adjust dosing interval based on age-related clearance maturation.

    Younger patients have lower clearance maturation → longer dosing interval.

    Parameters
    ----------
    age_years:
        Patient age in years.
    adult_frequency_h:
        Adult dosing interval in hours (e.g. 8, 12, 24).
    maturation_factor:
        Additional scaling factor (default 1.0 = no extra adjustment).

    Returns
    -------
    float
        Adjusted dosing interval in hours. Younger patients → larger value.
    """
    if age_years <= 0:
        raise ValueError("age_years must be positive.")
    if adult_frequency_h <= 0:
        raise ValueError("adult_frequency_h must be positive.")

    # Clearance maturation fraction (Hill equation)
    # mat = age^hill / (TM50^hill + age^hill)
    tm50 = _CL_MAT_TM50
    hill = _CL_MAT_HILL
    mat = (age_years**hill) / (tm50**hill + age_years**hill)

    # Clip to [0.05, 1.0] to prevent unreasonably long intervals
    mat = max(0.05, min(1.0, mat))

    # Adjusted interval = adult_interval / maturation (lower CL → longer interval)
    adjusted = adult_frequency_h / (mat * maturation_factor)

    # Practical ceiling: no more than 5× adult interval
    adjusted = min(adjusted, adult_frequency_h * 5.0)

    return adjusted
