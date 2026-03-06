"""Pediatric dose scaling — allometric + maturation function."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PediatricDosingResult",
    "pediatric_dose",
    "pediatric_dose_range",
    "allometric_dose_scaling",
]

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
