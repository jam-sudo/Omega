"""Pediatric dose finding via allometric scaling and maturation functions.

Calculates pediatric doses from adult reference data using weight-based
allometric scaling with age-dependent maturation corrections for hepatic
(CYP) and renal (GFR) elimination pathways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PediatricDoseResult",
    "calculate_pediatric_dose",
    "dose_across_ages",
]

_VALID_ELIMINATION = ("hepatic", "renal", "mixed")


@dataclass(frozen=True)
class PediatricDoseResult:
    """Result of pediatric dose calculation.

    Attributes
    ----------
    drug_name : Drug identifier.
    child_age_years : Child age in years.
    child_weight_kg : Child body weight in kg.
    adult_dose_mg : Reference adult dose (mg).
    pediatric_dose_mg : Calculated pediatric dose (mg).
    cl_adult_L_per_h : Adult clearance (L/h).
    cl_child_L_per_h : Pediatric clearance (L/h).
    vd_adult_L : Adult volume of distribution (L).
    vd_child_L : Pediatric volume of distribution (L).
    dose_per_kg_mg_kg : Pediatric dose per kg body weight (mg/kg).
    t_half_child_h : Estimated pediatric half-life (h).
    maturation_factor : Maturation factor applied to clearance.
    notes : Additional information.
    """

    drug_name: str
    child_age_years: float
    child_weight_kg: float
    adult_dose_mg: float
    pediatric_dose_mg: float
    cl_adult_L_per_h: float
    cl_child_L_per_h: float
    vd_adult_L: float
    vd_child_L: float
    dose_per_kg_mg_kg: float
    t_half_child_h: float
    maturation_factor: float
    notes: str


def _hepatic_maturation(age_years: float) -> float:
    """CYP maturation factor: age / (age + 0.5) for ages < 2, 1.0 otherwise."""
    if age_years < 2.0:
        return age_years / (age_years + 0.5)
    return 1.0


def _renal_maturation(age_years: float) -> float:
    """GFR maturation: 1 - exp(-0.5 * age)."""
    return 1.0 - math.exp(-0.5 * age_years)


def calculate_pediatric_dose(
    drug_name: str,
    adult_dose_mg: float,
    child_age_years: float,
    child_weight_kg: float,
    adult_weight_kg: float = 70.0,
    child_height_cm: float = 100.0,
    allometric_exponent: float = 0.75,
    elimination: str = "hepatic",
    cl_adult_L_per_h: float = 10.0,
    vd_adult_L: float = 70.0,
) -> PediatricDoseResult:
    """Calculate pediatric dose from adult reference using allometric scaling.

    Parameters
    ----------
    drug_name : Drug identifier.
    adult_dose_mg : Reference adult dose (mg). Must be > 0.
    child_age_years : Child age in years. Must be >= 0.
    child_weight_kg : Child body weight (kg). Must be > 0.
    adult_weight_kg : Adult reference weight (kg, default 70).
    child_height_cm : Child height (cm, default 100).
    allometric_exponent : Exponent for CL scaling (default 0.75).
    elimination : Elimination route: 'hepatic', 'renal', or 'mixed'.
    cl_adult_L_per_h : Adult clearance (L/h, default 10).
    vd_adult_L : Adult Vd (L, default 70).

    Returns
    -------
    PediatricDoseResult
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be > 0, got {adult_dose_mg}")
    if child_age_years < 0:
        raise ValueError(f"child_age_years must be >= 0, got {child_age_years}")
    if child_weight_kg <= 0:
        raise ValueError(f"child_weight_kg must be > 0, got {child_weight_kg}")
    if adult_weight_kg <= 0:
        raise ValueError(f"adult_weight_kg must be > 0, got {adult_weight_kg}")
    if cl_adult_L_per_h <= 0:
        raise ValueError(f"cl_adult_L_per_h must be > 0, got {cl_adult_L_per_h}")
    if vd_adult_L <= 0:
        raise ValueError(f"vd_adult_L must be > 0, got {vd_adult_L}")
    if elimination not in _VALID_ELIMINATION:
        raise ValueError(f"elimination must be one of {_VALID_ELIMINATION}, got '{elimination}'")

    bw_ratio = child_weight_kg / adult_weight_kg

    # Allometric CL scaling
    cl_child = cl_adult_L_per_h * (bw_ratio**allometric_exponent)

    # Maturation factor
    notes_parts: list[str] = []
    if elimination == "hepatic":
        mat_factor = _hepatic_maturation(child_age_years)
        cl_child *= mat_factor
        if child_age_years < 2.0:
            notes_parts.append("CYP maturation applied (age < 2y)")
    elif elimination == "renal":
        mat_factor = _renal_maturation(child_age_years)
        cl_child *= mat_factor
        notes_parts.append("GFR maturation applied")
    else:  # mixed
        hep_mat = _hepatic_maturation(child_age_years)
        ren_mat = _renal_maturation(child_age_years)
        mat_factor = 0.5 * (hep_mat + ren_mat)
        cl_child *= mat_factor
        notes_parts.append("Mixed hepatic/renal maturation applied")

    # Allometric Vd scaling (exponent = 1.0)
    vd_child = vd_adult_L * (bw_ratio**1.0)

    # Dose proportional to CL ratio
    ped_dose = adult_dose_mg * (cl_child / cl_adult_L_per_h)

    dose_per_kg = ped_dose / child_weight_kg

    # Half-life
    t_half = 0.693 * vd_child / cl_child if cl_child > 0 else float("inf")

    if not notes_parts:
        notes_parts.append("Standard allometric scaling")

    return PediatricDoseResult(
        drug_name=drug_name,
        child_age_years=child_age_years,
        child_weight_kg=child_weight_kg,
        adult_dose_mg=adult_dose_mg,
        pediatric_dose_mg=ped_dose,
        cl_adult_L_per_h=cl_adult_L_per_h,
        cl_child_L_per_h=cl_child,
        vd_adult_L=vd_adult_L,
        vd_child_L=vd_child,
        dose_per_kg_mg_kg=dose_per_kg,
        t_half_child_h=t_half,
        maturation_factor=mat_factor,
        notes="; ".join(notes_parts),
    )


def dose_across_ages(
    drug_name: str,
    adult_dose_mg: float,
    ages_years: list[float],
    child_weights_kg: list[float] | None = None,
    **kwargs,
) -> list[PediatricDoseResult]:
    """Calculate pediatric doses across a range of ages.

    Parameters
    ----------
    drug_name : Drug identifier.
    adult_dose_mg : Reference adult dose (mg).
    ages_years : List of child ages.
    child_weights_kg : Optional list of weights corresponding to ages.
        If not provided, estimated as 2.5 * age + 8 (rough approximation).
    **kwargs : Additional keyword arguments passed to ``calculate_pediatric_dose``.

    Returns
    -------
    list[PediatricDoseResult]
    """
    if not ages_years:
        raise ValueError("ages_years must not be empty")

    if child_weights_kg is not None:
        if len(child_weights_kg) != len(ages_years):
            raise ValueError(
                f"child_weights_kg length ({len(child_weights_kg)}) "
                f"must match ages_years length ({len(ages_years)})"
            )
        weights = child_weights_kg
    else:
        # Simple weight estimation: 2.5 * age + 8 (reasonable for 1-12y)
        weights = [max(3.5, 2.5 * age + 8.0) for age in ages_years]

    results = []
    for age, weight in zip(ages_years, weights, strict=True):
        r = calculate_pediatric_dose(
            drug_name=drug_name,
            adult_dose_mg=adult_dose_mg,
            child_age_years=age,
            child_weight_kg=weight,
            **kwargs,
        )
        results.append(r)
    return results
