"""Pediatric PBPK parameter scaling module."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PediatricScalingResult",
    "scale_to_pediatric",
]


@dataclass
class PediatricScalingResult:
    drug_name: str
    age_years: float
    weight_kg: float
    adult_cl_L_per_h: float
    adult_vd_L: float
    pediatric_cl_L_per_h: float
    pediatric_vd_L: float
    pediatric_t_half_h: float
    pediatric_dose_mg: float
    cyp3a4_maturation_fraction: float
    renal_maturation_fraction: float
    scaling_method: str


def _cyp3a4_maturation(age_years: float) -> float:
    """Sigmoid maturation function for CYP3A4, clamped to [0.05, 1.0]."""
    f = 1.0 / (1.0 + math.exp(-1.5 * (age_years - 1.5)))
    return max(0.05, min(1.0, f))


def _gfr_maturation(age_years: float) -> float:
    """GFR maturation based on postmenstrual age, clamped to [0.03, 1.0]."""
    postmenstrual_age = age_years * 52 + 40
    ratio = postmenstrual_age / 40.0
    ratio_pow = ratio**3.4
    gfr_fraction = ratio_pow / (ratio_pow + 1.0)
    return max(0.03, min(1.0, gfr_fraction))


def _allometric_cl(adult_cl: float, child_weight: float, adult_weight: float = 70.0) -> float:
    """Allometric scaling of clearance with exponent 0.75."""
    return adult_cl * (child_weight / adult_weight) ** 0.75


def _allometric_vd(adult_vd: float, child_weight: float, adult_weight: float = 70.0) -> float:
    """Allometric scaling of volume of distribution with exponent 1.0."""
    return adult_vd * (child_weight / adult_weight) ** 1.0


def scale_to_pediatric(
    drug_name: str,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    adult_dose_mg: float,
    age_years: float,
    weight_kg: float | None = None,
    fraction_cyp3a4: float = 0.5,
    fraction_renal: float = 0.3,
    method: str = "allometric_mat",
) -> PediatricScalingResult:
    """Scale adult PBPK parameters to pediatric population.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    adult_cl_L_per_h:
        Adult clearance in L/h.
    adult_vd_L:
        Adult volume of distribution in L.
    adult_dose_mg:
        Adult dose in mg (reference for same exposure calculation).
    age_years:
        Age of the pediatric patient in years.
    weight_kg:
        Weight in kg; if None, estimated from age.
    fraction_cyp3a4:
        Fraction of clearance mediated by CYP3A4 (0-1).
    fraction_renal:
        Fraction of clearance mediated by renal elimination (0-1).
    method:
        Scaling method label.

    Returns
    -------
    PediatricScalingResult

    Raises
    ------
    ValueError
        If adult_cl_L_per_h <= 0, adult_vd_L <= 0, adult_dose_mg <= 0,
        or age_years < 0.
    """
    if adult_cl_L_per_h <= 0:
        raise ValueError(f"adult_cl_L_per_h must be > 0, got {adult_cl_L_per_h}")
    if adult_vd_L <= 0:
        raise ValueError(f"adult_vd_L must be > 0, got {adult_vd_L}")
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be > 0, got {adult_dose_mg}")
    if age_years < 0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")

    if weight_kg is None:
        weight_kg = max(3.0, min(80.0, 3.0 + 2.0 * age_years))

    cyp3a4_f = _cyp3a4_maturation(age_years)
    gfr_f = _gfr_maturation(age_years)

    allometric_cl = _allometric_cl(adult_cl_L_per_h, weight_kg)
    cyp3a4_factor = (1.0 - fraction_cyp3a4) + fraction_cyp3a4 * cyp3a4_f
    renal_factor = (1.0 - fraction_renal) + fraction_renal * gfr_f
    pediatric_cl = allometric_cl * cyp3a4_factor * renal_factor

    pediatric_vd = _allometric_vd(adult_vd_L, weight_kg)

    pediatric_t_half = 0.693 * pediatric_vd / pediatric_cl

    pediatric_dose = adult_dose_mg * pediatric_cl / adult_cl_L_per_h

    return PediatricScalingResult(
        drug_name=drug_name,
        age_years=age_years,
        weight_kg=weight_kg,
        adult_cl_L_per_h=adult_cl_L_per_h,
        adult_vd_L=adult_vd_L,
        pediatric_cl_L_per_h=pediatric_cl,
        pediatric_vd_L=pediatric_vd,
        pediatric_t_half_h=pediatric_t_half,
        pediatric_dose_mg=pediatric_dose,
        cyp3a4_maturation_fraction=cyp3a4_f,
        renal_maturation_fraction=gfr_f,
        scaling_method=method,
    )
