"""
Neonatal PK Scaling Module (Phase 882)

Scales adult drug PK parameters to neonates (0-28 days) and infants
(1-12 months), accounting for immature renal function, CYP enzymes,
higher body water fraction, and lower protein binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NeonatalPKResult:
    """Result of neonatal PK scaling."""

    drug_name: str
    age_days: int
    weight_kg: float
    adult_cl_L_per_h: float
    adult_vd_L: float
    neonatal_cl_L_per_h: float
    neonatal_vd_L: float
    cl_scaling_factor: float
    vd_scaling_factor: float
    predicted_t_half_h: float
    dose_adjustment_factor: float
    gfr_fraction_adult: float
    cyp3a4_maturation_fraction: float
    protein_binding_adjustment: float
    notes: str


def scale_neonatal_pk(
    drug_name: str,
    age_days: int,
    weight_kg: float,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    primary_elimination: str = "renal",
    cyp_enzyme: str = "CYP3A4",
) -> NeonatalPKResult:
    """
    Scale adult PK parameters to neonatal/infant PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    age_days : int
        Age in days (0-365).
    weight_kg : float
        Body weight in kg.
    adult_cl_L_per_h : float
        Adult clearance in L/h.
    adult_vd_L : float
        Adult volume of distribution in L.
    primary_elimination : str
        "renal", "hepatic", or other (uses average).
    cyp_enzyme : str
        CYP enzyme responsible (informational).

    Returns
    -------
    NeonatalPKResult
    """
    if not (0 <= age_days <= 365):
        raise ValueError("age_days must be between 0 and 365")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")

    # GFR maturation
    if age_days <= 28:
        gfr_fraction = 0.04 + 0.96 / (1.0 + math.exp(-(age_days - 14) / 7.0))
    else:
        gfr_fraction = 0.3 + 0.7 * (age_days / 365.0) ** 0.5
    gfr_fraction = max(0.02, min(1.0, gfr_fraction))

    # CYP3A4 maturation
    cyp3a4_frac = 0.05 + 0.95 / (1.0 + math.exp(-(age_days - 30) / 20.0))
    cyp3a4_frac = max(0.05, min(1.0, cyp3a4_frac))

    # Vd scaling: higher body water fraction in neonates
    vd_scaling = 1.0 + (0.15 * max(0.0, 30.0 - age_days) / 30.0)

    # CL scaling
    weight_exponent = (weight_kg / 70.0) ** 0.75
    if primary_elimination == "renal":
        cl_scale = gfr_fraction * weight_exponent
    elif primary_elimination == "hepatic":
        cl_scale = cyp3a4_frac * weight_exponent
    else:
        cl_scale = 0.5 * (gfr_fraction + cyp3a4_frac) * weight_exponent

    # Protein binding adjustment
    if age_days < 60:
        protein_binding_adjustment = 0.7 + 0.3 * (age_days / 60.0)
    else:
        protein_binding_adjustment = 1.0

    # Neonatal PK parameters
    neonatal_cl = adult_cl_L_per_h * cl_scale
    neonatal_vd = adult_vd_L * vd_scaling * (weight_kg / 70.0)

    # Half-life
    if neonatal_cl > 0:
        predicted_t_half_h = 0.693 * neonatal_vd / neonatal_cl
    else:
        predicted_t_half_h = float("inf")

    # Dose adjustment factor
    dose_adjustment_factor = cl_scale * (weight_kg / 70.0)

    # Notes
    if age_days < 7:
        notes = "Extreme caution: very premature/early neonatal PK — highly unpredictable"
    elif age_days < 28:
        notes = "Neonatal dosing: use weight-based dosing with TDM"
    else:
        notes = "Infant dosing: standard weight-based dosing applicable"

    return NeonatalPKResult(
        drug_name=drug_name,
        age_days=age_days,
        weight_kg=weight_kg,
        adult_cl_L_per_h=adult_cl_L_per_h,
        adult_vd_L=adult_vd_L,
        neonatal_cl_L_per_h=neonatal_cl,
        neonatal_vd_L=neonatal_vd,
        cl_scaling_factor=cl_scale,
        vd_scaling_factor=vd_scaling,
        predicted_t_half_h=predicted_t_half_h,
        dose_adjustment_factor=dose_adjustment_factor,
        gfr_fraction_adult=gfr_fraction,
        cyp3a4_maturation_fraction=cyp3a4_frac,
        protein_binding_adjustment=protein_binding_adjustment,
        notes=notes,
    )


def compare_age_groups(
    drug_name: str,
    weight_kg: float,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    ages_days: list[int] | None = None,
) -> list[NeonatalPKResult]:
    """
    Compare PK across neonatal/infant age groups.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    weight_kg : float
        Body weight in kg.
    adult_cl_L_per_h : float
        Adult clearance in L/h.
    adult_vd_L : float
        Adult volume of distribution in L.
    ages_days : list[int] | None
        Ages in days to evaluate. Defaults to [1, 7, 14, 28, 90, 180, 270, 365].

    Returns
    -------
    list[NeonatalPKResult]
        Sorted by age_days ascending.
    """
    if ages_days is None:
        ages_days = [1, 7, 14, 28, 90, 180, 270, 365]

    results = []
    for age in ages_days:
        result = scale_neonatal_pk(
            drug_name=drug_name,
            age_days=age,
            weight_kg=weight_kg,
            adult_cl_L_per_h=adult_cl_L_per_h,
            adult_vd_L=adult_vd_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.age_days)
    return results
