"""Neonatal/Pediatric Physiological Parameters (Phase 1104).

Age-dependent physiological parameters for neonates, infants, and young children
(0-12 years), complementing the existing pediatric ontogeny module.

References: Johnson TN et al. (2006), Rhodin MM et al. (2009),
Edginton AN et al. (2006).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Age group definitions
# ---------------------------------------------------------------------------

_VALID_AGE_GROUPS = {
    "neonate",
    "infant_1_6mo",
    "infant_6_12mo",
    "toddler_1_3y",
    "preschool_3_6y",
    "school_6_12y",
}

# Body weight (kg) per age group
_BODY_WEIGHT_KG: dict[str, float] = {
    "neonate": 3.5,
    "infant_1_6mo": 6.0,
    "infant_6_12mo": 9.0,
    "toddler_1_3y": 13.0,
    "preschool_3_6y": 20.0,
    "school_6_12y": 32.0,
}

# Liver volume as fraction of body weight
_V_LIVER_FRACTION: dict[str, float] = {
    "neonate": 0.040,
    "infant_1_6mo": 0.040,
    "infant_6_12mo": 0.038,
    "toddler_1_3y": 0.035,
    "preschool_3_6y": 0.033,
    "school_6_12y": 0.030,
}

# Kidney volume as fraction of body weight
_V_KIDNEY_FRACTION: dict[str, float] = {
    "neonate": 0.008,
    "infant_1_6mo": 0.007,
    "infant_6_12mo": 0.007,
    "toddler_1_3y": 0.006,
    "preschool_3_6y": 0.006,
    "school_6_12y": 0.005,
}

# Cardiac output (L/h/kg)
_CARDIAC_OUTPUT_PER_KG: dict[str, float] = {
    "neonate": 0.60,
    "infant_1_6mo": 0.75,
    "infant_6_12mo": 0.70,
    "toddler_1_3y": 0.60,
    "preschool_3_6y": 0.52,
    "school_6_12y": 0.48,
}

# Blood flow fractions of cardiac output (same for all ages)
_Q_LIVER_FRACTION = 0.25
_Q_KIDNEY_FRACTION = 0.20

# GFR (mL/min/1.73 m²) — standardised values
_GFR_NORMALISED: dict[str, float] = {
    "neonate": 20.0,
    "infant_1_6mo": 50.0,
    "infant_6_12mo": 80.0,
    "toddler_1_3y": 100.0,
    "preschool_3_6y": 110.0,
    "school_6_12y": 120.0,
}

# CYP3A4 activity (% of adult)
_CYP3A4_FRACTION: dict[str, float] = {
    "neonate": 0.10,
    "infant_1_6mo": 0.30,
    "infant_6_12mo": 0.50,
    "toddler_1_3y": 0.70,
    "preschool_3_6y": 0.90,
    "school_6_12y": 1.00,
}

# UGT1A1 activity (% of adult)
_UGT1A1_FRACTION: dict[str, float] = {
    "neonate": 0.01,
    "infant_1_6mo": 0.20,
    "infant_6_12mo": 0.40,
    "toddler_1_3y": 0.60,
    "preschool_3_6y": 0.85,
    "school_6_12y": 1.00,
}

# Adult reference: BW=70 kg, BSA=1.73 m²
_ADULT_BW_KG = 70.0
_ADULT_BSA_M2 = 1.73


def _bsa_m2(bw_kg: float) -> float:
    """Mosteller formula: BSA (m²) = sqrt(BW_kg * 60 / 3600)."""
    # Simplified: BSA ≈ (BW/70)^0.5 * 1.73  — use Mosteller: sqrt(H*W/3600)
    # Without height, use power law approximation relative to adult
    return _ADULT_BSA_M2 * (bw_kg / _ADULT_BW_KG) ** 0.5


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeonatalParameters:
    """Physiological parameters for a pediatric age group."""

    age_group: str
    body_weight_kg: float
    v_liver_L: float  # absolute liver volume (L)
    v_kidney_L: float  # absolute kidney volume (L)
    cardiac_output_L_per_h: float  # total cardiac output (L/h)
    q_liver_L_per_h: float  # hepatic blood flow (L/h)
    q_kidney_L_per_h: float  # renal blood flow (L/h)
    gfr_mL_per_min: float  # actual GFR scaled to individual BSA
    cyp3a4_activity_fraction: float  # relative to adult [0, 1]
    ugt1a1_activity_fraction: float  # relative to adult [0, 1]
    adult_cl_scaling_factor: float  # geometric mean of GFR_frac and CYP3A4_frac
    allometric_dose_factor: float  # (BW/70)^0.75
    notes: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def get_neonatal_parameters(age_group: str) -> NeonatalParameters:
    """Return physiological parameters for a pediatric age group.

    Parameters
    ----------
    age_group:
        One of: "neonate", "infant_1_6mo", "infant_6_12mo",
        "toddler_1_3y", "preschool_3_6y", "school_6_12y".

    Returns
    -------
    NeonatalParameters dataclass with organ volumes, blood flows,
    enzyme activities, and scaling factors.

    Raises
    ------
    ValueError
        If age_group is not recognised.
    """
    if age_group not in _VALID_AGE_GROUPS:
        raise ValueError(f"age_group must be one of {sorted(_VALID_AGE_GROUPS)}, got '{age_group}'")

    bw = _BODY_WEIGHT_KG[age_group]
    v_liver = _V_LIVER_FRACTION[age_group] * bw
    v_kidney = _V_KIDNEY_FRACTION[age_group] * bw

    co_per_kg = _CARDIAC_OUTPUT_PER_KG[age_group]
    co_total = co_per_kg * bw
    q_liver = _Q_LIVER_FRACTION * co_total
    q_kidney = _Q_KIDNEY_FRACTION * co_total

    # Scale GFR from normalised (per 1.73 m²) to individual
    bsa = _bsa_m2(bw)
    gfr_norm = _GFR_NORMALISED[age_group]
    gfr_actual = gfr_norm * bsa / _ADULT_BSA_M2  # mL/min

    cyp_frac = _CYP3A4_FRACTION[age_group]
    ugt_frac = _UGT1A1_FRACTION[age_group]

    # Adult CL scaling: geometric mean of GFR fraction and CYP3A4 fraction
    gfr_frac = gfr_norm / 125.0  # adult GFR reference 125 mL/min/1.73 m²
    cl_scale = math.sqrt(gfr_frac * cyp_frac)

    # Allometric dose factor
    allometric = (bw / _ADULT_BW_KG) ** 0.75

    notes_parts: list[str] = []
    if cyp_frac < 0.5:
        notes_parts.append("CYP3A4 immature; dose reduction may be needed.")
    if ugt_frac < 0.1:
        notes_parts.append("Very low UGT1A1; risk of unconjugated bilirubin accumulation.")
    if gfr_frac < 0.5:
        notes_parts.append("Reduced GFR; renally cleared drugs require adjustment.")
    if not notes_parts:
        notes_parts.append("Enzyme activities approaching adult levels.")

    return NeonatalParameters(
        age_group=age_group,
        body_weight_kg=bw,
        v_liver_L=v_liver,
        v_kidney_L=v_kidney,
        cardiac_output_L_per_h=co_total,
        q_liver_L_per_h=q_liver,
        q_kidney_L_per_h=q_kidney,
        gfr_mL_per_min=gfr_actual,
        cyp3a4_activity_fraction=cyp_frac,
        ugt1a1_activity_fraction=ugt_frac,
        adult_cl_scaling_factor=cl_scale,
        allometric_dose_factor=allometric,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Dose calculation
# ---------------------------------------------------------------------------


def calculate_pediatric_dose(
    adult_dose_mg: float,
    age_group: str,
    scaling_method: str = "allometric",
) -> dict[str, Any]:
    """Calculate pediatric dose from adult dose using the specified scaling method.

    Parameters
    ----------
    adult_dose_mg:
        Adult reference dose in mg.
    age_group:
        Pediatric age group string.
    scaling_method:
        One of "allometric" (BW^0.75), "mg_per_kg" (linear BW/70),
        "organ_function" (CL-based adult_cl_scaling_factor).

    Returns
    -------
    dict with keys: pediatric_dose_mg, scaling_factor, method, age_group.
    """
    valid_methods = {"allometric", "mg_per_kg", "organ_function"}
    if scaling_method not in valid_methods:
        raise ValueError(f"scaling_method must be one of {valid_methods}, got '{scaling_method}'")

    params = get_neonatal_parameters(age_group)

    if scaling_method == "allometric":
        factor = params.allometric_dose_factor
    elif scaling_method == "mg_per_kg":
        factor = params.body_weight_kg / _ADULT_BW_KG
    else:  # organ_function
        factor = params.adult_cl_scaling_factor

    pediatric_dose = adult_dose_mg * factor

    return {
        "pediatric_dose_mg": round(pediatric_dose, 4),
        "scaling_factor": round(factor, 6),
        "method": scaling_method,
        "age_group": age_group,
    }


# ---------------------------------------------------------------------------
# Compare all age groups
# ---------------------------------------------------------------------------


def compare_age_groups() -> list[NeonatalParameters]:
    """Return parameters for all 6 age groups, sorted by body_weight_kg ascending.

    Returns
    -------
    List of 6 NeonatalParameters objects sorted lightest to heaviest.
    """
    groups = [get_neonatal_parameters(ag) for ag in _VALID_AGE_GROUPS]
    groups.sort(key=lambda p: p.body_weight_kg)
    return groups
