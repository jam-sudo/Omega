"""PK covariate analysis — Phase 478."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CovariateSensitivityResult",
    "power_law_covariate",
    "analyze_weight_effect",
    "analyze_renal_effect",
    "analyze_age_effect",
    "covariate_sensitivity",
]


@dataclass(frozen=True)
class CovariateSensitivityResult:
    """Result of a covariate sensitivity analysis.

    Attributes
    ----------
    covariate_values : list[float]
        Covariate values evaluated.
    cl_values : list[float]
        Predicted CL at each covariate value (same units as cl_typical).
    cl_ratio_to_typical : list[float]
        CL_individual / CL_typical at each covariate value.
    max_fold_change : float
        Maximum ratio of highest CL to lowest CL across the range.
    dose_adjustment_needed : bool
        True when max_fold_change > 2.0 (significant PK variability).
    notes : str
        Summary notes.
    """

    covariate_values: tuple[float, ...]
    cl_values: tuple[float, ...]
    cl_ratio_to_typical: tuple[float, ...]
    max_fold_change: float
    dose_adjustment_needed: bool
    notes: str


def power_law_covariate(
    param_typical: float,
    covariate_value: float,
    covariate_typical: float,
    theta_power: float,
) -> float:
    """Scale a PK parameter using a power-law covariate model.

    P_individual = P_typical * (cov / cov_typical)^theta_power

    Parameters
    ----------
    param_typical : float
        Typical population value of the PK parameter.
    covariate_value : float
        Individual covariate value.
    covariate_typical : float
        Typical (reference) covariate value. Must be > 0.
    theta_power : float
        Power-law exponent.

    Returns
    -------
    float
        Individual PK parameter value.
    """
    if param_typical <= 0:
        raise ValueError("param_typical must be > 0")
    if covariate_value <= 0:
        raise ValueError("covariate_value must be > 0")
    if covariate_typical <= 0:
        raise ValueError("covariate_typical must be > 0")

    ratio = covariate_value / covariate_typical
    return param_typical * (ratio**theta_power)


def analyze_weight_effect(
    cl_typical: float,
    vd_typical: float,
    weight_kg: float,
    weight_typical: float = 70.0,
    cl_exponent: float = 0.75,
    vd_exponent: float = 1.0,
) -> dict:
    """Predict individual CL and Vd based on body weight allometric scaling.

    CL_individual = CL_typical * (weight / weight_typical)^cl_exponent
    Vd_individual = Vd_typical * (weight / weight_typical)^vd_exponent

    The default exponents follow allometric theory:
    - CL: 0.75 (metabolic rate scales with mass^0.75)
    - Vd: 1.0  (volume scales linearly with mass)

    Parameters
    ----------
    cl_typical : float
        Typical clearance (L/h). Must be > 0.
    vd_typical : float
        Typical volume of distribution (L). Must be > 0.
    weight_kg : float
        Individual body weight (kg). Must be > 0.
    weight_typical : float
        Reference population weight (kg). Default 70 kg.
    cl_exponent : float
        Allometric exponent for CL. Default 0.75.
    vd_exponent : float
        Allometric exponent for Vd. Default 1.0.

    Returns
    -------
    dict
        Keys: cl_individual, vd_individual, t_half_h, weight_ratio,
              cl_ratio, vd_ratio.
    """
    if cl_typical <= 0:
        raise ValueError("cl_typical must be > 0")
    if vd_typical <= 0:
        raise ValueError("vd_typical must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if weight_typical <= 0:
        raise ValueError("weight_typical must be > 0")

    cl_ind = power_law_covariate(cl_typical, weight_kg, weight_typical, cl_exponent)
    vd_ind = power_law_covariate(vd_typical, weight_kg, weight_typical, vd_exponent)
    t_half = (math.log(2) * vd_ind) / cl_ind
    weight_ratio = weight_kg / weight_typical

    return {
        "cl_individual": cl_ind,
        "vd_individual": vd_ind,
        "t_half_h": t_half,
        "weight_ratio": weight_ratio,
        "cl_ratio": cl_ind / cl_typical,
        "vd_ratio": vd_ind / vd_typical,
    }


def analyze_renal_effect(
    cl_typical: float,
    egfr_mL_per_min: float,
    egfr_normal: float = 100.0,
    cl_renal_fraction: float = 0.5,
) -> dict:
    """Predict individual CL adjusted for renal function.

    Partitions CL into renal and non-renal components:
        CL_renal = CL_typical * cl_renal_fraction
        CL_nonrenal = CL_typical * (1 - cl_renal_fraction)
        CL_individual = CL_nonrenal + CL_renal * (eGFR / eGFR_normal)

    Parameters
    ----------
    cl_typical : float
        Typical CL in a normal-renal-function subject (L/h). Must be > 0.
    egfr_mL_per_min : float
        Individual eGFR (mL/min). Must be >= 0.
    egfr_normal : float
        Normal eGFR reference value (mL/min). Default 100.
    cl_renal_fraction : float
        Fraction of CL attributable to renal elimination [0, 1].

    Returns
    -------
    dict
        Keys: cl_individual, cl_nonrenal, cl_renal_component,
              egfr_ratio, cl_ratio, renal_impairment_category.
    """
    if cl_typical <= 0:
        raise ValueError("cl_typical must be > 0")
    if egfr_mL_per_min < 0:
        raise ValueError("egfr_mL_per_min must be >= 0")
    if egfr_normal <= 0:
        raise ValueError("egfr_normal must be > 0")
    if not (0.0 <= cl_renal_fraction <= 1.0):
        raise ValueError("cl_renal_fraction must be between 0 and 1")

    cl_renal_comp = cl_typical * cl_renal_fraction
    cl_nonrenal = cl_typical * (1.0 - cl_renal_fraction)
    egfr_ratio = egfr_mL_per_min / egfr_normal
    cl_ind = cl_nonrenal + cl_renal_comp * egfr_ratio

    # Renal impairment categories (NKF staging via eGFR)
    if egfr_mL_per_min >= 90:
        category = "normal"
    elif egfr_mL_per_min >= 60:
        category = "mild"
    elif egfr_mL_per_min >= 30:
        category = "moderate"
    elif egfr_mL_per_min >= 15:
        category = "severe"
    else:
        category = "kidney_failure"

    return {
        "cl_individual": cl_ind,
        "cl_nonrenal": cl_nonrenal,
        "cl_renal_component": cl_renal_comp * egfr_ratio,
        "egfr_ratio": egfr_ratio,
        "cl_ratio": cl_ind / cl_typical,
        "renal_impairment_category": category,
    }


def analyze_age_effect(
    cl_typical: float,
    vd_typical: float,
    age_years: float,
    age_typical: float = 35.0,
) -> dict:
    """Predict individual CL and Vd adjusted for age.

    Age scaling rules:
    - For age <= age_typical: CL and Vd are flat (no change).
    - For age > age_typical: CL = CL_typical * (age / age_typical)^(-0.2)
                              Vd = Vd_typical * (age / age_typical)^(-0.1)

    The negative exponents reflect the progressive decline in hepatic blood flow,
    renal function, and body composition with increasing age.

    Parameters
    ----------
    cl_typical : float
        Typical CL at the reference age (L/h). Must be > 0.
    vd_typical : float
        Typical Vd at the reference age (L). Must be > 0.
    age_years : float
        Individual age (years). Must be > 0.
    age_typical : float
        Reference age (years). Default 35 years.

    Returns
    -------
    dict
        Keys: cl_individual, vd_individual, t_half_h, age_ratio,
              cl_ratio, vd_ratio, age_category.
    """
    if cl_typical <= 0:
        raise ValueError("cl_typical must be > 0")
    if vd_typical <= 0:
        raise ValueError("vd_typical must be > 0")
    if age_years <= 0:
        raise ValueError("age_years must be > 0")
    if age_typical <= 0:
        raise ValueError("age_typical must be > 0")

    age_ratio = age_years / age_typical

    if age_years <= age_typical:
        cl_ind = cl_typical
        vd_ind = vd_typical
    else:
        cl_ind = cl_typical * (age_ratio ** (-0.2))
        vd_ind = vd_typical * (age_ratio ** (-0.1))

    t_half = (math.log(2) * vd_ind) / cl_ind

    if age_years < 18:
        age_category = "pediatric"
    elif age_years < 65:
        age_category = "adult"
    else:
        age_category = "elderly"

    return {
        "cl_individual": cl_ind,
        "vd_individual": vd_ind,
        "t_half_h": t_half,
        "age_ratio": age_ratio,
        "cl_ratio": cl_ind / cl_typical,
        "vd_ratio": vd_ind / vd_typical,
        "age_category": age_category,
    }


def covariate_sensitivity(
    cl_typical: float,
    covariate_range: list[float],
    covariate_typical: float = 1.0,
    theta_power: float = 0.75,
) -> CovariateSensitivityResult:
    """Evaluate CL sensitivity across a covariate range using the power-law model.

    Parameters
    ----------
    cl_typical : float
        Typical CL value (L/h). Must be > 0.
    covariate_range : list[float]
        Covariate values to evaluate. All must be > 0. Must contain >= 2 values.
    covariate_typical : float
        Typical (reference) covariate value. Must be > 0.
    theta_power : float
        Power-law exponent for scaling.

    Returns
    -------
    CovariateSensitivityResult
        Sensitivity result dataclass.
    """
    if cl_typical <= 0:
        raise ValueError("cl_typical must be > 0")
    if covariate_typical <= 0:
        raise ValueError("covariate_typical must be > 0")
    if not covariate_range:
        raise ValueError("covariate_range must not be empty")
    if any(v <= 0 for v in covariate_range):
        raise ValueError("all values in covariate_range must be > 0")
    if len(covariate_range) < 2:
        raise ValueError("covariate_range must contain at least 2 values")

    cl_values = [
        power_law_covariate(cl_typical, v, covariate_typical, theta_power) for v in covariate_range
    ]
    cl_ratio = [cl / cl_typical for cl in cl_values]

    max_cl = max(cl_values)
    min_cl = min(cl_values)
    max_fold_change = max_cl / min_cl if min_cl > 0 else float("inf")
    dose_adjustment_needed = max_fold_change > 2.0

    notes = (
        f"Power exponent={theta_power:.2f}; "
        f"CL range [{min_cl:.3f}, {max_cl:.3f}]; "
        f"max fold-change={max_fold_change:.2f}. "
    )
    if dose_adjustment_needed:
        notes += "Dose adjustment likely needed (>2-fold CL variability)."
    else:
        notes += "Dose adjustment not required (<2-fold CL variability)."

    return CovariateSensitivityResult(
        covariate_values=tuple(covariate_range),
        cl_values=tuple(cl_values),
        cl_ratio_to_typical=tuple(cl_ratio),
        max_fold_change=max_fold_change,
        dose_adjustment_needed=dose_adjustment_needed,
        notes=notes,
    )
