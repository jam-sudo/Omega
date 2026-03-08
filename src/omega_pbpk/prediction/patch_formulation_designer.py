"""Phase 239: Transdermal patch formulation designer.

Designs transdermal patch formulations based on drug physicochemical properties
using the Potts-Guy skin permeability model.
"""

import math
from dataclasses import dataclass

# Patch type lookup table
_PATCH_TYPES = {
    "reservoir": {
        "lag_time_h": 2.0,
        "flux_modifier": 1.0,
        "membrane_controlled": True,
    },
    "matrix": {
        "lag_time_h": 1.0,
        "flux_modifier": 0.8,
        "membrane_controlled": False,
    },
    "drug_in_adhesive": {
        "lag_time_h": 0.5,
        "flux_modifier": 0.6,
        "membrane_controlled": False,
    },
    "microreservoir": {
        "lag_time_h": 1.5,
        "flux_modifier": 0.9,
        "membrane_controlled": True,
    },
}

VALID_PATCH_TYPES = list(_PATCH_TYPES.keys())

_MIN_AREA_CM2 = 1.0
_MAX_AREA_CM2 = 100.0
_UNIT_SOLUBILITY_MG_ML = 1.0  # Reference Cs


@dataclass(frozen=True)
class PatchDesignResult:
    """Result of transdermal patch design calculation."""

    drug_name: str
    patch_type: str
    patch_area_cm2: float
    permeability_cm_h: float
    flux_mg_cm2_h: float
    daily_delivery_mg: float
    lag_time_h: float
    steady_state_time_h: float
    c_plasma_steady_mg_L: float
    therapeutic_coverage: str
    notes: str


def _potts_guy_permeability(logp: float, mw: float) -> float:
    """Compute skin permeability via Potts-Guy equation (cm/h)."""
    log10_perm = 0.71 * logp - 0.0061 * mw - 6.3
    return 10.0**log10_perm


def design_patch(
    drug_name: str,
    target_daily_dose_mg: float,
    logp: float,
    mw: float,
    cl_L_per_h: float,
    patch_type: str = "matrix",
    vd_L: float = 50.0,
    therapeutic_range_mg_L: tuple[float, float] = (0.1, 2.0),
) -> PatchDesignResult:
    """Design a transdermal patch formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_daily_dose_mg:
        Desired daily drug delivery (mg/day).
    logp:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    cl_L_per_h:
        Systemic clearance (L/h).
    patch_type:
        One of 'reservoir', 'matrix', 'drug_in_adhesive', 'microreservoir'.
    vd_L:
        Volume of distribution (L), used for half-life calculation.
    therapeutic_range_mg_L:
        (low, high) therapeutic plasma concentration range (mg/L).

    Returns
    -------
    PatchDesignResult
    """
    # Validation
    if target_daily_dose_mg <= 0:
        raise ValueError("target_daily_dose_mg must be > 0")
    if not (-3.0 <= logp <= 8.0):
        raise ValueError("logp must be in range [-3, 8]")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if patch_type not in _PATCH_TYPES:
        raise ValueError(f"patch_type must be one of {VALID_PATCH_TYPES}; got '{patch_type}'")

    pt = _PATCH_TYPES[patch_type]
    lag_time_h: float = pt["lag_time_h"]
    flux_modifier: float = pt["flux_modifier"]

    # Skin permeability (Potts-Guy)
    permeability_cm_h = _potts_guy_permeability(logp, mw)

    # Flux: permeability * Cs * flux_modifier
    flux_mg_cm2_h = permeability_cm_h * flux_modifier * _UNIT_SOLUBILITY_MG_ML

    # Required patch area, clamped to [1, 100] cm2
    if flux_mg_cm2_h > 0:
        raw_area = target_daily_dose_mg / (flux_mg_cm2_h * 24.0)
    else:
        raw_area = _MAX_AREA_CM2

    patch_area_cm2 = max(_MIN_AREA_CM2, min(_MAX_AREA_CM2, raw_area))

    # Actual daily delivery at clamped area
    daily_delivery_mg = flux_mg_cm2_h * patch_area_cm2 * 24.0

    # Steady-state plasma concentration
    c_plasma_steady_mg_L = daily_delivery_mg / 24.0 / cl_L_per_h

    # Therapeutic coverage
    lo, hi = therapeutic_range_mg_L
    therapeutic_coverage = "adequate" if lo <= c_plasma_steady_mg_L <= hi else "inadequate"

    # Steady-state time: lag + terminal half-life (4 half-lives)
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half = math.log(2.0) / ke if ke > 0 else 0.0
    steady_state_time_h = 4.0 * lag_time_h + 4.0 * t_half

    # Notes
    area_clamped = raw_area < _MIN_AREA_CM2 or raw_area > _MAX_AREA_CM2
    notes_parts = []
    if area_clamped:
        notes_parts.append(f"Area clamped from {raw_area:.2f} to {patch_area_cm2:.2f} cm2")
    if pt["membrane_controlled"]:
        notes_parts.append("Membrane-controlled release")
    else:
        notes_parts.append("Diffusion-controlled release")
    if therapeutic_coverage == "inadequate":
        notes_parts.append(
            f"Predicted Css {c_plasma_steady_mg_L:.3f} mg/L outside "
            f"therapeutic range [{lo}, {hi}] mg/L"
        )
    notes = "; ".join(notes_parts) if notes_parts else "Design within specifications"

    return PatchDesignResult(
        drug_name=drug_name,
        patch_type=patch_type,
        patch_area_cm2=patch_area_cm2,
        permeability_cm_h=permeability_cm_h,
        flux_mg_cm2_h=flux_mg_cm2_h,
        daily_delivery_mg=daily_delivery_mg,
        lag_time_h=lag_time_h,
        steady_state_time_h=steady_state_time_h,
        c_plasma_steady_mg_L=c_plasma_steady_mg_L,
        therapeutic_coverage=therapeutic_coverage,
        notes=notes,
    )


def compare_patch_types(
    drug_name: str,
    target_daily_dose_mg: float,
    logp: float,
    mw: float,
    cl_L_per_h: float,
) -> list[PatchDesignResult]:
    """Compare all four patch types and return sorted by c_plasma_steady descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_daily_dose_mg:
        Target daily dose (mg/day).
    logp:
        LogP value.
    mw:
        Molecular weight.
    cl_L_per_h:
        Systemic clearance (L/h).

    Returns
    -------
    List of PatchDesignResult sorted by c_plasma_steady_mg_L descending.
    """
    results = [
        design_patch(
            drug_name=drug_name,
            target_daily_dose_mg=target_daily_dose_mg,
            logp=logp,
            mw=mw,
            cl_L_per_h=cl_L_per_h,
            patch_type=pt,
        )
        for pt in VALID_PATCH_TYPES
    ]
    results.sort(key=lambda r: r.c_plasma_steady_mg_L, reverse=True)
    return results
