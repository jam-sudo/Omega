"""
Phase 946 — Drug Plasma Stability PK
Predict drug stability in plasma and its effect on PK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["PlasmaStabilityPKResult", "predict_plasma_stability_pk", "screen_plasma_stability"]

# Structural alert half-lives in minutes
_T_HALF_ESTER_MIN = 60.0
_T_HALF_CARBONATE_MIN = 30.0
_T_HALF_LACTONE_MIN = 120.0
_T_HALF_AMIDE_MIN = 480.0
_T_HALF_STABLE_MIN = 10000.0

_MIN_PER_HOUR = 60.0


@dataclass(frozen=True)
class PlasmaStabilityPKResult:
    drug_name: str
    logp: float
    mw: float
    has_ester: bool
    has_carbonate: bool
    has_lactone: bool
    has_amide: bool
    t_half_plasma_stability_min: float
    stability_category: str
    k_plasma_deg_per_h: float
    cl_plasma_deg_L_per_h: float
    fraction_degraded_vs_metabolized: float
    predicted_in_vivo_cl_L_per_h: float
    t_half_in_vivo_h: float
    notes: str


def _structural_t_half_min(
    has_ester: bool,
    has_carbonate: bool,
    has_lactone: bool,
    has_amide: bool,
) -> float:
    """Return shortest predicted half-life from structural alerts (minutes)."""
    candidates = []
    if has_carbonate:
        candidates.append(_T_HALF_CARBONATE_MIN)
    if has_ester:
        candidates.append(_T_HALF_ESTER_MIN)
    if has_lactone:
        candidates.append(_T_HALF_LACTONE_MIN)
    if has_amide:
        candidates.append(_T_HALF_AMIDE_MIN)

    if candidates:
        return min(candidates)
    return _T_HALF_STABLE_MIN


def _stability_category(t_half_min: float) -> str:
    if t_half_min < 60.0:
        return "unstable"
    elif t_half_min < 240.0:
        return "labile"
    else:
        return "stable"


def predict_plasma_stability_pk(
    drug_name: str,
    logp: float,
    mw: float,
    cl_hepatic_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    cl_renal_L_per_h: float = 0.0,
    has_ester: bool = False,
    has_carbonate: bool = False,
    has_lactone: bool = False,
    has_amide: bool = False,
    experimental_t_half_min: float | None = None,
) -> PlasmaStabilityPKResult:
    """Predict plasma stability and PK impact from structural features or experimental data."""
    # Validation
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if experimental_t_half_min is not None and experimental_t_half_min <= 0:
        raise ValueError("experimental_t_half_min must be > 0")

    # Determine plasma stability half-life
    if experimental_t_half_min is not None:
        t_half_min = experimental_t_half_min
        source = "experimental"
    else:
        t_half_min = _structural_t_half_min(has_ester, has_carbonate, has_lactone, has_amide)
        source = "structural prediction"

    category = _stability_category(t_half_min)

    # Convert half-life to degradation rate constant (per hour)
    t_half_h = t_half_min / _MIN_PER_HOUR
    k_plasma_deg_per_h = math.log(2) / t_half_h

    # Plasma degradation clearance
    cl_plasma_deg = k_plasma_deg_per_h * vd_L

    # Total in-vivo CL
    cl_total = cl_hepatic_L_per_h + cl_plasma_deg + cl_renal_L_per_h

    # Fraction degraded vs metabolized (excluding renal)
    metabolic_plus_deg = cl_plasma_deg + cl_hepatic_L_per_h
    if metabolic_plus_deg > 0:
        fraction_degraded = cl_plasma_deg / metabolic_plus_deg
    else:
        fraction_degraded = 0.0
    fraction_degraded = max(0.0, min(1.0, fraction_degraded))

    # In-vivo half-life
    ke_total = cl_total / vd_L
    t_half_in_vivo_h = math.log(2) / ke_total if ke_total > 0 else float("inf")

    notes = (
        f"Stability t1/2={t_half_min:.1f} min ({source}), "
        f"category={category}. "
        f"k_deg={k_plasma_deg_per_h:.4f}/h, "
        f"CL_total={cl_total:.2f} L/h, "
        f"t1/2_in_vivo={t_half_in_vivo_h:.2f} h."
    )

    return PlasmaStabilityPKResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        has_ester=has_ester,
        has_carbonate=has_carbonate,
        has_lactone=has_lactone,
        has_amide=has_amide,
        t_half_plasma_stability_min=t_half_min,
        stability_category=category,
        k_plasma_deg_per_h=k_plasma_deg_per_h,
        cl_plasma_deg_L_per_h=cl_plasma_deg,
        fraction_degraded_vs_metabolized=fraction_degraded,
        predicted_in_vivo_cl_L_per_h=cl_total,
        t_half_in_vivo_h=t_half_in_vivo_h,
        notes=notes,
    )


def screen_plasma_stability(
    compounds: list[dict],
) -> list[PlasmaStabilityPKResult]:
    """Screen a list of compound dicts, sorted by t_half_plasma_stability_min ascending."""
    results = []
    for compound in compounds:
        result = predict_plasma_stability_pk(**compound)
        results.append(result)
    results.sort(key=lambda r: r.t_half_plasma_stability_min)
    return results
