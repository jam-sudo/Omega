"""Transdermal/dermal drug absorption PBPK model.

4-compartment forward Euler ODE:
    Patch/gel dose -> Stratum Corneum -> Viable Epidermis -> Dermis -> Plasma
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["SkinPKResult", "simulate_skin_pk", "compare_transdermal_routes"]

_VALID_ROUTES = {"patch", "gel"}


@dataclass
class SkinPKResult:
    """Result of a transdermal/dermal PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    logP: float
    mw: float
    times_h: list = field(default_factory=list)
    a_sc_mg: list = field(default_factory=list)
    a_ve_mg: list = field(default_factory=list)
    a_d_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    k_sc_ve_per_h: float = 0.0
    lag_time_h: float = 0.0
    notes: str = ""


def _compute_k_sc_ve(logP: float, mw: float) -> float:
    """SC to VE permeation rate (1/h), clamped to [0.01, 5.0]."""
    rate = 0.1 * math.exp(0.5 * logP) * math.sqrt(200.0 / mw)
    return max(0.01, min(5.0, rate))


def simulate_skin_pk(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    mw: float = 300.0,
    route: str = "patch",
    cl_L_per_h: float = 5.0,
    vd_L: float = 100.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SkinPKResult:
    """Simulate transdermal/dermal drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total dose in the patch or gel (mg).
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    route:
        "patch" or "gel".
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Step size (h).

    Returns
    -------
    SkinPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")

    # Rate constants
    k_dose_to_sc = 2.0 if route == "gel" else 0.5  # 1/h
    k_sc_ve = _compute_k_sc_ve(logP, mw)
    k_ve_d = 2.0  # 1/h
    k_d_sys = 0.5  # 1/h
    ke = cl_L_per_h / vd_L  # 1/h

    # State variables
    a_patch = dose_mg  # remaining in patch/gel
    a_sc = 0.0  # amount in stratum corneum (mg)
    a_ve = 0.0  # amount in viable epidermis (mg)
    a_d = 0.0  # amount in dermis (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)

    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    a_sc_list: list[float] = []
    a_ve_list: list[float] = []
    a_d_list: list[float] = []
    c_plasma_list: list[float] = []

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        a_sc_list.append(a_sc)
        a_ve_list.append(a_ve)
        a_d_list.append(a_d)
        c_plasma_list.append(c_plasma)

        # Derivatives
        d_a_patch = -k_dose_to_sc * a_patch
        d_a_sc = k_dose_to_sc * a_patch - k_sc_ve * a_sc
        d_a_ve = k_sc_ve * a_sc - k_ve_d * a_ve
        d_a_d = k_ve_d * a_ve - k_d_sys * a_d
        d_c_plasma = k_d_sys * a_d / vd_L - ke * c_plasma

        # Forward Euler
        a_patch += d_a_patch * dt_h
        a_sc += d_a_sc * dt_h
        a_ve += d_a_ve * dt_h
        a_d += d_a_d * dt_h
        c_plasma += d_c_plasma * dt_h

        # Clamp negatives
        a_patch = max(0.0, a_patch)
        a_sc = max(0.0, a_sc)
        a_ve = max(0.0, a_ve)
        a_d = max(0.0, a_d)
        c_plasma = max(0.0, c_plasma)

    # Compute summary metrics
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)] if cmax > 0 else 0.0

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times[i] - times[i - 1])

    t_half = math.log(2.0) / ke if ke > 0 else 0.0

    # Lag time: time to reach 5% of cmax
    threshold = 0.05 * cmax
    lag_time = 0.0
    if cmax > 0:
        for i, c in enumerate(c_plasma_list):
            if c >= threshold:
                lag_time = times[i]
                break

    notes = (
        f"Route: {route}; k_sc_ve={k_sc_ve:.4f} 1/h; logP={logP}; MW={mw} g/mol; ke={ke:.4f} 1/h"
    )

    return SkinPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        logP=logP,
        mw=mw,
        times_h=times,
        a_sc_mg=a_sc_list,
        a_ve_mg=a_ve_list,
        a_d_mg=a_d_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        k_sc_ve_per_h=k_sc_ve,
        lag_time_h=lag_time,
        notes=notes,
    )


def compare_transdermal_routes(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    mw: float = 300.0,
) -> dict:
    """Compare patch vs gel transdermal delivery.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).

    Returns
    -------
    dict with keys: patch_result, gel_result, gel_faster, notes
    """
    patch_result = simulate_skin_pk(drug_name, dose_mg, logP=logP, mw=mw, route="patch")
    gel_result = simulate_skin_pk(drug_name, dose_mg, logP=logP, mw=mw, route="gel")

    gel_faster = gel_result.tmax_h < patch_result.tmax_h

    notes = (
        f"Patch tmax={patch_result.tmax_h:.1f}h, Cmax={patch_result.cmax_mg_L:.4f} mg/L; "
        f"Gel tmax={gel_result.tmax_h:.1f}h, Cmax={gel_result.cmax_mg_L:.4f} mg/L; "
        f"Gel faster={gel_faster}"
    )

    return {
        "patch_result": patch_result,
        "gel_result": gel_result,
        "gel_faster": gel_faster,
        "notes": notes,
    }
