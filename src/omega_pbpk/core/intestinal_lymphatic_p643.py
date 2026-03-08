"""Phase 643 — Intestinal Lymphatic Absorption.

Models drug absorption via intestinal lymphatic and portal venous pathways.
Highly lipophilic drugs (logP >= 3) can bypass hepatic first-pass metabolism
through lymphatic uptake. Uses Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IntestinalLymphaticResult:
    """Result from intestinal lymphatic absorption simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_lymph_mg_L: list
    c_portal_mg_L: list
    lymphatic_fraction: float
    portal_fraction: float
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    auc_plasma_mg_h_per_L: float
    lymph_to_portal_auc_ratio: float
    first_pass_avoidance_pct: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> None:
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0.0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")


def _trapezoidal_auc(times: list, concs: list) -> float:
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_intestinal_lymphatic(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    lipid_formulation: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IntestinalLymphaticResult:
    """Simulate intestinal lymphatic vs portal absorption."""
    _validate_inputs(dose_mg, cl_sys_L_per_h, vd_sys_L)

    # Lymphatic fraction based on logP
    if logP < 3.0:
        f_lymph = 0.0
    else:
        f_lymph = min(0.5, (logP - 3.0) * 0.1)

    if lipid_formulation:
        f_lymph = min(0.6, f_lymph * 1.5)

    f_portal = 1.0 - f_lymph

    # Rate constants
    ka = 1.0  # /h
    k_port = 1.0  # /h
    k_lymph = 0.3  # /h
    k_el = cl_sys_L_per_h / vd_sys_L

    # Compartment volumes
    v_portal = 3.0  # L
    v_lymph = 2.0  # L

    # Initial conditions
    a_gut = dose_mg
    c_portal = 0.0
    c_lymph = 0.0
    c_plasma = 0.0

    # Storage
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c_plasma_list: list[float] = []
    c_lymph_list: list[float] = []
    c_portal_list: list[float] = []

    cmax = 0.0
    tmax = 0.0

    for step in range(n_steps):
        t = step * dt_h

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_lymph_list.append(c_lymph)
        c_portal_list.append(c_portal)

        if c_plasma > cmax:
            cmax = c_plasma
            tmax = t

        # Forward Euler
        da_gut = -ka * a_gut
        dc_portal = ka * a_gut * f_portal / v_portal - k_port * c_portal
        dc_lymph = ka * a_gut * f_lymph / v_lymph - k_lymph * c_lymph
        dc_plasma = (
            k_port * c_portal * v_portal / vd_sys_L
            + k_lymph * c_lymph * v_lymph / vd_sys_L
            - k_el * c_plasma
        )

        a_gut += da_gut * dt_h
        c_portal += dc_portal * dt_h
        c_lymph += dc_lymph * dt_h
        c_plasma += dc_plasma * dt_h

    # AUC calculations
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_lymph = _trapezoidal_auc(times_h, c_lymph_list)
    auc_portal = _trapezoidal_auc(times_h, c_portal_list)

    if auc_portal > 0.0:
        lymph_to_portal_ratio = auc_lymph / auc_portal
    else:
        lymph_to_portal_ratio = 0.0

    first_pass_avoidance = min(100.0, f_lymph * 100.0 * 0.7)

    notes_parts = [f"logP={logP}"]
    if lipid_formulation:
        notes_parts.append("lipid formulation")
    if f_lymph == 0.0:
        notes_parts.append("no lymphatic uptake (logP<3)")

    return IntestinalLymphaticResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_lymph_mg_L=c_lymph_list,
        c_portal_mg_L=c_portal_list,
        lymphatic_fraction=f_lymph,
        portal_fraction=f_portal,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc_plasma,
        lymph_to_portal_auc_ratio=lymph_to_portal_ratio,
        first_pass_avoidance_pct=first_pass_avoidance,
        notes="; ".join(notes_parts),
    )
