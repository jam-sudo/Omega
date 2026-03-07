"""Phase 1029 — Drug Synovial Fluid PK.

Models drug distribution into synovial fluid for articular drug delivery
(arthritis, local injections).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SynovialFluidPKResult:
    drug_name: str
    dose_mg: float
    route: str  # "intraarticular" or "systemic"
    times_h: list
    c_plasma_mg_L: list
    c_synovial_mg_L: list
    auc_plasma: float
    auc_synovial: float
    cmax_synovial_mg_L: float
    tmax_synovial_h: float
    synovial_to_plasma_auc_ratio: float
    t_half_synovial_h: float  # elimination half-life in joint
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (values[i] + values[i + 1]) * dt
    return auc


def _safe_dt(dt_h: float, k_max: float) -> float:
    """Return a numerically stable dt for Forward Euler given max rate constant.

    Stability condition for Forward Euler: dt * k_max < 1.0
    Use dt <= 0.5 / k_max for safety margin, but no smaller than 1e-5 h.
    """
    if k_max <= 0.0:
        return dt_h
    stable_dt = 0.5 / k_max
    return max(1e-5, min(dt_h, stable_dt))


def simulate_synovial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "systemic",
    mw_Da: float = 300.0,
    fu_plasma: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    v_synovial_mL: float = 2.0,
    q_synovial_L_h: float = 0.1,
    kp_synovial: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SynovialFluidPKResult:
    """Simulate drug PK in synovial fluid compartment.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        route: "systemic" (IV bolus to plasma) or "intraarticular" (direct joint injection).
        mw_Da: Molecular weight in Daltons; higher MW -> lower membrane permeability.
        fu_plasma: Fraction unbound in plasma (unused directly, stored for reference).
        cl_sys_L_per_h: Systemic clearance (L/h).
        vd_sys_L: Volume of distribution (L).
        v_synovial_mL: Synovial fluid volume (mL).
        q_synovial_L_h: Blood flow to synovial membrane (L/h).
        kp_synovial: Synovial/plasma partition coefficient.
        t_end_h: Simulation end time (hours).
        dt_h: Requested time step for output (hours). Internal dt may be smaller for stability.

    Returns:
        SynovialFluidPKResult with time-course concentrations and PK metrics.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if v_synovial_mL <= 0:
        raise ValueError("v_synovial_mL must be > 0")
    if kp_synovial <= 0:
        raise ValueError("kp_synovial must be > 0")
    if route not in {"intraarticular", "systemic"}:
        raise ValueError(f"route must be 'intraarticular' or 'systemic', got '{route}'")

    # MW-dependent permeability factor: large molecules penetrate less
    perm_factor = math.exp(-mw_Da / 5000.0)

    # Transfer rate constants between plasma and synovial fluid
    v_synovial_L = v_synovial_mL / 1000.0
    k_in = q_synovial_L_h * perm_factor / v_synovial_L  # /h into synovial
    k_out = k_in / kp_synovial  # /h out of synovial

    # Systemic elimination rate constant
    ke = cl_sys_L_per_h / vd_sys_L

    # Choose a stable internal dt
    k_max = max(k_in, k_out, ke)
    dt_internal = _safe_dt(dt_h, k_max)

    # Record interval: how many internal steps per output point
    # We record at multiples of dt_h (user's desired output resolution)
    record_every = max(1, round(dt_h / dt_internal))
    # Recalculate dt_internal to align exactly
    dt_internal = dt_h / record_every

    # Initial conditions
    if route == "systemic":
        c_plasma = dose_mg / vd_sys_L
        c_syn = 0.0
    else:  # intraarticular
        c_plasma = 0.0
        c_syn = dose_mg / v_synovial_L

    # Forward Euler ODE integration
    n_output = int(round(t_end_h / dt_h))
    times_h = []
    c_plasma_list = []
    c_synovial_list = []

    t = 0.0
    step_count = 0
    total_internal_steps = n_output * record_every

    for _i in range(total_internal_steps + 1):
        if _i % record_every == 0:
            times_h.append(round(t, 10))
            c_plasma_list.append(max(c_plasma, 0.0))
            c_synovial_list.append(max(c_syn, 0.0))
            step_count += 1

        if _i < total_internal_steps:
            if route == "systemic":
                dc_plasma = -ke * c_plasma
                dc_syn = k_in * c_plasma - k_out * c_syn
            else:  # intraarticular
                dc_syn = -k_out * c_syn
                dc_plasma = k_out * c_syn * v_synovial_L / vd_sys_L - ke * c_plasma

            c_plasma += dc_plasma * dt_internal
            c_syn += dc_syn * dt_internal
            t += dt_internal

    # AUC using trapezoidal rule
    auc_plasma = _trapz(times_h, c_plasma_list)
    auc_synovial = _trapz(times_h, c_synovial_list)

    # Cmax and Tmax for synovial
    cmax_synovial = max(c_synovial_list) if c_synovial_list else 0.0
    tmax_idx = c_synovial_list.index(cmax_synovial)
    tmax_synovial = times_h[tmax_idx]

    # Analytical half-life in synovial (ln2/k_out)
    if k_out > 0:
        t_half_synovial = math.log(2.0) / k_out
    else:
        t_half_synovial = float("inf")

    # Synovial-to-plasma AUC ratio
    if auc_plasma > 0:
        stp_ratio = auc_synovial / auc_plasma
    else:
        stp_ratio = 0.0

    notes = (
        f"MW={mw_Da:.0f} Da, perm_factor={perm_factor:.3f}, "
        f"k_in={k_in:.4f}/h, k_out={k_out:.4f}/h, "
        f"route={route}, kp_synovial={kp_synovial}, "
        f"dt_internal={dt_internal:.5f}h"
    )

    return SynovialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_synovial_mg_L=c_synovial_list,
        auc_plasma=auc_plasma,
        auc_synovial=auc_synovial,
        cmax_synovial_mg_L=cmax_synovial,
        tmax_synovial_h=tmax_synovial,
        synovial_to_plasma_auc_ratio=stp_ratio,
        t_half_synovial_h=t_half_synovial,
        notes=notes,
    )
