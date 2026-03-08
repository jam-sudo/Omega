"""
Phase 197 — Joint/Synovial Fluid PK Model

Models drug distribution into synovial fluid for arthritis treatment
(NSAIDs, biologics) using a 2-compartment Forward Euler ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SynovialPKResult:
    """Result of a synovial fluid PK simulation."""

    drug_name: str
    dose_mg: float
    inflamed: bool
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_synovial_mg_L: list[float]
    cmax_synovial_mg_L: float
    tmax_synovial_h: float
    auc_synovial_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    synovial_to_plasma_ratio: float
    notes: str


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return total


def simulate_synovial_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    inflamed: bool = True,
    joint_size: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    k_sp: float = 0.15,
    k_syn_elim: float = 0.02,
) -> SynovialPKResult:
    """
    Simulate plasma and synovial fluid PK after IV bolus.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        IV bolus dose in mg.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_plasma_L : float
        Volume of distribution for plasma compartment (L).
    inflamed : bool
        If True uses k_ps=0.1/h (inflamed), else 0.03/h (healthy).
    joint_size : float
        Scale factor for synovial volume (default 1.0 → 3.5 mL).
    t_end_h : float
        Simulation end time (hours).
    dt_h : float
        Forward Euler step size (hours).
    k_sp : float
        Synovial→plasma efflux rate (1/h).
    k_syn_elim : float
        Local synovial elimination/dilution rate (1/h).

    Returns
    -------
    SynovialPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if joint_size <= 0:
        raise ValueError("joint_size must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Parameters
    vd_synovial_L = (3.5e-3) * joint_size  # 3.5 mL → L, scaled
    k_ps = 0.1 if inflamed else 0.03  # plasma→synovial transfer (1/h)
    ke = cl_L_per_h / vd_plasma_L  # elimination rate from plasma

    # Initial conditions (IV bolus)
    cp = dose_mg / vd_plasma_L  # mg/L
    cs = 0.0  # mg/L

    times: list[float] = []
    cp_list: list[float] = []
    cs_list: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        cp_list.append(cp)
        cs_list.append(cs)

        if t >= t_end_h:
            break

        # ODEs
        dcp_dt = -ke * cp - k_ps * cp + k_sp * cs * vd_synovial_L / vd_plasma_L
        dcs_dt = k_ps * cp * vd_plasma_L / vd_synovial_L - k_sp * cs - k_syn_elim * cs

        cp = max(0.0, cp + dcp_dt * dt_h)
        cs = max(0.0, cs + dcs_dt * dt_h)
        t = min(t + dt_h, t_end_h)

    # Metrics
    cmax_synovial = max(cs_list)
    tmax_synovial = times[cs_list.index(cmax_synovial)]
    auc_synovial = _trapz(times, cs_list)
    auc_plasma = _trapz(times, cp_list)

    ratio = auc_synovial / auc_plasma if auc_plasma > 0 else 0.0

    inflammation_state = "inflamed" if inflamed else "healthy"
    notes = (
        f"2-cpt synovial PK | {inflammation_state} joint | "
        f"k_ps={k_ps}/h | Vd_syn={vd_synovial_L * 1000:.2f} mL | "
        f"joint_size={joint_size}"
    )

    return SynovialPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        inflamed=inflamed,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_synovial_mg_L=cs_list,
        cmax_synovial_mg_L=cmax_synovial,
        tmax_synovial_h=tmax_synovial,
        auc_synovial_mg_h_per_L=auc_synovial,
        auc_plasma_mg_h_per_L=auc_plasma,
        synovial_to_plasma_ratio=ratio,
        notes=notes,
    )


def compare_joint_inflammation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    joint_size: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[SynovialPKResult]:
    """
    Compare synovial PK in inflamed vs. healthy joint.

    Returns list of two SynovialPKResult objects sorted by synovial AUC
    descending (inflamed first, since inflamed has higher k_ps).
    """
    inflamed_result = simulate_synovial_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_plasma_L=vd_plasma_L,
        inflamed=True,
        joint_size=joint_size,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    healthy_result = simulate_synovial_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_plasma_L=vd_plasma_L,
        inflamed=False,
        joint_size=joint_size,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    results = [inflamed_result, healthy_result]
    results.sort(key=lambda r: r.auc_synovial_mg_h_per_L, reverse=True)
    return results
