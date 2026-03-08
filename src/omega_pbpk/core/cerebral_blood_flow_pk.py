"""Phase 460 — Cerebral Blood Flow PK.

Physiological model of drug distribution in the brain considering
cerebral blood flow, BBB permeability, and ECF/ICF compartments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CerebralPKResult:
    """Result of cerebral blood flow PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_brain_ecf_mg_L: list
    c_brain_icf_mg_L: list
    cmax_brain_ecf_mg_L: float
    tmax_brain_ecf_h: float
    auc_brain_ecf_mg_h_per_L: float
    brain_to_plasma_ratio: float
    t_half_brain_ecf_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _estimate_t_half_brain(times_h: list, c_ecf: list) -> float:
    """Estimate terminal half-life from brain ECF profile."""
    cmax = max(c_ecf) if c_ecf else 0.0
    if cmax <= 0.0:
        return float("nan")
    # Find last index where concentration >= cmax * 0.5 from the end
    half_cmax = cmax * 0.5
    # Find cmax index
    cmax_idx = c_ecf.index(cmax)
    # Find index after cmax where concentration drops below half_cmax
    for i in range(cmax_idx, len(c_ecf)):
        if c_ecf[i] <= half_cmax:
            if i == 0:
                return float("nan")
            # Linear interpolation
            t1 = times_h[i - 1]
            t2 = times_h[i]
            c1 = c_ecf[i - 1]
            c2 = c_ecf[i]
            if abs(c2 - c1) < 1e-15:
                return float("nan")
            t_half = t1 + (half_cmax - c1) / (c2 - c1) * (t2 - t1)
            return t_half - times_h[cmax_idx]
    return float("nan")


def simulate_cerebral_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    Kp_brain: float = 0.3,
    PS_BBB_L_per_h: float = 0.1,
    cbf_L_per_h: float = 0.7,
    vd_ecf_L: float = 0.25,
    vd_icf_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CerebralPKResult:
    """Simulate cerebral blood flow PK for an IV bolus dose.

    Plasma follows a one-compartment IV bolus:
        C_plasma(t) = (dose_mg / vd_sys_L) * exp(-k_el * t)

    Brain ECF ODE (Forward Euler):
        dC_ecf/dt = (1/vd_ecf_L) * [J_in - J_out - J_ecf_to_icf]
        J_in  = PS_BBB * C_plasma          (influx from plasma)
        J_out = PS_BBB * C_ecf / Kp_brain  (efflux to plasma)
        J_ecf_to_icf = k_exchange * (C_ecf - C_icf * 0.5) * vd_ecf_L

    Brain ICF ODE:
        dC_icf/dt = (1/vd_icf_L) * J_ecf_to_icf

    Parameters
    ----------
    drug_name:
        Compound identifier.
    dose_mg:
        IV bolus dose (mg).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    Kp_brain:
        Brain ECF / plasma equilibrium partition coefficient.
    PS_BBB_L_per_h:
        BBB permeability-surface area product (L/h).
    cbf_L_per_h:
        Cerebral blood flow (L/h); limits effective BBB permeation.
    vd_ecf_L:
        Brain extracellular fluid volume (L).
    vd_icf_L:
        Brain intracellular fluid volume (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    CerebralPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if Kp_brain <= 0:
        raise ValueError("Kp_brain must be > 0")
    if PS_BBB_L_per_h <= 0:
        raise ValueError("PS_BBB_L_per_h must be > 0")

    k_el = cl_sys_L_per_h / vd_sys_L
    c0_plasma = dose_mg / vd_sys_L

    # Effective PS limited by cerebral blood flow (similar to flow-limited extraction)
    PS_eff = min(PS_BBB_L_per_h, cbf_L_per_h)

    # ECF <-> ICF exchange rate constant (h^-1)
    k_exchange = 0.1  # h^-1, passive exchange

    # Initialize
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times_h: list = []
    c_plasma_list: list = []
    c_ecf_list: list = []
    c_icf_list: list = []

    c_ecf = 0.0
    c_icf = 0.0

    for i in range(n_steps):
        t = i * dt_h
        c_plasma = c0_plasma * math.exp(-k_el * t)

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_ecf_list.append(c_ecf)
        c_icf_list.append(c_icf)

        # Forward Euler update
        # J_in: influx from plasma to ECF (mg/h)
        j_in = PS_eff * c_plasma
        # J_out: efflux from ECF to plasma (mg/h)
        j_out = PS_eff * c_ecf / Kp_brain
        # ECF <-> ICF exchange (mg/h)
        j_ecf_icf = k_exchange * vd_ecf_L * (c_ecf - c_icf * 0.5)

        dc_ecf = (j_in - j_out - j_ecf_icf) / vd_ecf_L
        dc_icf = j_ecf_icf / vd_icf_L

        c_ecf = max(0.0, c_ecf + dc_ecf * dt_h)
        c_icf = max(0.0, c_icf + dc_icf * dt_h)

    # Derived metrics
    cmax_ecf = max(c_ecf_list) if c_ecf_list else 0.0
    tmax_ecf = times_h[c_ecf_list.index(cmax_ecf)] if cmax_ecf > 0 else 0.0
    auc_ecf = _trapezoidal_auc(times_h, c_ecf_list)

    # Brain-to-plasma ratio at last time point
    c_plasma_last = c_plasma_list[-1] if c_plasma_list else 0.0
    c_ecf_last = c_ecf_list[-1] if c_ecf_list else 0.0
    if c_plasma_last > 1e-12:
        bpr = c_ecf_last / c_plasma_last
    else:
        # Use ratio at tmax_ecf if plasma is negligible at end
        tmax_idx = c_ecf_list.index(cmax_ecf) if cmax_ecf > 0 else 0
        cp_at_tmax = c_plasma_list[tmax_idx]
        bpr = c_ecf_list[tmax_idx] / cp_at_tmax if cp_at_tmax > 1e-12 else 0.0

    t_half = _estimate_t_half_brain(times_h, c_ecf_list)

    # Notes
    notes_parts = [
        f"IV bolus; k_el={k_el:.3f} h-1",
        f"PS_eff={PS_eff:.3f} L/h",
        f"Kp_brain={Kp_brain:.2f}",
    ]
    if math.isnan(t_half):
        notes_parts.append("t_half_brain not determinable")
    notes = "; ".join(notes_parts)

    return CerebralPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_brain_ecf_mg_L=c_ecf_list,
        c_brain_icf_mg_L=c_icf_list,
        cmax_brain_ecf_mg_L=cmax_ecf,
        tmax_brain_ecf_h=tmax_ecf,
        auc_brain_ecf_mg_h_per_L=auc_ecf,
        brain_to_plasma_ratio=bpr,
        t_half_brain_ecf_h=t_half,
        notes=notes,
    )


__all__ = [
    "CerebralPKResult",
    "simulate_cerebral_pk",
]
