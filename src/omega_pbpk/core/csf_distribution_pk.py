"""CSF drug distribution PK: 3-compartment plasma -> CSF -> brain ISF model."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(times)):
        total += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return total


@dataclass
class CSFDistributionResult:
    """Result of CSF drug distribution simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_csf_mg_L: list
    c_isf_mg_L: list
    cmax_csf_mg_L: float
    tmax_csf_h: float
    auc_csf_mg_h_per_L: float
    csf_plasma_ratio: float
    ps_cp_L_per_h: float
    notes: str


def simulate_csf_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    fu_plasma: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CSFDistributionResult:
    """Simulate 3-compartment CSF drug distribution using Forward Euler.

    Compartments:
        1. Plasma (analytical decay)
        2. CSF (lateral/4th ventricle/subarachnoid)
        3. Brain ISF

    Args:
        drug_name: Drug identifier.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient.
        cl_sys_L_per_h: Systemic clearance (L/h).
        vd_plasma_L: Plasma volume of distribution (L).
        fu_plasma: Unbound fraction in plasma.
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        CSFDistributionResult with concentration-time profiles and derived PK.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")

    # Derived plasma PK
    k_el = cl_sys_L_per_h / vd_plasma_L
    c0_plasma = dose_mg / vd_plasma_L

    # CSF parameters
    v_csf = 0.15  # L (total CSF volume)
    q_csf = 0.021  # L/h (CSF bulk flow rate: 500 mL/day)
    kp_csf = fu_plasma  # CSF is essentially protein-free
    k_exchange = 0.5  # /h (CSF-ISF exchange rate)
    k_isf_el = 0.1  # /h (ISF drug elimination)

    # Choroid plexus permeability-surface area product
    if logP > 0:
        ps_cp = 0.001 * (1.0 + logP)
    else:
        ps_cp = 0.001  # L/h

    # Initial conditions
    c_csf = 0.0
    c_isf = 0.0

    times: list = []
    c_plasma_list: list = []
    c_csf_list: list = []
    c_isf_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _i in range(n_steps):
        # Plasma concentration (analytical solution)
        c_plasma = c0_plasma * math.exp(-k_el * t)

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_csf_list.append(c_csf)
        c_isf_list.append(c_isf)

        # ODE: CSF
        dcsf_dt = (ps_cp / v_csf) * (c_plasma - c_csf / kp_csf) - (q_csf / v_csf) * c_csf
        # ODE: Brain ISF
        disf_dt = k_exchange * (c_csf - c_isf) - k_isf_el * c_isf

        c_csf = max(0.0, c_csf + dcsf_dt * dt_h)
        c_isf = max(0.0, c_isf + disf_dt * dt_h)

        t += dt_h

    # Derived PK metrics
    auc_csf = _trapz(times, c_csf_list)
    cmax_csf = max(c_csf_list) if c_csf_list else 0.0
    tmax_idx = c_csf_list.index(cmax_csf)
    tmax_csf = times[tmax_idx]

    # CSF/plasma ratio at tmax
    c_plasma_at_tmax = c_plasma_list[tmax_idx]
    if c_plasma_at_tmax > 0:
        csf_plasma_ratio = c_csf_list[tmax_idx] / c_plasma_at_tmax
    else:
        csf_plasma_ratio = 0.0

    notes = (
        f"PS_cp={ps_cp:.4f} L/h; Kp_csf={kp_csf:.3f}; "
        f"CSF/plasma ratio at Tmax={csf_plasma_ratio:.4f}; "
        f"Tmax_csf={tmax_csf:.1f}h"
    )

    return CSFDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_csf_mg_L=c_csf_list,
        c_isf_mg_L=c_isf_list,
        cmax_csf_mg_L=cmax_csf,
        tmax_csf_h=tmax_csf,
        auc_csf_mg_h_per_L=auc_csf,
        csf_plasma_ratio=csf_plasma_ratio,
        ps_cp_L_per_h=ps_cp,
        notes=notes,
    )


def compare_logp_csf(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    fu_plasma: float = 0.1,
) -> list:
    """Compare CSF penetration across a range of logP values.

    Args:
        drug_name: Base drug name (logP appended automatically).
        dose_mg: IV bolus dose in mg.
        logp_values: List of logP values to screen.
        cl_sys_L_per_h: Systemic clearance (L/h).
        vd_plasma_L: Plasma volume of distribution (L).
        fu_plasma: Unbound fraction in plasma.

    Returns:
        List of CSFDistributionResult sorted by auc_csf descending.
    """
    results = []
    for lp in logp_values:
        name = f"{drug_name}_logP{lp}"
        res = simulate_csf_distribution(
            drug_name=name,
            dose_mg=dose_mg,
            logP=lp,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
            fu_plasma=fu_plasma,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_csf_mg_h_per_L, reverse=True)
    return results
