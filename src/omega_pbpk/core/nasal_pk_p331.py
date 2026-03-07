"""Phase 331 — Nasal Drug Delivery PK Model.

3-compartment Forward Euler model:
  nasal_depot -> mucosal_tissue -> systemic
  nasal_depot -> mucociliary clearance (loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class NasalPKResult:
    """Result of nasal PK simulation."""

    times_h: list
    c_nasal_depot_mg_mL: list
    c_mucosal_mg_mL: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    mucociliary_loss_pct: float
    bioavailability_pct: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_nasal_pk(
    drug_name: str,
    dose_mg: float,
    k_abs_nasal_per_h: float,
    k_mucosal_per_h: float,
    k_muco_clearance_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float,
    dt_h: float,
    v_nasal_mL: float = 10.0,
    v_mucosal_mL: float = 2.0,
) -> NasalPKResult:
    """Simulate nasal drug delivery PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Nasal dose in mg.
    k_abs_nasal_per_h:
        First-order rate constant for nasal absorption to systemic (1/h).
    k_mucosal_per_h:
        First-order rate constant for drug partitioning into mucosal tissue (1/h).
    k_muco_clearance_per_h:
        First-order rate constant for mucociliary clearance loss (1/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).
    v_nasal_mL:
        Volume of nasal cavity fluid (mL), default 10.0.
    v_mucosal_mL:
        Volume of mucosal tissue (mL), default 2.0.

    Returns
    -------
    NasalPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if k_abs_nasal_per_h <= 0:
        raise ValueError("k_abs_nasal_per_h must be > 0")
    if k_mucosal_per_h < 0:
        raise ValueError("k_mucosal_per_h must be >= 0")
    if k_muco_clearance_per_h < 0:
        raise ValueError("k_muco_clearance_per_h must be >= 0")
    if v_nasal_mL <= 0:
        raise ValueError("v_nasal_mL must be > 0")
    if v_mucosal_mL <= 0:
        raise ValueError("v_mucosal_mL must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0 or dt_h > t_end_h:
        raise ValueError("dt_h must be > 0 and <= t_end_h")

    # Fixed mucosal absorption rate constant
    k_abs_mucosal_per_h = 0.1

    # ke for systemic
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Initial conditions
    A_nasal = dose_mg  # amount in nasal depot (mg)
    C_mucosal = 0.0  # concentration in mucosal tissue (mg/mL)
    C_sys = 0.0  # concentration in systemic (mg/L)

    # Track mucociliary loss amount
    A_muco_lost = 0.0

    # Storage
    times_h: list = []
    c_nasal_depot: list = []
    c_mucosal: list = []
    c_sys_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_nasal_depot.append(A_nasal / v_nasal_mL)
        c_mucosal.append(C_mucosal)
        c_sys_list.append(C_sys)

        if step < n_steps:
            # Rates
            # nasal depot: absorption to systemic + mucosal partitioning + mucociliary clearance
            dA_nasal = -(k_abs_nasal_per_h + k_mucosal_per_h + k_muco_clearance_per_h) * A_nasal

            # mucosal tissue: uptake from nasal - absorption to systemic
            dC_mucosal = (
                k_mucosal_per_h * A_nasal / v_mucosal_mL
            ) - k_abs_mucosal_per_h * C_mucosal

            # systemic: from nasal direct + from mucosal
            input_rate_mg_h = (
                k_abs_nasal_per_h * A_nasal + k_abs_mucosal_per_h * C_mucosal * v_mucosal_mL
            )
            # convert mg/h to (mg/L)/h: divide by vd_sys_L
            dC_sys = input_rate_mg_h / vd_sys_L - ke_sys * C_sys

            # mucociliary loss rate
            dA_muco_lost = k_muco_clearance_per_h * A_nasal

            # Forward Euler update
            A_nasal += dA_nasal * dt_h
            if A_nasal < 0.0:
                A_nasal = 0.0
            C_mucosal += dC_mucosal * dt_h
            if C_mucosal < 0.0:
                C_mucosal = 0.0
            C_sys += dC_sys * dt_h
            if C_sys < 0.0:
                C_sys = 0.0
            A_muco_lost += dA_muco_lost * dt_h

    # Derived metrics
    auc_sys = _trapezoidal_auc(times_h, c_sys_list)
    cmax_sys = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax_sys)
    tmax_sys = times_h[tmax_idx]

    # Mucociliary loss percentage
    mucociliary_loss_pct = min(100.0, max(0.0, A_muco_lost / dose_mg * 100.0))

    # Bioavailability: compare AUC_nasal vs AUC_iv
    # AUC_iv (1-cpt IV bolus) = dose / cl_sys
    auc_iv = dose_mg / cl_sys_L_per_h
    if auc_iv > 0 and isfinite(auc_iv):
        bioavailability_pct = min(100.0, max(0.0, auc_sys / auc_iv * 100.0))
    else:
        bioavailability_pct = 0.0

    notes = (
        f"Nasal PK simulation for {drug_name}. "
        f"Mucociliary clearance removes {mucociliary_loss_pct:.1f}% of dose. "
        f"Nasal bioavailability: {bioavailability_pct:.1f}%."
    )

    return NasalPKResult(
        times_h=times_h,
        c_nasal_depot_mg_mL=c_nasal_depot,
        c_mucosal_mg_mL=c_mucosal,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc_sys,
        cmax_systemic_mg_L=cmax_sys,
        tmax_systemic_h=tmax_sys,
        mucociliary_loss_pct=mucociliary_loss_pct,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )
