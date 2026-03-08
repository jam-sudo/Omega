"""
Phase 507 — Renal Tubular Secretion Model

Mechanistic renal PK model:
  - GFR filtration
  - Active tubular secretion (Michaelis-Menten)
  - Passive reabsorption

Uses Forward Euler for simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RenalTubularResult:
    """Result of renal tubular secretion simulation."""

    drug_name: str
    dose_mg: float
    gfr_L_per_h: float
    fup: float
    vmax_s_mg_per_h: float
    km_s_mg_per_L: float
    kreab: float
    cl_nonrenal_L_per_h: float

    # Clearance components (at C=0 or initial conditions)
    cl_filtration_L_per_h: float
    cl_secretion_L_per_h: float
    cl_reabsorption_L_per_h: float
    cl_renal_L_per_h: float
    cl_total_L_per_h: float
    fe_renal: float

    # Time-course data
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    urine_rate_mg_per_h: list[float] = field(default_factory=list)
    cumulative_urine_mg: list[float] = field(default_factory=list)

    total_urine_mg_excreted: float = 0.0
    t_half_h: float = 0.0
    notes: str = ""


def simulate_renal_tubular_secretion(
    drug_name: str,
    dose_mg: float,
    fup: float,
    vmax_s_mg_per_h: float,
    km_s_mg_per_L: float,
    kreab: float = 0.4,
    cl_nonrenal_L_per_h: float = 5.0,
    gfr_L_per_h: float = 7.2,
    vd_L: float = 30.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RenalTubularResult:
    """
    Simulate 1-compartment IV bolus PK with mechanistic renal elimination.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        IV bolus dose in mg.
    fup : float
        Unbound fraction in plasma (0–1].
    vmax_s_mg_per_h : float
        Maximum tubular secretion rate (mg/h).
    km_s_mg_per_L : float
        Michaelis constant for secretion (mg/L).
    kreab : float
        Passive reabsorption fraction [0, 1).
    cl_nonrenal_L_per_h : float
        Non-renal clearance (hepatic, etc.) in L/h.
    gfr_L_per_h : float
        Glomerular filtration rate in L/h (default 7.2 L/h = 120 mL/min).
    vd_L : float
        Volume of distribution in L.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Forward Euler step size in hours.

    Returns
    -------
    RenalTubularResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if vmax_s_mg_per_h <= 0:
        raise ValueError("vmax_s_mg_per_h must be positive")
    if km_s_mg_per_L <= 0:
        raise ValueError("km_s_mg_per_L must be positive")
    if not (0 <= kreab < 1):
        raise ValueError("kreab must be in [0, 1)")
    if cl_nonrenal_L_per_h <= 0:
        raise ValueError("cl_nonrenal_L_per_h must be positive")

    # Clearance components at C=0 (linear approximation for initial rates)
    cl_filtration = gfr_L_per_h * fup  # L/h
    # Intrinsic secretion CL at C->0: CLs = Vmax_s / Km_s (L/h)
    cl_secretion_linear = vmax_s_mg_per_h / km_s_mg_per_L  # L/h, intrinsic secretion capacity

    cl_reabsorption = kreab * cl_filtration  # L/h equivalent
    cl_renal_linear = cl_filtration * (1.0 - kreab) + cl_secretion_linear * fup
    cl_total_linear = cl_nonrenal_L_per_h + cl_renal_linear
    fe_renal_linear = cl_renal_linear / cl_total_linear

    # Forward Euler simulation
    n_steps = max(1, int(t_end_h / dt_h))
    actual_dt = t_end_h / n_steps

    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []
    urine_rate_mg_per_h: list[float] = []
    cumulative_urine_mg: list[float] = []

    C = dose_mg / vd_L  # initial plasma concentration (mg/L)
    cum_urine = 0.0

    for i in range(n_steps + 1):
        t = i * actual_dt
        times_h.append(t)
        c_plasma_mg_L.append(C)

        # Michaelis-Menten secretion rate at current concentration
        # Rate (mg/h) = Vmax_s * C_unbound / (Km_s + C_unbound)
        c_unbound = C * fup
        secretion_rate_mg_per_h = vmax_s_mg_per_h * c_unbound / (km_s_mg_per_L + c_unbound)

        # Filtration rate (mg/h) = cl_filtration * C
        filtration_rate = cl_filtration * C  # mg/h filtered into tubule

        # Net excretion into urine (mg/h) = filtration*(1-kreab) + secretion
        net_excretion = filtration_rate * (1.0 - kreab) + secretion_rate_mg_per_h

        urine_rate_mg_per_h.append(net_excretion)

        if i > 0:
            # Trapezoidal increment for cumulative urine
            trapez = 0.5 * (urine_rate_mg_per_h[i] + urine_rate_mg_per_h[i - 1]) * actual_dt
            cum_urine += trapez
        cumulative_urine_mg.append(cum_urine)

        # ODE for plasma: dC/dt = -(CL_total / Vd) * C
        # Where CL_total = CL_nonrenal + CL_filtration*(1-kreab) + (secretion_rate/C)*fup
        # But secretion is a rate (mg/h), so effective CL = secretion_rate / C (if C > 0)
        if C > 1e-12:
            cl_secretion_eff = secretion_rate_mg_per_h / C  # effective secretion clearance (L/h)
        else:
            cl_secretion_eff = cl_secretion_linear * fup

        cl_renal_eff = cl_filtration * (1.0 - kreab) + cl_secretion_eff
        cl_total_eff = cl_nonrenal_L_per_h + cl_renal_eff

        dCdt = -(cl_total_eff / vd_L) * C
        C = C + actual_dt * dCdt
        if C < 0:
            C = 0.0

    total_urine = cumulative_urine_mg[-1]

    # t_half: using linear CL approximation (dominant at low C)
    t_half = math.log(2.0) * vd_L / cl_total_linear if cl_total_linear > 0 else float("inf")

    notes = (
        f"Nonlinear renal: Vmax_s={vmax_s_mg_per_h:.2f} mg/h, "
        f"Km_s={km_s_mg_per_L:.2f} mg/L; "
        f"kreab={kreab:.2f}; GFR={gfr_L_per_h:.1f} L/h"
    )

    return RenalTubularResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gfr_L_per_h=gfr_L_per_h,
        fup=fup,
        vmax_s_mg_per_h=vmax_s_mg_per_h,
        km_s_mg_per_L=km_s_mg_per_L,
        kreab=kreab,
        cl_nonrenal_L_per_h=cl_nonrenal_L_per_h,
        cl_filtration_L_per_h=cl_filtration,
        cl_secretion_L_per_h=cl_secretion_linear,
        cl_reabsorption_L_per_h=cl_reabsorption,
        cl_renal_L_per_h=cl_renal_linear,
        cl_total_L_per_h=cl_total_linear,
        fe_renal=fe_renal_linear,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        urine_rate_mg_per_h=urine_rate_mg_per_h,
        cumulative_urine_mg=cumulative_urine_mg,
        total_urine_mg_excreted=total_urine,
        t_half_h=t_half,
        notes=notes,
    )


def compare_gfr_levels(
    drug_name: str,
    dose_mg: float,
    fup: float,
    vmax_s_mg_per_h: float,
    km_s_mg_per_L: float,
    gfr_values_L_per_h: list,
) -> list:
    """
    Compare renal elimination across different GFR levels.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    fup : float
    vmax_s_mg_per_h : float
    km_s_mg_per_L : float
    gfr_values_L_per_h : list
        List of GFR values (L/h) to compare.

    Returns
    -------
    list of RenalTubularResult
        Sorted by fe_renal descending.
    """
    results = []
    for gfr in gfr_values_L_per_h:
        r = simulate_renal_tubular_secretion(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fup=fup,
            vmax_s_mg_per_h=vmax_s_mg_per_h,
            km_s_mg_per_L=km_s_mg_per_L,
            gfr_L_per_h=gfr,
        )
        results.append(r)

    results.sort(key=lambda r: r.fe_renal, reverse=True)
    return results
