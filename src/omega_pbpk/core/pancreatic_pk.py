"""Pancreatic drug delivery PK model — Phase 807.

Relevant for insulin, pancreatic cancer drugs, and other compounds targeting
pancreatic tissue. Models 2-compartment PK (plasma + pancreas) with portal
and arterial supply fractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PancreaticPKResult:
    """Result of pancreatic PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreas_mg_L: list
    cmax_plasma: float
    cmax_pancreas: float
    auc_plasma: float
    auc_pancreas: float
    pancreas_plasma_kp: float
    portal_delivered_pct: float  # % reaching pancreas via portal vein
    arterial_delivered_pct: float  # % via arterial supply
    t_half_pancreas_h: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_pancreatic_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    q_pancreas_L_per_h: float = 0.9,  # pancreatic blood flow
    v_pancreas_L: float = 0.1,
    kp_pancreas: float = 2.0,
    f_portal: float = 0.7,  # fraction of pancreatic blood from portal
    route: str = "iv",
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PancreaticPKResult:
    """Simulate pancreatic drug delivery PK.

    Uses a 2-compartment model: plasma + pancreas.
    Pancreas receives blood from both portal vein (hepatic first-pass)
    and arterial supply.

    Args:
        drug_name: Name of the drug compound.
        dose_mg: Dose in milligrams.
        cl_L_per_h: Plasma clearance (L/h).
        vd_plasma_L: Volume of distribution in plasma (L).
        q_pancreas_L_per_h: Pancreatic blood flow (L/h).
        v_pancreas_L: Pancreatic tissue volume (L).
        kp_pancreas: Pancreas-to-plasma partition coefficient.
        f_portal: Fraction of pancreatic blood from portal vein.
        route: Administration route ("iv" or "oral").
        f_oral: Oral bioavailability fraction.
        ka_per_h: Oral absorption rate constant (1/h).
        t_end_h: End time for simulation (h).
        dt_h: Integration time step (h).

    Returns:
        PancreaticPKResult with PK profiles and key parameters.

    Raises:
        ValueError: If dose <= 0 or f_portal not in [0, 1].
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not 0.0 <= f_portal <= 1.0:
        raise ValueError(f"f_portal must be in [0, 1], got {f_portal}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if v_pancreas_L <= 0:
        raise ValueError(f"v_pancreas_L must be > 0, got {v_pancreas_L}")

    # Initial conditions
    # Plasma amount (mg), pancreas amount (mg), gut depot for oral (mg)
    if route == "iv":
        a_plasma = dose_mg  # IV bolus directly into plasma
        a_gut = 0.0
    else:
        a_plasma = 0.0
        a_gut = dose_mg  # oral dose in gut depot

    a_pancreas = 0.0

    # Pancreatic equilibrium concentration ratio
    # At steady state: C_pancreas = kp_pancreas * C_plasma
    # Transfer clearance: Q_pan * (C_plasma - C_pancreas/kp_pancreas)
    q_pan = q_pancreas_L_per_h
    kp = kp_pancreas

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_pancreas_list = []

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)

        c_plasma = a_plasma / vd_plasma_L
        c_pancreas = a_pancreas / v_pancreas_L

        c_plasma_list.append(c_plasma)
        c_pancreas_list.append(c_pancreas)

        # Forward Euler derivatives
        # Plasma: input - clearance - distribution to pancreas
        # For oral: absorption from gut adds to plasma
        if route == "oral":
            input_rate = ka_per_h * a_gut * f_oral
        else:
            input_rate = 0.0

        # Net flux from plasma to pancreas: Q*(C_plasma - C_pancreas/kp)
        flux_to_pancreas = q_pan * (c_plasma - c_pancreas / kp)

        da_plasma_dt = input_rate - cl_L_per_h * c_plasma - flux_to_pancreas
        da_pancreas_dt = flux_to_pancreas

        if route == "oral":
            da_gut_dt = -ka_per_h * a_gut
        else:
            da_gut_dt = 0.0

        # Update state
        a_plasma += da_plasma_dt * dt_h
        a_pancreas += da_pancreas_dt * dt_h
        a_gut += da_gut_dt * dt_h

        # Prevent negative amounts
        if a_plasma < 0.0:
            a_plasma = 0.0
        if a_pancreas < 0.0:
            a_pancreas = 0.0
        if a_gut < 0.0:
            a_gut = 0.0

    # Compute derived metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_pancreas = max(c_pancreas_list) if c_pancreas_list else 0.0

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_pancreas = _trapezoidal_auc(times, c_pancreas_list)

    # Steady-state Kp ratio (AUC-based)
    if auc_plasma > 0:
        pancreas_plasma_kp = auc_pancreas / auc_plasma
    else:
        pancreas_plasma_kp = kp

    # Portal vs arterial delivery percentages
    portal_delivered_pct = f_portal * 100.0
    arterial_delivered_pct = (1.0 - f_portal) * 100.0

    # Terminal half-life in pancreas: estimated from elimination rate
    # k_elim_pancreas = Q_pan / (kp * V_pancreas) at steady state
    k_elim_pancreas = q_pan / (kp * v_pancreas_L)
    if k_elim_pancreas > 0:
        t_half_pancreas_h = math.log(2.0) / k_elim_pancreas
    else:
        t_half_pancreas_h = float("inf")

    notes = (
        f"Route={route}; Q_pancreas={q_pancreas_L_per_h} L/h; "
        f"Kp={kp_pancreas}; f_portal={f_portal}; "
        f"CL={cl_L_per_h} L/h; Vd_plasma={vd_plasma_L} L"
    )

    return PancreaticPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_pancreas_mg_L=c_pancreas_list,
        cmax_plasma=cmax_plasma,
        cmax_pancreas=cmax_pancreas,
        auc_plasma=auc_plasma,
        auc_pancreas=auc_pancreas,
        pancreas_plasma_kp=pancreas_plasma_kp,
        portal_delivered_pct=portal_delivered_pct,
        arterial_delivered_pct=arterial_delivered_pct,
        t_half_pancreas_h=t_half_pancreas_h,
        notes=notes,
    )
