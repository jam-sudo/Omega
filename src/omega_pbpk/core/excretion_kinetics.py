"""Drug renal and biliary excretion kinetics — Phase 487."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExcretionResult:
    """Result of excretion simulation."""

    drug_name: str
    times_h: list
    c_plasma_mg_L: list
    cumulative_renal_mg: list
    cumulative_biliary_mg: list
    cumulative_metabolized_mg: list
    fe_renal: float
    fe_biliary: float
    fe_metabolic: float
    t_half_h: float
    notes: str


def renal_excretion_rate(
    c_plasma_mg_L: float,
    fu_plasma: float,
    gfr_mL_per_min: float = 120.0,
    cl_sec_mL_per_min: float = 0.0,
    f_reab: float = 0.0,
) -> float:
    """Calculate renal excretion rate (mg/h).

    CLrenal = fu * GFR + CL_secretion - CL_reabsorption (mL/min → L/h)

    Args:
        c_plasma_mg_L: Plasma concentration (mg/L).
        fu_plasma: Unbound fraction in plasma (0-1).
        gfr_mL_per_min: Glomerular filtration rate (mL/min).
        cl_sec_mL_per_min: Active secretion clearance (mL/min).
        f_reab: Fractional tubular reabsorption (0-1).

    Returns:
        Renal excretion rate in mg/h.
    """
    if c_plasma_mg_L < 0:
        raise ValueError("c_plasma_mg_L must be >= 0")
    if not 0.0 <= fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be between 0 and 1")
    if gfr_mL_per_min < 0:
        raise ValueError("gfr_mL_per_min must be >= 0")
    if cl_sec_mL_per_min < 0:
        raise ValueError("cl_sec_mL_per_min must be >= 0")
    if not 0.0 <= f_reab <= 1.0:
        raise ValueError("f_reab must be between 0 and 1")

    # Net renal clearance (mL/min)
    cl_filtration = fu_plasma * gfr_mL_per_min
    cl_secretion = cl_sec_mL_per_min
    cl_reabsorption = f_reab * (cl_filtration + cl_secretion)
    cl_renal_mL_per_min = max(0.0, cl_filtration + cl_secretion - cl_reabsorption)

    # Convert mL/min to L/h: * 60 / 1000
    cl_renal_L_per_h = cl_renal_mL_per_min * 60.0 / 1000.0

    return cl_renal_L_per_h * c_plasma_mg_L


def biliary_excretion_rate(
    c_liver_mg_L: float,
    cl_bil_L_per_h: float = 0.1,
    mw: float = 500.0,
    logP: float = 2.0,
) -> float:
    """Calculate biliary excretion rate (mg/h).

    Enhanced for MW > 500 Da and moderate logP (2-4).

    Args:
        c_liver_mg_L: Liver concentration (mg/L).
        cl_bil_L_per_h: Baseline biliary clearance (L/h).
        mw: Molecular weight (Da).
        logP: Lipophilicity.

    Returns:
        Biliary excretion rate in mg/h.
    """
    if c_liver_mg_L < 0:
        raise ValueError("c_liver_mg_L must be >= 0")
    if cl_bil_L_per_h < 0:
        raise ValueError("cl_bil_L_per_h must be >= 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")

    # Moderate logP enhancement factor
    logp_factor = 1.0 + logP * 0.2
    # MW enhancement
    mw_factor = 1.5 if mw >= 500.0 else 1.0

    cl_bil_effective = cl_bil_L_per_h * logp_factor * mw_factor
    return cl_bil_effective * c_liver_mg_L


def calculate_fe(cl_renal: float, cl_total: float) -> float:
    """Calculate fraction excreted unchanged renally.

    Args:
        cl_renal: Renal clearance (L/h).
        cl_total: Total clearance (L/h).

    Returns:
        Fraction excreted unchanged (0-1).
    """
    if cl_renal < 0:
        raise ValueError("cl_renal must be >= 0")
    if cl_total <= 0:
        raise ValueError("cl_total must be > 0")
    if cl_renal > cl_total:
        raise ValueError("cl_renal cannot exceed cl_total")

    return cl_renal / cl_total


def clearance_from_excretion(
    gfr_mL_per_min: float,
    fu_plasma: float,
    f_secretion: float = 0.0,
    f_reabsorption: float = 0.0,
) -> float:
    """Calculate renal clearance from excretion parameters.

    CLrenal (L/h) = fu * GFR + CL_sec - CL_reab

    Args:
        gfr_mL_per_min: Glomerular filtration rate (mL/min).
        fu_plasma: Unbound fraction in plasma (0-1).
        f_secretion: Fractional secretion contribution (mL/min as fraction of GFR).
        f_reabsorption: Fractional tubular reabsorption (0-1).

    Returns:
        Renal clearance in L/h.
    """
    if gfr_mL_per_min < 0:
        raise ValueError("gfr_mL_per_min must be >= 0")
    if not 0.0 <= fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be between 0 and 1")
    if f_secretion < 0:
        raise ValueError("f_secretion must be >= 0")
    if not 0.0 <= f_reabsorption <= 1.0:
        raise ValueError("f_reabsorption must be between 0 and 1")

    cl_filtration = fu_plasma * gfr_mL_per_min
    cl_sec = f_secretion * gfr_mL_per_min
    cl_reab = f_reabsorption * (cl_filtration + cl_sec)
    cl_renal_mL_per_min = max(0.0, cl_filtration + cl_sec - cl_reab)

    # Convert mL/min → L/h
    return cl_renal_mL_per_min * 60.0 / 1000.0


def simulate_excretion(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    cl_biliary_L_per_h: float,
    cl_metabolic_L_per_h: float,
    vd_L: float = 70.0,
    fu_plasma: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    route: str = "iv",
) -> ExcretionResult:
    """Simulate drug excretion kinetics using forward Euler ODE.

    1-compartment model: dA/dt = -(CL_renal + CL_biliary + CL_metabolic) * C

    Args:
        drug_name: Name of the drug.
        dose_mg: Initial dose (mg).
        cl_renal_L_per_h: Renal clearance (L/h).
        cl_biliary_L_per_h: Biliary clearance (L/h).
        cl_metabolic_L_per_h: Metabolic clearance (L/h).
        vd_L: Volume of distribution (L).
        fu_plasma: Unbound fraction in plasma.
        t_end_h: End time (h).
        dt_h: Time step (h).
        route: Administration route ('iv' or 'oral').

    Returns:
        ExcretionResult with time course and summary metrics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if cl_biliary_L_per_h < 0:
        raise ValueError("cl_biliary_L_per_h must be >= 0")
    if cl_metabolic_L_per_h < 0:
        raise ValueError("cl_metabolic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 < fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    cl_total = cl_renal_L_per_h + cl_biliary_L_per_h + cl_metabolic_L_per_h
    if cl_total <= 0:
        raise ValueError("Total clearance must be > 0")

    # For oral route, assume F=1 for simplicity (absorbed fraction)
    # Simple 1-cpt model: A = amount in body (mg)
    # dA/dt = -CL_total * C = -CL_total * A / Vd

    # Initial conditions
    a_body = dose_mg  # mg in body
    cum_renal = 0.0
    cum_biliary = 0.0
    cum_metabolized = 0.0

    times_h = []
    c_plasma = []
    cum_renal_list = []
    cum_biliary_list = []
    cum_metabolized_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _ in range(n_steps):
        c = a_body / vd_L  # mg/L

        times_h.append(round(t, 6))
        c_plasma.append(c)
        cum_renal_list.append(cum_renal)
        cum_biliary_list.append(cum_biliary)
        cum_metabolized_list.append(cum_metabolized)

        if t >= t_end_h:
            break

        # Rates (mg/h)
        rate_renal = cl_renal_L_per_h * c
        rate_biliary = cl_biliary_L_per_h * c
        rate_metabolic = cl_metabolic_L_per_h * c
        rate_total = rate_renal + rate_biliary + rate_metabolic

        # Forward Euler update
        a_body = max(0.0, a_body - rate_total * dt_h)
        cum_renal += rate_renal * dt_h
        cum_biliary += rate_biliary * dt_h
        cum_metabolized += rate_metabolic * dt_h

        t = round(t + dt_h, 9)

    # Calculate fraction excreted
    total_excreted = cum_renal + cum_biliary + cum_metabolized
    if total_excreted > 0:
        fe_renal = cum_renal / dose_mg
        fe_biliary = cum_biliary / dose_mg
        fe_metabolic = cum_metabolized / dose_mg
    else:
        fe_renal = 0.0
        fe_biliary = 0.0
        fe_metabolic = 0.0

    # Half-life
    t_half = math.log(2.0) * vd_L / cl_total if cl_total > 0 else float("inf")

    notes = (
        f"1-cpt forward Euler simulation; route={route}; "
        f"CL_total={cl_total:.3f} L/h; t½={t_half:.2f} h"
    )

    return ExcretionResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cumulative_renal_mg=cum_renal_list,
        cumulative_biliary_mg=cum_biliary_list,
        cumulative_metabolized_mg=cum_metabolized_list,
        fe_renal=fe_renal,
        fe_biliary=fe_biliary,
        fe_metabolic=fe_metabolic,
        t_half_h=t_half,
        notes=notes,
    )
