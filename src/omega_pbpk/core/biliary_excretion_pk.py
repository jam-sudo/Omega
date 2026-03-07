"""Biliary excretion PK model (Phase 729).

Models drug secretion into bile via active transporters (MRP2/BSEP),
alongside metabolic and renal elimination. High-MW compounds (>500 Da)
and amphiphilic drugs preferentially undergo biliary excretion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BiliaryExcretionResult:
    """Result of biliary excretion PK simulation."""

    drug_name: str
    dose_mg: float
    mw_da: float
    times_h: list
    c_plasma_mg_L: list
    a_bile_mg: list
    cmax_mg_L: float
    t_half_h: float
    auc_mg_h_per_L: float
    cl_biliary_L_per_h: float
    cl_total_L_per_h: float
    fe_biliary: float
    fe_renal: float
    fe_metabolic: float
    cumulative_bile_mg: float
    notes: str


def estimate_biliary_cl(mw_da: float, cl_biliary_base_L_per_h: float) -> float:
    """Estimate biliary clearance based on molecular weight.

    High MW compounds (>500 Da) have higher biliary clearance; small molecules
    have minimal biliary secretion (10% of base).

    Args:
        mw_da: Molecular weight in Daltons.
        cl_biliary_base_L_per_h: Base biliary CL for MW > 500 Da compounds.

    Returns:
        Estimated biliary clearance in L/h.
    """
    if mw_da > 500.0:
        return cl_biliary_base_L_per_h * math.sqrt(mw_da / 500.0)
    else:
        return cl_biliary_base_L_per_h * 0.1


def simulate_biliary_excretion_pk(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    mw_da: float = 400.0,
    cl_metabolic_L_per_h: float = 3.0,
    cl_renal_L_per_h: float = 1.0,
    cl_biliary_base_L_per_h: float = 1.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> BiliaryExcretionResult:
    """Simulate biliary excretion PK using a 2-compartment model with bile compartment.

    Models plasma concentration and biliary accumulation via forward Euler integration.
    IV bolus administration is assumed.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        mw_da: Molecular weight in Daltons (affects biliary CL).
        cl_metabolic_L_per_h: Metabolic clearance in L/h.
        cl_renal_L_per_h: Renal clearance in L/h.
        cl_biliary_base_L_per_h: Base biliary clearance for MW > 500 Da.
        vd_L: Volume of distribution in L.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        BiliaryExcretionResult with full PK profiles and derived metrics.

    Raises:
        ValueError: If dose_mg <= 0, vd_L <= 0, any CL < 0, or all CLs are zero.
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_metabolic_L_per_h < 0.0:
        raise ValueError(f"cl_metabolic_L_per_h must be >= 0, got {cl_metabolic_L_per_h}")
    if cl_renal_L_per_h < 0.0:
        raise ValueError(f"cl_renal_L_per_h must be >= 0, got {cl_renal_L_per_h}")
    if cl_biliary_base_L_per_h < 0.0:
        raise ValueError(f"cl_biliary_base_L_per_h must be >= 0, got {cl_biliary_base_L_per_h}")

    # Estimate actual biliary CL based on MW
    cl_biliary = estimate_biliary_cl(mw_da, cl_biliary_base_L_per_h)

    cl_total = cl_metabolic_L_per_h + cl_renal_L_per_h + cl_biliary
    if cl_total <= 0.0:
        raise ValueError("At least one clearance term must be > 0")

    # Elimination rate constants
    ke_total = cl_total / vd_L
    k_transit = 0.1  # bile transit rate to intestine (1/h)

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_L  # mg/L
    a_bile = 0.0  # mg

    times_h: list = []
    c_plasma_list: list = []
    a_bile_list: list = []

    t = 0.0
    cumulative_bile_mg = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_bile_list.append(a_bile)

        # Bile secretion rate at this step
        bile_secretion_rate = cl_biliary * c_plasma  # mg/h

        # Forward Euler
        dc_plasma = -ke_total * c_plasma
        da_bile = bile_secretion_rate - k_transit * a_bile

        # Accumulate bile secretion (trapezoidal approximation via rate * dt)
        cumulative_bile_mg += bile_secretion_rate * dt_h

        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)

        t += dt_h

    # Derived metrics
    cmax_mg_L = c_plasma_list[0]  # IV bolus: Cmax at t=0

    # t_half from analytical formula: ln(2) * Vd / CL_total
    t_half_h = math.log(2.0) * vd_L / cl_total

    # AUC via trapezoidal integration
    auc = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * dt

    # Fraction eliminated via each route
    fe_biliary = cl_biliary / cl_total
    fe_renal = cl_renal_L_per_h / cl_total
    fe_metabolic = cl_metabolic_L_per_h / cl_total

    # Notes
    if mw_da > 500.0:
        mw_note = (
            f"High MW ({mw_da:.0f} Da): significant biliary excretion (fe_bil={fe_biliary:.2f})"
        )
    else:
        mw_note = f"Low MW ({mw_da:.0f} Da): minimal biliary excretion (fe_bil={fe_biliary:.3f})"

    notes = (
        f"Biliary excretion PK for {drug_name}. {mw_note}. "
        f"t½={t_half_h:.2f} h, CL_total={cl_total:.2f} L/h, "
        f"fe_met={fe_metabolic:.2f}, fe_ren={fe_renal:.2f}, fe_bil={fe_biliary:.3f}."
    )

    return BiliaryExcretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_da=mw_da,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_bile_mg=a_bile_list,
        cmax_mg_L=cmax_mg_L,
        t_half_h=t_half_h,
        auc_mg_h_per_L=auc,
        cl_biliary_L_per_h=cl_biliary,
        cl_total_L_per_h=cl_total,
        fe_biliary=fe_biliary,
        fe_renal=fe_renal,
        fe_metabolic=fe_metabolic,
        cumulative_bile_mg=cumulative_bile_mg,
        notes=notes,
    )
