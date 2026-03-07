"""
Phase 201 — Adipose Tissue PK
==============================
2-compartment model for drug distribution into adipose tissue.
Critical for lipophilic drugs with prolonged action.

Pure Python (stdlib: math, dataclasses) — no numpy, no scipy.
Forward Euler ODE; manual trapz for AUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdiposePKResult:
    """Result of adipose tissue PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    fat_fraction: float
    body_weight_kg: float
    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_kg: list
    cmax_adipose_mg_kg: float
    tmax_adipose_h: float
    auc_adipose: float
    auc_plasma: float
    adipose_to_plasma_ratio: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i - 1] + values[i]) * dt
    return total


def _compute_k_pa(logP: float) -> float:
    """Plasma-to-adipose transfer rate constant (1/h)."""
    if logP > 0:
        logP_factor = max(0.1, logP / 3.0)
    else:
        logP_factor = 0.02 / 0.05  # results in k_pa = 0.05 * 0.02/0.05 = 0.02
    return 0.05 * logP_factor


def simulate_adipose_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    logP: float,
    fat_fraction: float = 0.25,
    body_weight_kg: float = 70.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> AdiposePKResult:
    """
    Simulate adipose tissue PK using a 2-compartment Forward Euler ODE.

    Parameters
    ----------
    drug_name : str
    dose_mg : float            IV bolus dose in mg
    cl_L_per_h : float        Plasma clearance (L/h)
    vd_plasma_L : float       Volume of distribution — plasma compartment (L)
    logP : float               Octanol-water partition coefficient
    fat_fraction : float       Fraction of body that is fat (0, 1)
    body_weight_kg : float     Body weight in kg (>0)
    t_end_h : float            Simulation end time (h)
    dt_h : float               Forward Euler step size (h)

    Returns
    -------
    AdiposePKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 < fat_fraction < 1.0):
        raise ValueError(f"fat_fraction must be in (0, 1), got {fat_fraction}")
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")

    fat_mass_kg = fat_fraction * body_weight_kg

    # Rate constants
    ke = cl_L_per_h / vd_plasma_L  # elimination rate (1/h)
    k_pa = _compute_k_pa(logP)  # plasma→adipose (1/h)
    k_ap = 0.02  # adipose→plasma (1/h) — slow release

    # Initial conditions: IV bolus
    C_plasma = dose_mg / vd_plasma_L  # mg/L
    C_adipose = 0.0  # mg/kg

    n_steps = int(math.ceil(t_end_h / dt_h))
    times = []
    c_plasma_list = []
    c_adipose_list = []

    t = 0.0
    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_adipose_list.append(C_adipose)

        # ODEs
        dC_plasma_dt = (
            -ke * C_plasma
            - k_pa * C_plasma * fat_fraction
            + k_ap * C_adipose * fat_mass_kg / vd_plasma_L
        )
        dC_adipose_dt = k_pa * C_plasma * vd_plasma_L / fat_mass_kg - k_ap * C_adipose

        # Forward Euler
        C_plasma = max(0.0, C_plasma + dC_plasma_dt * dt_h)
        C_adipose = max(0.0, C_adipose + dC_adipose_dt * dt_h)
        t += dt_h

    # Derived metrics
    cmax_adipose = max(c_adipose_list)
    tmax_adipose = times[c_adipose_list.index(cmax_adipose)]
    auc_adipose = _trapz(times, c_adipose_list)
    auc_plasma = _trapz(times, c_plasma_list)

    if auc_plasma > 0:
        adipose_to_plasma_ratio = auc_adipose / auc_plasma
    else:
        adipose_to_plasma_ratio = 0.0

    notes = f"k_pa={k_pa:.4f}/h, k_ap={k_ap:.4f}/h, ke={ke:.4f}/h, fat_mass={fat_mass_kg:.1f}kg"

    return AdiposePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        fat_fraction=fat_fraction,
        body_weight_kg=body_weight_kg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_adipose_mg_kg=c_adipose_list,
        cmax_adipose_mg_kg=cmax_adipose,
        tmax_adipose_h=tmax_adipose,
        auc_adipose=auc_adipose,
        auc_plasma=auc_plasma,
        adipose_to_plasma_ratio=adipose_to_plasma_ratio,
        notes=notes,
    )


def assess_accumulation(result: AdiposePKResult) -> dict:
    """
    Assess adipose accumulation risk from a simulation result.

    Returns a dict with:
    - accumulation_index: AUC_adipose / AUC_plasma ratio
    - washout_t50_h: estimated time for adipose concentration to halve
                      from Cmax (based on k_ap = 0.02/h → t½ = ln2/k_ap)
    - accumulation_risk: "low", "moderate", or "high"
    """
    k_ap = 0.02  # fixed in model
    washout_t50_h = math.log(2) / k_ap

    accumulation_index = result.adipose_to_plasma_ratio

    if accumulation_index < 1.0:
        risk = "low"
    elif accumulation_index < 5.0:
        risk = "moderate"
    else:
        risk = "high"

    return {
        "accumulation_index": accumulation_index,
        "washout_t50_h": washout_t50_h,
        "accumulation_risk": risk,
    }
