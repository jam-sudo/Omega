"""Phase 866 — Breast tissue drug distribution PK model.

Models drug distribution to breast tissue, relevant for breast cancer
pharmacology and lactation safety assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BreastTissuePKResult:
    """Result of breast tissue PK simulation."""

    drug_name: str
    dose_mg: float
    lactation: bool
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_breast_mg_g: list = field(default_factory=list)
    c_milk_mg_L: list = field(default_factory=list)
    kp_breast: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_breast_mg_g: float = 0.0
    cmax_milk_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_breast_mg_h_per_g: float = 0.0
    milk_to_plasma_ratio: float = 0.0
    t_half_breast_h: float = 0.0
    notes: str = ""


def simulate_breast_tissue_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    breast_weight_g: float = 500.0,
    lactation: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BreastTissuePKResult:
    """Simulate drug distribution to breast tissue.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    logP : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    breast_weight_g : float
        Combined breast weight (g).
    lactation : bool
        Whether the patient is lactating.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Output time step (h).

    Returns
    -------
    BreastTissuePKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if breast_weight_g <= 0:
        raise ValueError(f"breast_weight_g must be > 0, got {breast_weight_g}")

    # Breast blood flow
    q_breast = 1.5 if lactation else 0.3

    # Partition coefficient (high lipid content in breast)
    base_kp = (logP + 2.0) * 0.5 + 0.5
    kp_breast = max(0.2, min(10.0, base_kp / max(1.0, mw_Da / 350.0)))

    # Breast volume of distribution
    vd_breast = max(0.01, breast_weight_g * kp_breast / 1000.0)

    # Rate constants
    k_in = q_breast / vd_sys_L
    k_out = q_breast / vd_breast
    ke = cl_sys_L_per_h / vd_sys_L

    # Half-life in breast
    t_half_breast = math.log(2) / k_out

    # Milk-to-plasma ratio (only relevant during lactation)
    mp_ratio = max(0.1, min(3.0, 1.0 + logP * 0.2))

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(1.2, ke, k_in, k_out))

    # Oral absorption: 1-cpt model
    ka = 1.2
    a_gut = dose_mg * 0.85

    # Initial conditions
    c_plasma = 0.0
    c_breast = 0.0

    # Simulation
    times = []
    c_plasma_list = []
    c_breast_mg_g_list = []
    c_milk_list = []

    t = 0.0
    next_sample = 0.0
    n_steps = int(t_end_h / dt_int) + 2

    for _ in range(n_steps):
        if t > t_end_h + dt_int:
            break

        # Record at output intervals
        if t >= next_sample - 1e-12:
            times.append(t)
            c_plasma_list.append(c_plasma)
            c_breast_mg_g_list.append(c_breast * kp_breast / 1000.0)
            if lactation:
                c_milk_list.append(c_plasma * mp_ratio)
            else:
                c_milk_list.append(0.0)
            next_sample += dt_h

        # ODE: forward Euler
        absorption = ka * a_gut / vd_sys_L
        dc_plasma = absorption - ke * c_plasma - k_in * c_plasma + k_out * c_breast
        dc_breast = k_in * c_plasma - k_out * c_breast
        da_gut = -ka * a_gut

        c_plasma = max(0.0, c_plasma + dt_int * dc_plasma)
        c_breast = max(0.0, c_breast + dt_int * dc_breast)
        a_gut = max(0.0, a_gut + dt_int * da_gut)

        t += dt_int

    # Derived metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_breast = max(c_breast_mg_g_list) if c_breast_mg_g_list else 0.0
    cmax_milk = max(c_milk_list) if c_milk_list else 0.0

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_breast = sum(
        0.5 * (c_breast_mg_g_list[i] + c_breast_mg_g_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    milk_to_plasma_ratio = mp_ratio if lactation else 0.0

    # Notes
    notes_parts = []
    if lactation:
        notes_parts.append("Lactation increases breast blood flow and enables milk transfer.")
    if kp_breast > 5.0:
        notes_parts.append("High breast tissue partitioning due to lipophilicity.")
    if mw_Da > 500:
        notes_parts.append("High molecular weight may reduce breast tissue uptake.")
    notes = " ".join(notes_parts) if notes_parts else "Standard breast tissue distribution."

    return BreastTissuePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        lactation=lactation,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_breast_mg_g=c_breast_mg_g_list,
        c_milk_mg_L=c_milk_list,
        kp_breast=kp_breast,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_breast_mg_g=cmax_breast,
        cmax_milk_mg_L=cmax_milk,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_breast_mg_h_per_g=auc_breast,
        milk_to_plasma_ratio=milk_to_plasma_ratio,
        t_half_breast_h=t_half_breast,
        notes=notes,
    )


__all__ = ["BreastTissuePKResult", "simulate_breast_tissue_pk"]
