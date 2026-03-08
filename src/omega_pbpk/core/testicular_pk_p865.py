"""Phase 865 — Testicular drug distribution PK model.

Models drug distribution to testicular tissue, relevant for male
reproductive toxicology and testicular cancer pharmacology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TesticularPKResult:
    """Result of testicular PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_testis_mg_g: list = field(default_factory=list)
    kp_testis: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_testis_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_testis_mg_h_per_g: float = 0.0
    testis_to_plasma_ratio: float = 0.0
    t_half_testis_h: float = 0.0
    btb_penetration: str = ""
    notes: str = ""


def simulate_testicular_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    testis_weight_g: float = 35.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TesticularPKResult:
    """Simulate drug distribution to testicular tissue.

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
    testis_weight_g : float
        Combined testicular weight (g).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Output time step (h).

    Returns
    -------
    TesticularPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if testis_weight_g <= 0:
        raise ValueError(f"testis_weight_g must be > 0, got {testis_weight_g}")

    # BTB permeability factor
    btb_factor = max(0.05, min(1.0, (logP + 1.0) * 0.3 / max(1.0, mw_Da / 300.0)))

    # Partition coefficient
    base_kp = (logP + 1.0) * 0.3 * btb_factor + 0.3
    kp_testis = max(0.05, min(5.0, base_kp))

    # Testicular volume of distribution
    vd_testis = max(0.005, testis_weight_g * kp_testis / 1000.0)

    # Effective testicular blood flow (reduced by BTB)
    q_testis = 0.15 * btb_factor

    # Rate constants
    k_in = q_testis / vd_sys_L
    k_out = q_testis / vd_testis
    ke = cl_sys_L_per_h / vd_sys_L

    # Half-life in testis
    t_half_testis = math.log(2) / k_out

    # BTB penetration classification
    if btb_factor > 0.6:
        btb_penetration = "high"
    elif btb_factor > 0.2:
        btb_penetration = "moderate"
    else:
        btb_penetration = "low"

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(1.2, ke, k_in, k_out))

    # Oral absorption: 1-cpt model
    ka = 1.2
    a_gut = dose_mg * 0.85

    # Initial conditions
    c_plasma = 0.0
    c_testis = 0.0

    # Simulation
    times = []
    c_plasma_list = []
    c_testis_mg_g_list = []

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
            c_testis_mg_g_list.append(c_testis * kp_testis / 1000.0)
            next_sample += dt_h

        # ODE: forward Euler
        absorption = ka * a_gut / vd_sys_L
        dc_plasma = absorption - ke * c_plasma - k_in * c_plasma + k_out * c_testis
        dc_testis = k_in * c_plasma - k_out * c_testis
        da_gut = -ka * a_gut

        c_plasma = max(0.0, c_plasma + dt_int * dc_plasma)
        c_testis = max(0.0, c_testis + dt_int * dc_testis)
        a_gut = max(0.0, a_gut + dt_int * da_gut)

        t += dt_int

    # Derived metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_testis = max(c_testis_mg_g_list) if c_testis_mg_g_list else 0.0

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_testis = sum(
        0.5 * (c_testis_mg_g_list[i] + c_testis_mg_g_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    testis_to_plasma_ratio = auc_testis / auc_plasma if auc_plasma > 0 else 0.0

    # Notes
    notes_parts = []
    if btb_penetration == "low":
        notes_parts.append("Low BTB penetration limits testicular drug exposure.")
    if kp_testis > 2.0:
        notes_parts.append("High testicular partitioning observed.")
    if mw_Da > 500:
        notes_parts.append("High molecular weight may limit BTB crossing.")
    notes = " ".join(notes_parts) if notes_parts else "Standard testicular distribution."

    return TesticularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_testis_mg_g=c_testis_mg_g_list,
        kp_testis=kp_testis,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_testis_mg_g=cmax_testis,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_testis_mg_h_per_g=auc_testis,
        testis_to_plasma_ratio=testis_to_plasma_ratio,
        t_half_testis_h=t_half_testis,
        btb_penetration=btb_penetration,
        notes=notes,
    )


__all__ = ["TesticularPKResult", "simulate_testicular_pk"]
