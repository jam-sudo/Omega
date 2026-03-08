"""Phase 859 — Urinary Bladder Drug Concentration.

Models drug concentration in urine and bladder wall, relevant for
bladder cancer treatments and UTI drugs.
"""

import math
from dataclasses import dataclass


@dataclass
class BladderPKResult:
    """Result of bladder PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_urine_mg_L: list
    c_bladder_wall_mg_g: list
    kp_bladder_wall: float
    cmax_plasma_mg_L: float
    cmax_urine_mg_L: float
    cmax_bladder_wall_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_urine_mg_h_per_L: float
    urine_excretion_pct: float
    notes: str


def _trapezoidal_auc(times, values):
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )


def simulate_bladder_pk(
    drug_name,
    dose_mg,
    logP,
    mw_Da,
    fu_plasma,
    cl_sys_L_per_h,
    vd_sys_L,
    fe_urine=0.3,
    urine_volume_L_per_h=0.07,
    t_end_h=24.0,
    dt_h=0.05,
):
    """Simulate drug PK in urine and bladder wall.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    fu_plasma : float
        Fraction unbound in plasma.
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    fe_urine : float
        Fraction excreted unchanged in urine (0-1).
    urine_volume_L_per_h : float
        Urine flow rate (L/h).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Output time step (h).

    Returns
    -------
    BladderPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if not (0 <= fe_urine <= 1):
        raise ValueError("fe_urine must be in [0, 1]")
    if urine_volume_L_per_h <= 0:
        raise ValueError("urine_volume_L_per_h must be positive")

    ka = 1.2  # absorption rate constant /h
    ke = cl_sys_L_per_h / vd_sys_L
    k_void = 0.25  # voiding every 4h
    avg_bladder_volume = 0.3  # L

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka, ke, k_void))

    # Partition coefficient for bladder wall
    kp_bladder_wall = max(0.1, min(5.0, 1.0 + max(0.0, logP) * 0.4))

    # Initial conditions
    a_gut = dose_mg * 0.85  # bioavailable fraction in gut
    c_plasma = 0.0
    c_urine = 0.0

    # Output storage
    times_h = []
    c_plasma_list = []
    c_urine_list = []
    c_bladder_wall_list = []

    t = 0.0
    next_out = 0.0

    while t <= t_end_h + 1e-9:
        # Record output at dt_h intervals
        if t >= next_out - 1e-9:
            c_bw = c_urine * kp_bladder_wall * 0.001
            times_h.append(round(t, 10))
            c_plasma_list.append(c_plasma)
            c_urine_list.append(c_urine)
            c_bladder_wall_list.append(c_bw)
            next_out += dt_h

        # Forward Euler
        da_gut = -ka * a_gut
        dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma
        dc_urine = (fe_urine * cl_sys_L_per_h * c_plasma) / avg_bladder_volume - k_void * c_urine

        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_urine += dc_urine * dt_int

        # Clamp non-negative
        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_urine = max(0.0, c_urine)

        t += dt_int

    # Capture final point if needed
    if len(times_h) == 0 or times_h[-1] < t_end_h - 1e-9:
        c_bw = c_urine * kp_bladder_wall * 0.001
        times_h.append(round(t_end_h, 10))
        c_plasma_list.append(c_plasma)
        c_urine_list.append(c_urine)
        c_bladder_wall_list.append(c_bw)

    cmax_plasma = max(c_plasma_list)
    cmax_urine = max(c_urine_list)
    cmax_bladder_wall = max(c_bladder_wall_list)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_urine = _trapezoidal_auc(times_h, c_urine_list)
    urine_excretion_pct = fe_urine * 100

    return BladderPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_urine_mg_L=c_urine_list,
        c_bladder_wall_mg_g=c_bladder_wall_list,
        kp_bladder_wall=kp_bladder_wall,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_urine_mg_L=cmax_urine,
        cmax_bladder_wall_mg_g=cmax_bladder_wall,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_urine_mg_h_per_L=auc_urine,
        urine_excretion_pct=urine_excretion_pct,
        notes=f"Bladder PK for {drug_name}, fe={fe_urine}",
    )
