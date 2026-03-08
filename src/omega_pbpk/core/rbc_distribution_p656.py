"""Drug accumulation in red blood cells.

Models drug partitioning into RBCs using a 2-compartment (plasma + RBC)
model with oral absorption.  Blood-to-plasma ratio (Rb/p) affects apparent
volume of distribution and blood cell sequestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RBCDistributionResult:
    """Result container for RBC distribution simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_rbc_mg_L: list[float] = field(default_factory=list)
    c_whole_blood_mg_L: list[float] = field(default_factory=list)
    rb_p_ratio: float = 0.0
    kp_rbc: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_rbc_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_rbc_mg_h_per_L: float = 0.0
    rbc_sequestration_pct: float = 0.0
    t_half_blood_h: float = 0.0
    notes: str = ""


def simulate_rbc_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    hematocrit: float = 0.45,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RBCDistributionResult:
    """Simulate drug distribution between plasma and RBCs."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0 <= hematocrit < 1):
        raise ValueError("hematocrit must be in [0, 1)")

    # Kp_RBC from logP and MW
    kp_rbc = 0.3 * 10 ** (logP * 0.3) * (1 - mw_Da / 2000.0)
    kp_rbc = max(0.1, min(10.0, kp_rbc))

    # Blood-to-plasma ratio
    rb_p = 1 - hematocrit + hematocrit * kp_rbc
    rb_p = max(0.3, rb_p)

    # Volumes
    total_blood_vol = 5.0
    v_rbc = total_blood_vol * hematocrit

    # Rate constants
    ka = 1.2
    k_el = cl_sys_L_per_h / vd_sys_L
    q_rbc = 100.0
    k_in_rbc = q_rbc / vd_sys_L
    k_out_rbc = q_rbc / (v_rbc * kp_rbc) if v_rbc * kp_rbc > 0 else 0.0

    # Initial conditions
    a_gut = dose_mg * 0.85
    c_plasma = 0.0
    c_rbc = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_rbc_list: list[float] = []
    c_blood_list: list[float] = []

    # Sub-step for numerical stability with fast RBC equilibration
    max_rate = max(ka, k_el, k_in_rbc, k_out_rbc)
    sub_dt = min(dt_h, 0.4 / max_rate) if max_rate > 0 else dt_h
    n_sub = max(1, int(dt_h / sub_dt))
    sub_dt = dt_h / n_sub

    n_steps = int(t_end_h / dt_h)
    t = 0.0

    for _ in range(n_steps + 1):
        c_blood = c_plasma * (1 - hematocrit) + c_rbc * hematocrit
        times.append(round(t, 6))
        c_plasma_list.append(max(0.0, c_plasma))
        c_rbc_list.append(max(0.0, c_rbc))
        c_blood_list.append(max(0.0, c_blood))

        # Forward Euler with sub-stepping
        for _s in range(n_sub):
            da_gut = -ka * a_gut
            dc_plasma = (
                ka * a_gut / vd_sys_L
                - k_el * c_plasma
                - k_in_rbc * c_plasma
                + k_out_rbc * c_rbc * (v_rbc / vd_sys_L)
            )
            dc_rbc = (
                k_in_rbc * c_plasma * vd_sys_L / v_rbc - k_out_rbc * c_rbc if v_rbc > 0 else 0.0
            )

            a_gut += da_gut * sub_dt
            c_plasma += dc_plasma * sub_dt
            c_rbc += dc_rbc * sub_dt

            a_gut = max(0.0, a_gut)
            c_plasma = max(0.0, c_plasma)
            c_rbc = max(0.0, c_rbc)

        t += dt_h

    # Metrics
    cmax_plasma = max(c_plasma_list)
    cmax_rbc = max(c_rbc_list)

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_rbc = sum(
        0.5 * (c_rbc_list[i] + c_rbc_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # RBC sequestration
    rbc_seq = (cmax_rbc * v_rbc) / (dose_mg * 0.85) * 100.0
    rbc_seq = max(0.0, min(100.0, rbc_seq))

    t_half = math.log(2) / k_el

    notes = f"Kp_RBC={kp_rbc:.3f}, Rb/p={rb_p:.3f}, hematocrit={hematocrit:.2f}."

    return RBCDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_rbc_mg_L=c_rbc_list,
        c_whole_blood_mg_L=c_blood_list,
        rb_p_ratio=rb_p,
        kp_rbc=kp_rbc,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_rbc_mg_L=cmax_rbc,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_rbc_mg_h_per_L=auc_rbc,
        rbc_sequestration_pct=rbc_seq,
        t_half_blood_h=t_half,
        notes=notes,
    )
