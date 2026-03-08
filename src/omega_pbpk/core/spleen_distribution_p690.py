"""Phase 690 — Spleen Drug Distribution model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SpleenDistributionResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_spleen_mg_g: list = field(default_factory=list)
    kp_spleen: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_spleen_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_spleen_mg_h_per_g: float = 0.0
    spleen_to_plasma_ratio: float = 0.0
    t_half_spleen_h: float = 0.0
    red_pulp_fraction: float = 0.0
    notes: str = ""


def simulate_spleen_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    spleen_weight_g: float = 150.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SpleenDistributionResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if spleen_weight_g <= 0:
        raise ValueError("spleen_weight_g must be > 0")

    # PK parameters
    ka = 1.2
    k_elim = cl_sys_L_per_h / vd_sys_L
    q_spleen = 2.4  # splenic blood flow L/h
    red_pulp_fraction = 0.75

    # Kp calculation
    base_kp = (logP + 2.0) * 0.45 + 0.6
    kp_spleen = max(0.2, min(12.0, base_kp / max(1.0, mw_Da / 350.0)))

    # Spleen volume of distribution
    vd_spleen = max(0.02, spleen_weight_g * kp_spleen / 1000.0)

    # Rate constants
    k_in = q_spleen / vd_sys_L
    k_out = q_spleen / vd_spleen
    t_half_spleen = math.log(2) / k_out

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka, k_elim, k_in, k_out))

    # Initial conditions
    a_gut = dose_mg * 0.85
    c_plasma = 0.0
    c_spleen = 0.0
    t = 0.0

    times = [0.0]
    c_plasma_list = [0.0]
    c_spleen_list = [0.0]

    n_steps = int(math.ceil(t_end_h / dt_int))
    for _ in range(n_steps):
        # Derivatives
        da_gut = -ka * a_gut
        dc_plasma = (
            (ka * a_gut / vd_sys_L)
            - k_elim * c_plasma
            - k_in * c_plasma
            + k_out * c_spleen * (vd_spleen / vd_sys_L)
        )
        dc_spleen = k_in * c_plasma * (vd_sys_L / vd_spleen) - k_out * c_spleen

        # Forward Euler
        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_spleen += dc_spleen * dt_int
        t += dt_int

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_spleen = max(0.0, c_spleen)

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_spleen_list.append(c_spleen * kp_spleen / 1000.0)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_spleen = max(c_spleen_list)

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_spleen = sum(
        0.5 * (c_spleen_list[i] + c_spleen_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    spleen_to_plasma_ratio = auc_spleen / auc_plasma if auc_plasma > 0 else 0.0

    return SpleenDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_g=c_spleen_list,
        kp_spleen=kp_spleen,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_spleen_mg_g=cmax_spleen,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_spleen_mg_h_per_g=auc_spleen,
        spleen_to_plasma_ratio=spleen_to_plasma_ratio,
        t_half_spleen_h=t_half_spleen,
        red_pulp_fraction=red_pulp_fraction,
        notes="Spleen distribution simulation using 1-compartment oral model with splenic blood flow.",
    )
