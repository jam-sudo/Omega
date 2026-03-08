"""Phase 689 — Lymph Node Drug Concentration model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LymphNodePKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_lymph_node_mg_g: list = field(default_factory=list)
    kp_lymph_node: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_lymph_node_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_lymph_node_mg_h_per_g: float = 0.0
    lymph_to_plasma_ratio: float = 0.0
    t_half_lymph_h: float = 0.0
    notes: str = ""


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    lymph_node_mass_g: float = 30.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> LymphNodePKResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if lymph_node_mass_g <= 0:
        raise ValueError("lymph_node_mass_g must be > 0")

    # PK parameters
    ka = 1.2  # absorption rate constant
    k_elim = cl_sys_L_per_h / vd_sys_L
    q_lymph = 1.5  # lymph flow L/h

    # Kp calculation
    base_kp = (max(0.0, logP) + 1.0) * 0.6 + 0.8
    kp = max(0.2, min(15.0, base_kp / max(1.0, mw_Da / 300.0)))

    # Lymph node volume of distribution
    vd_lymph = max(0.02, lymph_node_mass_g * kp / 1000.0)

    # Rate constants for lymph node
    k_in = q_lymph / vd_sys_L
    k_out = q_lymph / vd_lymph
    t_half_lymph = math.log(2) / k_out

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka, k_elim, k_in, k_out))

    # Initial conditions
    a_gut = dose_mg * 0.85
    c_plasma = 0.0
    c_lymph = 0.0
    t = 0.0

    times = [0.0]
    c_plasma_list = [0.0]
    c_lymph_node_list = [0.0]

    n_steps = int(math.ceil(t_end_h / dt_int))
    for _ in range(n_steps):
        # Derivatives
        da_gut = -ka * a_gut
        dc_plasma = (
            (ka * a_gut / vd_sys_L)
            - k_elim * c_plasma
            - k_in * c_plasma
            + k_out * c_lymph * (vd_lymph / vd_sys_L)
        )
        dc_lymph = k_in * c_plasma * (vd_sys_L / vd_lymph) - k_out * c_lymph

        # Forward Euler
        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_lymph += dc_lymph * dt_int
        t += dt_int

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_lymph = max(0.0, c_lymph)

        times.append(t)
        c_plasma_list.append(c_plasma)
        # Convert to mg/g
        c_lymph_node_list.append(c_lymph * kp / 1000.0)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_lymph = max(c_lymph_node_list)

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_lymph = sum(
        0.5 * (c_lymph_node_list[i] + c_lymph_node_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Lymph to plasma ratio
    lymph_to_plasma_ratio = auc_lymph / auc_plasma if auc_plasma > 0 else 0.0

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_lymph_node_mg_g=c_lymph_node_list,
        kp_lymph_node=kp,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_lymph_node_mg_g=cmax_lymph,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_lymph_node_mg_h_per_g=auc_lymph,
        lymph_to_plasma_ratio=lymph_to_plasma_ratio,
        t_half_lymph_h=t_half_lymph,
        notes="Lymph node PK simulation using 1-compartment oral model with lymph node distribution.",
    )
