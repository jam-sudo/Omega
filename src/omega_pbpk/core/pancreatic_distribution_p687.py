"""Phase 687 — Pancreatic Drug Distribution model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PancreaticDistributionResult:
    """Result of pancreatic drug distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_pancreas_mg_g: list = field(default_factory=list)
    kp_pancreas: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_pancreas_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_pancreas_mg_h_per_g: float = 0.0
    pancreas_to_plasma_ratio: float = 0.0
    t_half_pancreas_h: float = 0.0
    acinar_vs_islet: str = ""
    notes: str = ""


def simulate_pancreatic_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    pancreas_weight_g: float = 85.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PancreaticDistributionResult:
    """Simulate drug distribution to pancreatic tissue."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if pancreas_weight_g <= 0:
        raise ValueError("pancreas_weight_g must be positive")

    # Pancreatic parameters
    q_panc = 1.1  # L/h

    # Kp_pancreas
    raw_kp = (logP + 2.0) * 0.4 + 0.5
    kp_pancreas = max(0.1, min(10.0, raw_kp / max(1.0, mw_Da / 400.0)))

    # Volume of distribution for pancreas
    vd_pancreas = max(0.01, pancreas_weight_g * kp_pancreas / 1000.0)

    # Rate constants
    ka = 1.2  # absorption rate constant /h
    k_elim = cl_sys_L_per_h / vd_sys_L
    k_in = q_panc / vd_sys_L
    k_out = q_panc / vd_pancreas
    t_half_pancreas = math.log(2) / k_out

    # Numerical stability
    dt_int = min(dt_h, 0.4 / max(ka, k_elim, k_in, k_out))

    # Initial conditions
    f_oral = 0.85
    a_gut = dose_mg * f_oral
    a_central = 0.0
    a_pancreas = 0.0

    # Simulation
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_mg_L = []
    c_pancreas_mg_g = []

    cmax_plasma = 0.0
    cmax_pancreas = 0.0

    for i in range(n_steps):
        t = i * dt_h
        c_p = max(0.0, a_central / vd_sys_L)
        c_panc = max(0.0, a_pancreas / vd_pancreas)
        c_panc_mg_g = c_panc * kp_pancreas / 1000.0

        times_h.append(round(t, 6))
        c_plasma_mg_L.append(c_p)
        c_pancreas_mg_g.append(c_panc_mg_g)

        if c_p > cmax_plasma:
            cmax_plasma = c_p
        if c_panc_mg_g > cmax_pancreas:
            cmax_pancreas = c_panc_mg_g

        # Forward Euler with internal substeps
        n_sub = max(1, int(math.ceil(dt_h / dt_int)))
        dt_sub = dt_h / n_sub
        for _s in range(n_sub):
            da_gut = -ka * a_gut
            da_central = ka * a_gut - k_elim * a_central - k_in * a_central + k_out * a_pancreas
            da_pancreas = k_in * a_central - k_out * a_pancreas

            a_gut += da_gut * dt_sub
            a_central += da_central * dt_sub
            a_pancreas += da_pancreas * dt_sub

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_pancreas = sum(
        0.5 * (c_pancreas_mg_g[i] + c_pancreas_mg_g[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Pancreas to plasma ratio
    pancreas_to_plasma_ratio = auc_pancreas / auc_plasma if auc_plasma > 0 else 0.0

    # Acinar vs islet
    acinar_vs_islet = "acinar" if kp_pancreas < 2.0 else "islet"

    notes = (
        f"Pancreas weight: {pancreas_weight_g} g; "
        f"Kp: {kp_pancreas:.3f}; "
        f"Blood flow: {q_panc} L/h; "
        f"Distribution: {acinar_vs_islet}"
    )

    return PancreaticDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_pancreas_mg_g=c_pancreas_mg_g,
        kp_pancreas=kp_pancreas,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_pancreas_mg_g=cmax_pancreas,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_pancreas_mg_h_per_g=auc_pancreas,
        pancreas_to_plasma_ratio=pancreas_to_plasma_ratio,
        t_half_pancreas_h=t_half_pancreas,
        acinar_vs_islet=acinar_vs_islet,
        notes=notes,
    )
