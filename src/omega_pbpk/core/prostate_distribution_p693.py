"""Phase 693 — Prostate Drug Distribution model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProstateDistributionResult:
    """Result of prostate drug distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_prostate_mg_g: list = field(default_factory=list)
    kp_prostate: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_prostate_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_prostate_mg_h_per_g: float = 0.0
    prostate_to_plasma_ratio: float = 0.0
    t_half_prostate_h: float = 0.0
    notes: str = ""


def simulate_prostate_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    prostate_weight_g: float = 20.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProstateDistributionResult:
    """Simulate drug distribution to prostate tissue."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if prostate_weight_g <= 0:
        raise ValueError("prostate_weight_g must be positive")

    # Prostate parameters
    q_prostate = 0.18  # L/h

    # Kp calculation
    base_kp = (logP + 1.5) * 0.4 + 0.5
    kp_prostate = max(0.1, min(10.0, base_kp / max(1.0, mw_Da / 350.0)))

    # Volumes and rate constants
    vd_prostate = max(0.005, prostate_weight_g * kp_prostate / 1000.0)
    k_in = q_prostate / vd_sys_L
    k_out = q_prostate / vd_prostate
    t_half_prostate = math.log(2) / k_out

    # Oral absorption
    ka = 1.2  # /h
    f_oral = 0.85
    k_elim = cl_sys_L_per_h / vd_sys_L

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka, k_elim, k_in, k_out))

    # Forward Euler simulation
    n_steps = int(t_end_h / dt_int) + 1
    record_interval = max(1, int(dt_h / dt_int))

    a_gut = dose_mg * f_oral
    a_central = 0.0
    c_prostate = 0.0

    times_h = []
    c_plasma_mg_L = []
    c_prostate_mg_g = []

    for i in range(n_steps):
        t = i * dt_int

        c_plasma = a_central / vd_sys_L
        if c_plasma < 0:
            c_plasma = 0.0
        c_prost_tissue = c_prostate * kp_prostate / 1000.0
        if c_prost_tissue < 0:
            c_prost_tissue = 0.0

        if i % record_interval == 0:
            times_h.append(round(t, 6))
            c_plasma_mg_L.append(c_plasma)
            c_prostate_mg_g.append(c_prost_tissue)

        # Euler steps
        da_gut = -ka * a_gut
        da_central = ka * a_gut - k_elim * a_central
        dc_prostate = k_in * c_plasma - k_out * c_prostate

        a_gut += da_gut * dt_int
        a_central += da_central * dt_int
        c_prostate += dc_prostate * dt_int

    # Cmax
    cmax_plasma = max(c_plasma_mg_L) if c_plasma_mg_L else 0.0
    cmax_prostate = max(c_prostate_mg_g) if c_prostate_mg_g else 0.0

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_prostate = sum(
        0.5 * (c_prostate_mg_g[i] + c_prostate_mg_g[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Ratios
    prostate_to_plasma_ratio = auc_prostate / auc_plasma if auc_plasma > 0 else 0.0

    notes_parts = [
        f"Kp_prostate={kp_prostate:.3f}",
        f"Prostate weight={prostate_weight_g:.1f} g",
        f"Q_prostate={q_prostate:.2f} L/h",
        f"t1/2_prostate={t_half_prostate:.2f} h",
    ]

    return ProstateDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_prostate_mg_g=c_prostate_mg_g,
        kp_prostate=kp_prostate,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_prostate_mg_g=cmax_prostate,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_prostate_mg_h_per_g=auc_prostate,
        prostate_to_plasma_ratio=prostate_to_plasma_ratio,
        t_half_prostate_h=t_half_prostate,
        notes="; ".join(notes_parts),
    )
