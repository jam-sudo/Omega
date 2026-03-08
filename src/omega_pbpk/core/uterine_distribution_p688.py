"""Phase 688 — Uterine Drug Distribution model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class UterineDistributionResult:
    """Result of uterine drug distribution simulation."""

    drug_name: str
    dose_mg: float
    cycle_phase: str = ""
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_uterus_mg_g: list = field(default_factory=list)
    kp_uterus: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_uterus_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_uterus_mg_h_per_g: float = 0.0
    uterus_to_plasma_ratio: float = 0.0
    t_half_uterus_h: float = 0.0
    notes: str = ""


_VALID_PHASES = {"follicular", "luteal", "pregnant"}

_PHASE_BLOOD_FLOW = {
    "follicular": 0.18,
    "luteal": 0.25,
    "pregnant": 0.75,
}

_PHASE_MODIFIER = {
    "follicular": 1.0,
    "luteal": 1.15,
    "pregnant": 1.3,
}


def simulate_uterine_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    cycle_phase: str = "follicular",
    uterus_weight_g: float = 70.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> UterineDistributionResult:
    """Simulate drug distribution to uterine tissue."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if uterus_weight_g <= 0:
        raise ValueError("uterus_weight_g must be positive")
    if cycle_phase not in _VALID_PHASES:
        raise ValueError(f"cycle_phase must be one of {sorted(_VALID_PHASES)}, got '{cycle_phase}'")

    # Uterine parameters
    q_ut = _PHASE_BLOOD_FLOW[cycle_phase]
    phase_mod = _PHASE_MODIFIER[cycle_phase]

    # Kp_uterus
    base_kp = (logP + 1.5) * 0.35 + 0.4
    kp_uterus = max(0.1, min(8.0, base_kp / max(1.0, mw_Da / 350.0) * phase_mod))

    # Volume of distribution for uterus
    vd_uterus = max(0.01, uterus_weight_g * kp_uterus / 1000.0)

    # Rate constants
    ka = 1.2  # absorption rate constant /h
    k_elim = cl_sys_L_per_h / vd_sys_L
    k_in = q_ut / vd_sys_L
    k_out = q_ut / vd_uterus
    t_half_uterus = math.log(2) / k_out

    # Numerical stability
    dt_int = min(dt_h, 0.4 / max(ka, k_elim, k_in, k_out))

    # Initial conditions
    f_oral = 0.85
    a_gut = dose_mg * f_oral
    a_central = 0.0
    a_uterus = 0.0

    # Simulation
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_mg_L = []
    c_uterus_mg_g = []

    cmax_plasma = 0.0
    cmax_uterus = 0.0

    for i in range(n_steps):
        t = i * dt_h
        c_p = max(0.0, a_central / vd_sys_L)
        c_ut = max(0.0, a_uterus / vd_uterus)
        c_ut_mg_g = c_ut * kp_uterus / 1000.0

        times_h.append(round(t, 6))
        c_plasma_mg_L.append(c_p)
        c_uterus_mg_g.append(c_ut_mg_g)

        if c_p > cmax_plasma:
            cmax_plasma = c_p
        if c_ut_mg_g > cmax_uterus:
            cmax_uterus = c_ut_mg_g

        # Forward Euler with internal substeps
        n_sub = max(1, int(math.ceil(dt_h / dt_int)))
        dt_sub = dt_h / n_sub
        for _s in range(n_sub):
            da_gut = -ka * a_gut
            da_central = ka * a_gut - k_elim * a_central - k_in * a_central + k_out * a_uterus
            da_uterus = k_in * a_central - k_out * a_uterus

            a_gut += da_gut * dt_sub
            a_central += da_central * dt_sub
            a_uterus += da_uterus * dt_sub

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_uterus = sum(
        0.5 * (c_uterus_mg_g[i] + c_uterus_mg_g[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Uterus to plasma ratio
    uterus_to_plasma_ratio = auc_uterus / auc_plasma if auc_plasma > 0 else 0.0

    notes = (
        f"Cycle phase: {cycle_phase}; "
        f"Uterus weight: {uterus_weight_g} g; "
        f"Kp: {kp_uterus:.3f}; "
        f"Blood flow: {q_ut} L/h"
    )

    return UterineDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cycle_phase=cycle_phase,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_uterus_mg_g=c_uterus_mg_g,
        kp_uterus=kp_uterus,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_uterus_mg_g=cmax_uterus,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_uterus_mg_h_per_g=auc_uterus,
        uterus_to_plasma_ratio=uterus_to_plasma_ratio,
        t_half_uterus_h=t_half_uterus,
        notes=notes,
    )
