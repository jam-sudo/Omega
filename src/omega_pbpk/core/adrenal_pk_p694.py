"""Phase 694 — Adrenal Gland Drug Accumulation model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AdrenalPKResult:
    """Result of adrenal gland drug accumulation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_adrenal_mg_g: list = field(default_factory=list)
    kp_adrenal: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_adrenal_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_adrenal_mg_h_per_g: float = 0.0
    adrenal_to_plasma_ratio: float = 0.0
    t_half_adrenal_h: float = 0.0
    cortex_vs_medulla: str = ""
    notes: str = ""


def simulate_adrenal_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    adrenal_weight_g: float = 14.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> AdrenalPKResult:
    """Simulate drug accumulation in adrenal glands."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if adrenal_weight_g <= 0:
        raise ValueError("adrenal_weight_g must be positive")

    # Cortex vs medulla determination
    cortex_vs_medulla = "cortex" if logP > 1.5 else "medulla"

    # Adrenal parameters
    q_adrenal = 0.6  # L/h (very high perfusion)

    # Kp calculation (adrenals have very high lipid content)
    base_kp = (logP + 2.5) * 0.6 + 0.8
    kp_adrenal = max(0.3, min(20.0, base_kp / max(1.0, mw_Da / 300.0)))

    # Volumes and rate constants
    vd_adrenal = max(0.005, adrenal_weight_g * kp_adrenal / 1000.0)
    k_in = q_adrenal / vd_sys_L
    k_out = q_adrenal / vd_adrenal
    t_half_adrenal = math.log(2) / k_out

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
    c_adrenal = 0.0

    times_h = []
    c_plasma_mg_L = []
    c_adrenal_mg_g = []

    for i in range(n_steps):
        t = i * dt_int

        c_plasma = a_central / vd_sys_L
        if c_plasma < 0:
            c_plasma = 0.0
        c_adr_tissue = c_adrenal * kp_adrenal / 1000.0
        if c_adr_tissue < 0:
            c_adr_tissue = 0.0

        if i % record_interval == 0:
            times_h.append(round(t, 6))
            c_plasma_mg_L.append(c_plasma)
            c_adrenal_mg_g.append(c_adr_tissue)

        # Euler steps
        da_gut = -ka * a_gut
        da_central = ka * a_gut - k_elim * a_central
        dc_adrenal = k_in * c_plasma - k_out * c_adrenal

        a_gut += da_gut * dt_int
        a_central += da_central * dt_int
        c_adrenal += dc_adrenal * dt_int

    # Cmax
    cmax_plasma = max(c_plasma_mg_L) if c_plasma_mg_L else 0.0
    cmax_adrenal = max(c_adrenal_mg_g) if c_adrenal_mg_g else 0.0

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_adrenal = sum(
        0.5 * (c_adrenal_mg_g[i] + c_adrenal_mg_g[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Ratios
    adrenal_to_plasma_ratio = auc_adrenal / auc_plasma if auc_plasma > 0 else 0.0

    notes_parts = [
        f"Kp_adrenal={kp_adrenal:.3f}",
        f"Adrenal weight={adrenal_weight_g:.1f} g",
        f"Q_adrenal={q_adrenal:.2f} L/h",
        f"t1/2_adrenal={t_half_adrenal:.2f} h",
        f"Preferred zone: {cortex_vs_medulla}",
    ]

    return AdrenalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_adrenal_mg_g=c_adrenal_mg_g,
        kp_adrenal=kp_adrenal,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adrenal_mg_g=cmax_adrenal,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_adrenal_mg_h_per_g=auc_adrenal,
        adrenal_to_plasma_ratio=adrenal_to_plasma_ratio,
        t_half_adrenal_h=t_half_adrenal,
        cortex_vs_medulla=cortex_vs_medulla,
        notes="; ".join(notes_parts),
    )
