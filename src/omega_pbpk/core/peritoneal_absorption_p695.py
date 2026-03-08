"""Phase 695 — Peritoneal Drug Absorption model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PeritonealAbsorptionResult:
    """Result of peritoneal drug absorption simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_peritoneal_mg_L: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_tumor_mg_g: list = field(default_factory=list)
    peritoneal_volume_L: float = 0.0
    cmax_peritoneal_mg_L: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_tumor_mg_g: float = 0.0
    auc_peritoneal_mg_h_per_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    peritoneal_advantage: float = 0.0
    drug_exposure_ratio: float = 0.0
    notes: str = ""


def simulate_peritoneal_absorption(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    peritoneal_volume_L: float = 2.0,
    tumor_mass_g: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> PeritonealAbsorptionResult:
    """Simulate drug absorption from the peritoneal cavity."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if peritoneal_volume_L <= 0:
        raise ValueError("peritoneal_volume_L must be positive")
    if tumor_mass_g <= 0:
        raise ValueError("tumor_mass_g must be positive")

    # Peritoneal absorption rate
    ka_peri = 0.1 * (1.0 - mw_Da / 5000.0) * (1.0 + max(0.0, logP) * 0.1)
    ka_peri = max(0.01, min(0.5, ka_peri))

    # Elimination rate
    ke = cl_sys_L_per_h / vd_sys_L

    # Tumor parameters
    kp_tumor = max(0.5, min(5.0, 0.8 + logP * 0.3))
    vd_tumor = max(0.005, tumor_mass_g * kp_tumor / 1000.0)
    q_tumor = 0.05  # L/h
    k_in_t = q_tumor / vd_sys_L
    k_out_t = q_tumor / vd_tumor

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka_peri, ke, k_in_t, k_out_t))

    # Initial conditions
    c_peri = dose_mg / peritoneal_volume_L
    c_plasma = 0.0
    c_tumor = 0.0

    n_steps = int(t_end_h / dt_int) + 1
    record_interval = max(1, int(dt_h / dt_int))

    times_h = []
    c_peritoneal_mg_L = []
    c_plasma_mg_L = []
    c_tumor_mg_g = []

    for i in range(n_steps):
        t = i * dt_int

        c_peri_safe = max(0.0, c_peri)
        c_plasma_safe = max(0.0, c_plasma)
        c_tumor_tissue = max(0.0, c_tumor * kp_tumor / 1000.0)

        if i % record_interval == 0:
            times_h.append(round(t, 6))
            c_peritoneal_mg_L.append(c_peri_safe)
            c_plasma_mg_L.append(c_plasma_safe)
            c_tumor_mg_g.append(c_tumor_tissue)

        # Forward Euler
        dc_peri = -ka_peri * c_peri
        dc_plasma = ka_peri * c_peri * peritoneal_volume_L / vd_sys_L - ke * c_plasma
        dc_tumor = k_in_t * c_plasma - k_out_t * c_tumor

        c_peri += dc_peri * dt_int
        c_plasma += dc_plasma * dt_int
        c_tumor += dc_tumor * dt_int

    # Cmax
    cmax_peritoneal = max(c_peritoneal_mg_L) if c_peritoneal_mg_L else 0.0
    cmax_plasma = max(c_plasma_mg_L) if c_plasma_mg_L else 0.0
    cmax_tumor = max(c_tumor_mg_g) if c_tumor_mg_g else 0.0

    # Trapezoidal AUC
    auc_peritoneal = sum(
        0.5 * (c_peritoneal_mg_L[i] + c_peritoneal_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Ratios
    peritoneal_advantage = auc_peritoneal / auc_plasma if auc_plasma > 0 else 0.0
    drug_exposure_ratio = peritoneal_advantage

    notes_parts = [
        f"ka_peri={ka_peri:.4f} /h",
        f"peritoneal_volume={peritoneal_volume_L:.2f} L",
        f"tumor_mass={tumor_mass_g:.1f} g",
        f"kp_tumor={kp_tumor:.3f}",
    ]

    return PeritonealAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_peritoneal_mg_L=c_peritoneal_mg_L,
        c_plasma_mg_L=c_plasma_mg_L,
        c_tumor_mg_g=c_tumor_mg_g,
        peritoneal_volume_L=peritoneal_volume_L,
        cmax_peritoneal_mg_L=cmax_peritoneal,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_tumor_mg_g=cmax_tumor,
        auc_peritoneal_mg_h_per_L=auc_peritoneal,
        auc_plasma_mg_h_per_L=auc_plasma,
        peritoneal_advantage=peritoneal_advantage,
        drug_exposure_ratio=drug_exposure_ratio,
        notes="; ".join(notes_parts),
    )
