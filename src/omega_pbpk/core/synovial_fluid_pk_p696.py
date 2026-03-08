"""Phase 696 — Synovial Fluid Drug Distribution model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

VALID_JOINTS = {"knee", "hip", "wrist", "ankle"}

SYNOVIAL_VOLUMES_ML = {
    "knee": 1.1,
    "hip": 0.5,
    "wrist": 0.3,
    "ankle": 0.4,
}


@dataclass
class SynovialFluidPKResult:
    """Result of synovial fluid PK simulation."""

    drug_name: str
    dose_mg: float
    joint: str = ""
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_synovial_mg_L: list = field(default_factory=list)
    synovial_volume_mL: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_synovial_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_synovial_mg_h_per_L: float = 0.0
    synovial_to_plasma_ratio: float = 0.0
    t_half_synovial_h: float = 0.0
    penetration_index: float = 0.0
    notes: str = ""


def simulate_synovial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    joint: str = "knee",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SynovialFluidPKResult:
    """Simulate drug distribution to synovial fluid in joints."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if joint not in VALID_JOINTS:
        raise ValueError(f"joint must be one of {sorted(VALID_JOINTS)}")

    synovial_volume_mL = SYNOVIAL_VOLUMES_ML[joint]
    # Transfer rates
    k_transfer = 0.3 * fu_plasma / max(1.0, mw_Da / 300.0)
    k_transfer = max(0.01, min(1.0, k_transfer))
    k_out_sf = 0.15  # /h

    # PK parameters
    ka = 1.2  # /h oral absorption
    f_oral = 0.85
    ke = cl_sys_L_per_h / vd_sys_L

    # Derived
    t_half_synovial = math.log(2) / k_out_sf
    penetration_index = k_transfer / (k_transfer + k_out_sf)

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka, ke, k_transfer, k_out_sf))

    # Initial conditions
    a_gut = dose_mg * f_oral
    a_central = 0.0
    c_sf = 0.0

    n_steps = int(t_end_h / dt_int) + 1
    record_interval = max(1, int(dt_h / dt_int))

    times_h = []
    c_plasma_mg_L = []
    c_synovial_mg_L = []

    for i in range(n_steps):
        t = i * dt_int

        c_plasma = max(0.0, a_central / vd_sys_L)
        c_sf_safe = max(0.0, c_sf)

        if i % record_interval == 0:
            times_h.append(round(t, 6))
            c_plasma_mg_L.append(c_plasma)
            c_synovial_mg_L.append(c_sf_safe)

        # Forward Euler
        da_gut = -ka * a_gut
        da_central = ka * a_gut - ke * a_central
        dc_sf = k_transfer * c_plasma - k_out_sf * c_sf

        a_gut += da_gut * dt_int
        a_central += da_central * dt_int
        c_sf += dc_sf * dt_int

    # Cmax
    cmax_plasma = max(c_plasma_mg_L) if c_plasma_mg_L else 0.0
    cmax_synovial = max(c_synovial_mg_L) if c_synovial_mg_L else 0.0

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_synovial = sum(
        0.5 * (c_synovial_mg_L[i] + c_synovial_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    synovial_to_plasma_ratio = auc_synovial / auc_plasma if auc_plasma > 0 else 0.0

    notes_parts = [
        f"joint={joint}",
        f"synovial_volume={synovial_volume_mL:.1f} mL",
        f"k_transfer={k_transfer:.4f} /h",
        f"k_out_sf={k_out_sf:.2f} /h",
        f"t1/2_synovial={t_half_synovial:.2f} h",
        f"penetration_index={penetration_index:.4f}",
    ]

    return SynovialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        joint=joint,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_synovial_mg_L=c_synovial_mg_L,
        synovial_volume_mL=synovial_volume_mL,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_synovial_mg_L=cmax_synovial,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_synovial_mg_h_per_L=auc_synovial,
        synovial_to_plasma_ratio=synovial_to_plasma_ratio,
        t_half_synovial_h=t_half_synovial,
        penetration_index=penetration_index,
        notes="; ".join(notes_parts),
    )
