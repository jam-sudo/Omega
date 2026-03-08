"""Phase 863 — Bone Marrow Drug Distribution Model."""

import math
from dataclasses import dataclass, field


@dataclass
class BoneMarrowPKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_marrow_mg_g: list = field(default_factory=list)
    kp_marrow: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_marrow_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_marrow_mg_h_per_g: float = 0.0
    marrow_to_plasma_ratio: float = 0.0
    t_half_marrow_h: float = 0.0
    myelotoxicity_index: float = 0.0
    notes: str = ""


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    marrow_weight_g: float = 2600.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> BoneMarrowPKResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if marrow_weight_g <= 0:
        raise ValueError("marrow_weight_g must be > 0")

    # Partition coefficient
    base_kp = (logP + 2.0) * 0.5 + 0.6
    kp_marrow = max(0.2, min(12.0, base_kp / max(1.0, mw_Da / 350.0)))

    # Volumes and flows
    vd_marrow = max(0.05, marrow_weight_g * kp_marrow / 1000.0)
    q_marrow = 4.5  # L/h

    # Rate constants
    ka = 1.2  # absorption
    ke = cl_sys_L_per_h / vd_sys_L
    k_in = q_marrow / vd_sys_L
    k_out = q_marrow / vd_marrow
    t_half_marrow = math.log(2) / k_out

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(1.2, ke, k_in, k_out))

    # Initial conditions
    a_gut = dose_mg * 0.85
    c_plasma = 0.0
    c_marrow = 0.0

    times_h = []
    c_plasma_list = []
    c_marrow_mg_g_list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_int))

    for _ in range(n_steps + 1):
        c_marrow_tissue = c_marrow * kp_marrow / 1000.0

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_marrow_mg_g_list.append(c_marrow_tissue)

        # Forward Euler
        da_gut = -ka * a_gut
        dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma - k_in * c_plasma + k_out * c_marrow
        dc_marrow = k_in * c_plasma - k_out * c_marrow

        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_marrow += dc_marrow * dt_int

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_marrow = max(0.0, c_marrow)

        t += dt_int

    # Metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_marrow = max(c_marrow_mg_g_list) if c_marrow_mg_g_list else 0.0

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_marrow = sum(
        0.5 * (c_marrow_mg_g_list[i] + c_marrow_mg_g_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    marrow_to_plasma_ratio = auc_marrow / auc_plasma if auc_plasma > 0 else 0.0
    myelotoxicity_index = cmax_marrow / 0.1

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_marrow_mg_g=c_marrow_mg_g_list,
        kp_marrow=kp_marrow,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_marrow_mg_g=cmax_marrow,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_marrow_mg_h_per_g=auc_marrow,
        marrow_to_plasma_ratio=marrow_to_plasma_ratio,
        t_half_marrow_h=t_half_marrow,
        myelotoxicity_index=myelotoxicity_index,
        notes=f"Bone marrow PK for {drug_name}, Kp={kp_marrow:.2f}",
    )
