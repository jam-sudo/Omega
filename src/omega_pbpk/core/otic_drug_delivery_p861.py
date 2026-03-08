"""Phase 861 — Otic (Ear) Drug Delivery model."""

import math
from dataclasses import dataclass, field


@dataclass
class OticDrugDeliveryResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_ear_canal_mg_mL: list = field(default_factory=list)
    c_middle_ear_mg_L: list = field(default_factory=list)
    c_inner_ear_mg_L: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_ear_canal_mg_mL: float = 0.0
    cmax_middle_ear_mg_L: float = 0.0
    cmax_inner_ear_mg_L: float = 0.0
    auc_middle_ear_mg_h_per_L: float = 0.0
    auc_inner_ear_mg_h_per_L: float = 0.0
    tympanic_membrane_permeability: float = 0.0
    round_window_permeability: float = 0.0
    systemic_absorption_pct: float = 0.0
    notes: str = ""


def simulate_otic_drug_delivery(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ear_canal_volume_mL: float = 0.5,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> OticDrugDeliveryResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if ear_canal_volume_mL <= 0:
        raise ValueError("ear_canal_volume_mL must be positive")

    # Permeabilities
    P_tym = 0.01 * max(0.1, logP + 1.0) / max(1.0, mw_Da / 300.0)
    P_tym = max(0.001, min(0.2, P_tym))

    P_rw = P_tym * 0.3
    P_rw = max(0.0001, min(0.05, P_rw))

    # Rate constants
    k_drain = 0.5
    k_et = 0.1
    k_ie = 0.05
    k_sys = 0.01
    ke = cl_sys_L_per_h / vd_sys_L

    # Volumes
    vol_ec_L = ear_canal_volume_mL / 1000.0
    vol_me_L = 0.0008
    vol_ie_L = 8e-5

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(k_drain + P_tym + k_sys, k_et + P_rw, k_ie, ke))

    # Initial conditions
    C_ec = dose_mg / ear_canal_volume_mL  # mg/mL
    C_me = 0.0  # mg/L
    C_ie = 0.0  # mg/L
    C_plasma = 0.0  # mg/L

    times = [0.0]
    c_ec_list = [C_ec]
    c_me_list = [C_me]
    c_ie_list = [C_ie]
    c_plasma_list = [C_plasma]

    t = 0.0
    while t < t_end_h - 1e-12:
        dt_step = min(dt_int, t_end_h - t)

        dC_ec = -(k_drain + P_tym + k_sys) * C_ec
        dC_me = P_tym * C_ec * vol_ec_L / vol_me_L - (k_et + P_rw) * C_me
        dC_ie = P_rw * C_me * vol_me_L / vol_ie_L - k_ie * C_ie
        dC_plasma = k_sys * C_ec * vol_ec_L / vd_sys_L - ke * C_plasma

        C_ec = max(0.0, C_ec + dC_ec * dt_step)
        C_me = max(0.0, C_me + dC_me * dt_step)
        C_ie = max(0.0, C_ie + dC_ie * dt_step)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_step)

        t += dt_step
        times.append(t)
        c_ec_list.append(C_ec)
        c_me_list.append(C_me)
        c_ie_list.append(C_ie)
        c_plasma_list.append(C_plasma)

    # Compute metrics
    cmax_ec = max(c_ec_list)
    cmax_me = max(c_me_list)
    cmax_ie = max(c_ie_list)

    # Trapezoidal AUC
    auc_me = sum(
        0.5 * (c_me_list[i] + c_me_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_ie = sum(
        0.5 * (c_ie_list[i] + c_ie_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    systemic_absorption_pct = min(
        100.0, auc_plasma * cl_sys_L_per_h / dose_mg * 100
    )

    return OticDrugDeliveryResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_ear_canal_mg_mL=c_ec_list,
        c_middle_ear_mg_L=c_me_list,
        c_inner_ear_mg_L=c_ie_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_ear_canal_mg_mL=cmax_ec,
        cmax_middle_ear_mg_L=cmax_me,
        cmax_inner_ear_mg_L=cmax_ie,
        auc_middle_ear_mg_h_per_L=auc_me,
        auc_inner_ear_mg_h_per_L=auc_ie,
        tympanic_membrane_permeability=P_tym,
        round_window_permeability=P_rw,
        systemic_absorption_pct=systemic_absorption_pct,
        notes="Otic drug delivery: ear canal -> middle ear -> inner ear via tympanic and round window membranes",
    )
