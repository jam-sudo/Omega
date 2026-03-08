"""Phase 683 — CSF Drug Penetration model."""

import math
from dataclasses import dataclass, field


@dataclass
class CSFPenetrationResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_csf_mg_L: list = field(default_factory=list)
    csf_to_plasma_ratio: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_csf_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_csf_mg_h_per_L: float = 0.0
    kp_csf: float = 0.0
    t_half_csf_h: float = 0.0
    penetration_class: str = ""
    notes: str = ""


def simulate_csf_penetration(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    psa_A2: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CSFPenetrationResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")

    # Parameters
    v_csf = 0.15  # L
    k_csf_out = 0.03  # /h
    q_csf = 0.35  # L/h

    # Kp_csf from physicochemistry
    base_kp = 1.0 / (1.0 + mw_Da / 500.0) * (1.0 + max(0.0, logP) * 0.3)
    psa_penalty = max(0.0, (psa_A2 - 90.0) / 90.0) * 0.5
    kp_csf = max(0.01, min(1.5, base_kp - psa_penalty))

    # Transfer rates
    k_in_csf = q_csf / vd_sys_L
    k_out_csf = q_csf / (v_csf * kp_csf) + k_csf_out

    # Systemic PK
    ke = cl_sys_L_per_h / vd_sys_L
    ka = 1.2  # /h
    f_oral = 0.85

    # Initial conditions
    a_gut = dose_mg * f_oral
    c_plasma = 0.0
    c_csf = 0.0

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_csf_list = []

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_csf_list.append(c_csf)

        # Forward Euler
        d_a_gut = -ka * a_gut
        d_c_plasma = ka * a_gut / vd_sys_L - ke * c_plasma
        d_c_csf = k_in_csf * c_plasma - k_out_csf * c_csf

        a_gut += d_a_gut * dt_h
        c_plasma += d_c_plasma * dt_h
        c_csf += d_c_csf * dt_h

        # Clamp negatives
        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_csf = max(0.0, c_csf)

    # Metrics
    cmax_plasma = max(c_plasma_list)
    cmax_csf = max(c_csf_list)

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_csf = sum(
        0.5 * (c_csf_list[i] + c_csf_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    csf_to_plasma_ratio = auc_csf / auc_plasma if auc_plasma > 0 else 0.0

    # t_half_csf from k_out_csf
    t_half_csf = math.log(2) / k_out_csf if k_out_csf > 0 else 0.0

    # Penetration class
    if kp_csf > 0.5:
        penetration_class = "high"
    elif kp_csf > 0.1:
        penetration_class = "moderate"
    else:
        penetration_class = "low"

    notes = (
        f"CSF penetration simulation for {drug_name}. "
        f"Kp_csf={kp_csf:.4f}, CSF volume={v_csf} L, "
        f"choroid plexus flow={q_csf} L/h."
    )

    return CSFPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_csf_mg_L=c_csf_list,
        csf_to_plasma_ratio=csf_to_plasma_ratio,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_csf_mg_L=cmax_csf,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_csf_mg_h_per_L=auc_csf,
        kp_csf=kp_csf,
        t_half_csf_h=t_half_csf,
        penetration_class=penetration_class,
        notes=notes,
    )
