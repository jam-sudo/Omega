"""Phase 705 — Renal Cortex Drug Concentration."""

import math
from dataclasses import dataclass, field


@dataclass
class RenalCortexPKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_cortex_mg_g: list = field(default_factory=list)
    kp_cortex: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_cortex_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_cortex_mg_h_per_g: float = 0.0
    cortex_to_plasma_ratio: float = 0.0
    t_half_cortex_h: float = 0.0
    nephrotoxicity_index: float = 0.0
    notes: str = ""


def simulate_renal_cortex_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    cortex_weight_g: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> RenalCortexPKResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if cortex_weight_g <= 0:
        raise ValueError("cortex_weight_g must be > 0")

    ka = 1.2
    f_oral = 0.85
    ke = cl_sys_L_per_h / vd_sys_L

    base_kp = max(0.5, (logP + 1.0) * 0.5 + 1.0)
    kp_cortex = max(0.3, min(15.0, base_kp / max(1.0, mw_Da / 300.0)))

    vd_cortex = max(0.01, cortex_weight_g * kp_cortex / 1000.0)
    q_cortex = 12.0
    k_in = q_cortex / vd_sys_L
    k_out = q_cortex / vd_cortex
    t_half_cortex = math.log(2) / k_out

    dt_int = min(dt_h, 0.4 / max(1.2, ke, k_in, k_out))

    n_steps = int(math.ceil(t_end_h / dt_int))
    a_gut = dose_mg * f_oral
    c_plasma = 0.0
    c_cortex = 0.0

    times = []
    c_plasma_list = []
    c_cortex_mg_g_list = []

    for i in range(n_steps + 1):
        t = i * dt_int
        cortex_conc_mg_g = c_cortex * kp_cortex / 1000.0

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_cortex_mg_g_list.append(cortex_conc_mg_g)

        if i < n_steps:
            da_gut = -ka * a_gut * dt_int
            dc_plasma = (
                ka * a_gut / vd_sys_L - ke * c_plasma - k_in * c_plasma + k_out * c_cortex
            ) * dt_int
            dc_cortex = (k_in * c_plasma - k_out * c_cortex) * dt_int

            a_gut += da_gut
            c_plasma = max(0.0, c_plasma + dc_plasma)
            c_cortex = max(0.0, c_cortex + dc_cortex)

    cmax_plasma = max(c_plasma_list)
    cmax_cortex = max(c_cortex_mg_g_list)

    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_cortex = sum(
        0.5 * (c_cortex_mg_g_list[i] + c_cortex_mg_g_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    cortex_to_plasma_ratio = auc_cortex / auc_plasma if auc_plasma > 0 else 0.0
    nephrotoxicity_index = cmax_cortex / 1.0

    # Downsample to match dt_h
    step_ratio = max(1, int(round(dt_h / dt_int)))
    out_times = [times[i] for i in range(0, len(times), step_ratio)]
    out_plasma = [c_plasma_list[i] for i in range(0, len(times), step_ratio)]
    out_cortex = [c_cortex_mg_g_list[i] for i in range(0, len(times), step_ratio)]

    return RenalCortexPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=out_times,
        c_plasma_mg_L=out_plasma,
        c_cortex_mg_g=out_cortex,
        kp_cortex=kp_cortex,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_cortex_mg_g=cmax_cortex,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_cortex_mg_h_per_g=auc_cortex,
        cortex_to_plasma_ratio=cortex_to_plasma_ratio,
        t_half_cortex_h=t_half_cortex,
        nephrotoxicity_index=nephrotoxicity_index,
        notes="Renal cortex PK simulation (Phase 705)",
    )
