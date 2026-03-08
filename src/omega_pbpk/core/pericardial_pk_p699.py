"""Phase 699 — Pericardial Fluid Drug Distribution."""

import math
from dataclasses import dataclass, field


@dataclass
class PericardialPKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_pericardial_mg_L: list = field(default_factory=list)
    pericardial_volume_mL: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_pericardial_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_pericardial_mg_h_per_L: float = 0.0
    pericardial_to_plasma_ratio: float = 0.0
    t_half_pericardial_h: float = 0.0
    notes: str = ""


def simulate_pericardial_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    pericardial_volume_mL: float = 15.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PericardialPKResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if pericardial_volume_mL <= 0:
        raise ValueError("pericardial_volume_mL must be > 0")

    # Rate constants
    k_peri_in = 0.5 * fu_plasma / max(1.0, mw_Da / 300.0)
    k_peri_in = max(0.01, min(1.0, k_peri_in))
    k_peri_out = 0.5  # /h drainage
    ke = cl_sys_L_per_h / vd_sys_L
    ka = 1.2  # oral absorption rate /h
    f_bio = 0.85

    t_half_pericardial = math.log(2) / k_peri_out

    # Numerical stability
    dt_int = min(dt_h, 0.4 / max(ka, ke, k_peri_in, k_peri_out))

    n_steps = int(t_end_h / dt_int) + 1
    step = dt_int

    # State variables
    a_gut = dose_mg * f_bio
    c_plasma = 0.0
    c_peri = 0.0

    times = []
    cp_list = []
    cperi_list = []

    t = 0.0
    for _i in range(n_steps):
        times.append(t)
        cp_list.append(c_plasma)
        cperi_list.append(max(0.0, c_peri))

        # Euler step
        da_gut = -ka * a_gut
        dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma
        dc_peri = k_peri_in * c_plasma - k_peri_out * c_peri

        a_gut += da_gut * step
        c_plasma += dc_plasma * step
        c_peri += dc_peri * step
        c_plasma = max(0.0, c_plasma)
        c_peri = max(0.0, c_peri)

        t += step

    # Downsample to dt_h grid
    out_times = []
    out_cp = []
    out_cperi = []
    sample_interval = max(1, int(dt_h / dt_int + 0.5))
    for i in range(0, len(times), sample_interval):
        out_times.append(times[i])
        out_cp.append(cp_list[i])
        out_cperi.append(cperi_list[i])
    if out_times[-1] < times[-1]:
        out_times.append(times[-1])
        out_cp.append(cp_list[-1])
        out_cperi.append(cperi_list[-1])

    # Cmax
    cmax_plasma = max(out_cp) if out_cp else 0.0
    cmax_peri = max(out_cperi) if out_cperi else 0.0

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (out_cp[i] + out_cp[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )
    auc_peri = sum(
        0.5 * (out_cperi[i] + out_cperi[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )

    pericardial_to_plasma_ratio = auc_peri / auc_plasma if auc_plasma > 0 else 0.0

    notes_parts = [
        f"k_peri_in={k_peri_in:.4f}/h",
        f"k_peri_out={k_peri_out:.2f}/h",
        f"t_half_peri={t_half_pericardial:.3f}h",
    ]

    return PericardialPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=out_times,
        c_plasma_mg_L=out_cp,
        c_pericardial_mg_L=out_cperi,
        pericardial_volume_mL=pericardial_volume_mL,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_pericardial_mg_L=cmax_peri,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_pericardial_mg_h_per_L=auc_peri,
        pericardial_to_plasma_ratio=pericardial_to_plasma_ratio,
        t_half_pericardial_h=t_half_pericardial,
        notes="; ".join(notes_parts),
    )
