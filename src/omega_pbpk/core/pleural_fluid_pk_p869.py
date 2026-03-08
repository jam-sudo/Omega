"""Phase 869 — Pleural Fluid Drug Distribution model."""

import math
from dataclasses import dataclass, field


@dataclass
class PleuralFluidPKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_pleural_mg_L: list = field(default_factory=list)
    pleural_volume_mL: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_pleural_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_pleural_mg_h_per_L: float = 0.0
    pleural_to_plasma_ratio: float = 0.0
    t_half_pleural_h: float = 0.0
    notes: str = ""


def simulate_pleural_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    pleural_volume_mL: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PleuralFluidPKResult:
    """Simulate drug distribution to pleural fluid."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if pleural_volume_mL <= 0:
        raise ValueError("pleural_volume_mL must be > 0")

    # Pleural membrane permeability
    k_pl_in = 0.3 * fu_plasma / max(1.0, mw_Da / 400.0)
    k_pl_in = max(0.01, min(1.0, k_pl_in))

    # Lymphatic drainage
    k_pl_out = 0.15  # /h

    # Half-life in pleural fluid
    t_half_pleural = math.log(2) / k_pl_out

    # PK parameters
    ka = 1.2  # /h
    ke = cl_sys_L_per_h / vd_sys_L

    # Adaptive time step
    max_rate = max(ka, ke, k_pl_in, k_pl_out)
    dt_int = min(dt_h, 0.4 / max(1.2, max_rate))

    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_pl = 0.0

    times = [0.0]
    c_plasma_list = [C_plasma]
    c_pleural_list = [C_pl]

    for _ in range(n_steps):
        dA_gut = -ka * A_gut
        dC_plasma = ka * A_gut / vd_sys_L - ke * C_plasma
        dC_pl = k_pl_in * C_plasma - k_pl_out * C_pl

        A_gut += dA_gut * dt_int
        C_plasma += dC_plasma * dt_int
        C_pl += dC_pl * dt_int

        A_gut = max(0.0, A_gut)
        C_plasma = max(0.0, C_plasma)
        C_pl = max(0.0, C_pl)

        t = times[-1] + dt_int
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_pleural_list.append(C_pl)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_pleural = max(c_pleural_list)

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_pleural = sum(
        0.5 * (c_pleural_list[i] + c_pleural_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Pleural-to-plasma ratio
    pleural_to_plasma_ratio = auc_pleural / auc_plasma if auc_plasma > 0 else 0.0

    notes = (
        f"Pleural fluid PK for {drug_name}, k_pl_in={k_pl_in:.4f}, "
        f"k_pl_out={k_pl_out:.4f}, t_half_pl={t_half_pleural:.2f} h"
    )

    return PleuralFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_pleural_mg_L=c_pleural_list,
        pleural_volume_mL=pleural_volume_mL,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_pleural_mg_L=cmax_pleural,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_pleural_mg_h_per_L=auc_pleural,
        pleural_to_plasma_ratio=pleural_to_plasma_ratio,
        t_half_pleural_h=t_half_pleural,
        notes=notes,
    )
