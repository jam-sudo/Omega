"""Phase 870 — Ascitic Fluid Drug Distribution model."""

import math
from dataclasses import dataclass, field


@dataclass
class AsciticFluidPKResult:
    drug_name: str
    dose_mg: float
    ascites_volume_L: float = 0.0
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_ascites_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_ascites_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_ascites_mg_h_per_L: float = 0.0
    ascites_to_plasma_ratio: float = 0.0
    t_half_ascites_h: float = 0.0
    dose_adjustment_factor: float = 0.0
    notes: str = ""


def simulate_ascitic_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ascites_volume_L: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> AsciticFluidPKResult:
    """Simulate drug distribution to ascitic fluid."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if ascites_volume_L <= 0:
        raise ValueError("ascites_volume_L must be > 0")

    # Peritoneal membrane permeability
    k_asc_in = 0.2 * fu_plasma / max(1.0, mw_Da / 400.0)
    k_asc_in = max(0.005, min(0.5, k_asc_in))

    # Slow drainage in ascites
    k_asc_out = 0.02  # /h

    # Effective Vd (ascites acts as drug reservoir)
    vd_effective = vd_sys_L + ascites_volume_L

    # Dose adjustment factor
    dose_adjustment_factor = vd_effective / vd_sys_L

    # Half-life in ascitic fluid
    t_half_ascites = math.log(2) / k_asc_out

    # PK parameters
    ka = 1.2  # /h
    ke = cl_sys_L_per_h / vd_effective

    # Adaptive time step
    max_rate = max(ka, ke, k_asc_in, k_asc_out)
    dt_int = min(dt_h, 0.4 / max(1.2, max_rate))

    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_asc = 0.0

    times = [0.0]
    c_plasma_list = [C_plasma]
    c_ascites_list = [C_asc]

    for _ in range(n_steps):
        dA_gut = -ka * A_gut
        dC_plasma = ka * A_gut / vd_effective - ke * C_plasma
        dC_asc = k_asc_in * C_plasma - k_asc_out * C_asc

        A_gut += dA_gut * dt_int
        C_plasma += dC_plasma * dt_int
        C_asc += dC_asc * dt_int

        A_gut = max(0.0, A_gut)
        C_plasma = max(0.0, C_plasma)
        C_asc = max(0.0, C_asc)

        t = times[-1] + dt_int
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_ascites_list.append(C_asc)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_ascites = max(c_ascites_list)

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_ascites = sum(
        0.5 * (c_ascites_list[i] + c_ascites_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Ascites-to-plasma ratio
    ascites_to_plasma_ratio = auc_ascites / auc_plasma if auc_plasma > 0 else 0.0

    notes = (
        f"Ascitic fluid PK for {drug_name}, k_asc_in={k_asc_in:.4f}, "
        f"k_asc_out={k_asc_out:.4f}, dose_adj={dose_adjustment_factor:.2f}"
    )

    return AsciticFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ascites_volume_L=ascites_volume_L,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_ascites_mg_L=c_ascites_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_ascites_mg_L=cmax_ascites,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_ascites_mg_h_per_L=auc_ascites,
        ascites_to_plasma_ratio=ascites_to_plasma_ratio,
        t_half_ascites_h=t_half_ascites,
        dose_adjustment_factor=dose_adjustment_factor,
        notes=notes,
    )
