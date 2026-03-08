"""Phase 868 — Interstitial Fluid Drug Distribution model."""

import math
from dataclasses import dataclass, field


@dataclass
class InterstitialFluidPKResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_isf_mg_L: list = field(default_factory=list)
    isf_volume_L: float = 0.0
    kp_isf: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_isf_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_isf_mg_h_per_L: float = 0.0
    isf_to_plasma_ratio: float = 0.0
    t_equilibrium_h: float = 0.0
    capillary_permeability: float = 0.0
    notes: str = ""


def simulate_interstitial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    isf_volume_L: float = 12.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> InterstitialFluidPKResult:
    """Simulate drug distribution to interstitial fluid."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if isf_volume_L <= 0:
        raise ValueError("isf_volume_L must be > 0")

    # Capillary permeability
    P_cap = 1.0 / max(1.0, mw_Da / 500.0) * fu_plasma
    P_cap = max(0.1, min(5.0, P_cap))

    # Rate constants
    k_isf_in = P_cap
    k_isf_out = P_cap * vd_sys_L / isf_volume_L

    # Partition coefficient
    kp_isf = max(0.1, min(2.0, fu_plasma * (1.0 + max(0.0, logP) * 0.1)))

    # Equilibrium time
    t_eq = 3.0 / k_isf_in
    t_equilibrium_h = max(0.1, min(24.0, t_eq))

    ka = 1.2  # /h
    ke = cl_sys_L_per_h / vd_sys_L

    # Adaptive time step
    max_rate = max(ka, ke, k_isf_in, k_isf_out)
    dt_int = min(dt_h, 0.4 / max(1.2, max_rate))

    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_isf = 0.0

    times = [0.0]
    c_plasma_list = [C_plasma]
    c_isf_list = [C_isf]

    for _ in range(n_steps):
        dA_gut = -ka * A_gut
        dC_plasma = ka * A_gut / vd_sys_L - ke * C_plasma
        dC_isf = k_isf_in * C_plasma - k_isf_out * C_isf

        A_gut += dA_gut * dt_int
        C_plasma += dC_plasma * dt_int
        C_isf += dC_isf * dt_int

        A_gut = max(0.0, A_gut)
        C_plasma = max(0.0, C_plasma)
        C_isf = max(0.0, C_isf)

        t = times[-1] + dt_int
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_isf_list.append(C_isf)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_isf = max(c_isf_list)

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_isf = sum(
        0.5 * (c_isf_list[i] + c_isf_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # ISF-to-plasma ratio
    isf_to_plasma_ratio = auc_isf / auc_plasma if auc_plasma > 0 else 0.0

    notes = (
        f"ISF PK for {drug_name}, P_cap={P_cap:.4f}, "
        f"kp_isf={kp_isf:.4f}, t_eq={t_equilibrium_h:.2f} h"
    )

    return InterstitialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_isf_mg_L=c_isf_list,
        isf_volume_L=isf_volume_L,
        kp_isf=kp_isf,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_isf_mg_L=cmax_isf,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_isf_mg_h_per_L=auc_isf,
        isf_to_plasma_ratio=isf_to_plasma_ratio,
        t_equilibrium_h=t_equilibrium_h,
        capillary_permeability=P_cap,
        notes=notes,
    )
