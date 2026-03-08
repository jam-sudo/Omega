"""Phase 867 — Cochlear Drug Distribution model."""

from dataclasses import dataclass, field
import math


@dataclass
class CochlearPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_perilymph_mg_L: list = field(default_factory=list)
    c_endolymph_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_perilymph_mg_L: float = 0.0
    cmax_endolymph_mg_L: float = 0.0
    auc_perilymph_mg_h_per_L: float = 0.0
    auc_endolymph_mg_h_per_L: float = 0.0
    blood_labyrinth_barrier_factor: float = 0.0
    t_half_perilymph_h: float = 0.0
    notes: str = ""


def simulate_cochlear_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    route: str = "systemic",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CochlearPKResult:
    """Simulate drug distribution to cochlear fluids."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if route not in ("systemic", "intratympanic"):
        raise ValueError("route must be 'systemic' or 'intratympanic'")

    # Blood-labyrinth barrier factor
    blb_factor = max(0.01, min(0.8, (logP + 1.0) * 0.15 / max(1.0, mw_Da / 300.0)))

    ke = cl_sys_L_per_h / vd_sys_L
    ka = 1.2  # /h

    # Rate constants
    k_peri_out = 0.1  # /h

    if route == "systemic":
        k_peri_in = 0.05 * blb_factor
        k_endo_in = 0.02 * blb_factor
        k_endo_out = 0.05
    else:
        k_endo_in = 0.05
        k_endo_out = 0.05
        k_peri_in = 0.0  # not used for intratympanic

    # Adaptive time step
    max_rate = max(ka, ke, k_peri_in, k_peri_out, k_endo_in, k_endo_out)
    dt_int = min(dt_h, 0.4 / max(1.2, max_rate))

    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    # Initial conditions
    if route == "systemic":
        A_gut = dose_mg * 0.85
        C_plasma = 0.0
        C_peri = 0.0
        C_endo = 0.0
    else:
        # Intratympanic: drug injected directly into perilymph
        A_gut = 0.0
        C_peri = dose_mg / (0.08 / 1000.0)  # dose / perilymph volume in L
        C_plasma = 0.0
        C_endo = 0.0

    times = [0.0]
    c_plasma_list = [C_plasma]
    c_peri_list = [C_peri]
    c_endo_list = [C_endo]

    for _ in range(n_steps):
        if route == "systemic":
            dA_gut = -ka * A_gut
            dC_plasma = ka * A_gut / vd_sys_L - ke * C_plasma
            dC_peri = k_peri_in * C_plasma - k_peri_out * C_peri
            dC_endo = k_endo_in * C_peri - k_endo_out * C_endo
        else:
            dA_gut = 0.0
            dC_peri = -k_peri_out * C_peri - k_endo_in * C_peri
            dC_endo = k_endo_in * C_peri - k_endo_out * C_endo
            dC_plasma = k_peri_out * C_peri * 8e-5 / vd_sys_L - ke * C_plasma

        A_gut += dA_gut * dt_int
        C_plasma += dC_plasma * dt_int
        C_peri += dC_peri * dt_int
        C_endo += dC_endo * dt_int

        A_gut = max(0.0, A_gut)
        C_plasma = max(0.0, C_plasma)
        C_peri = max(0.0, C_peri)
        C_endo = max(0.0, C_endo)

        t = times[-1] + dt_int
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_peri_list.append(C_peri)
        c_endo_list.append(C_endo)

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_peri = max(c_peri_list)
    cmax_endo = max(c_endo_list)

    # AUC (trapezoidal)
    auc_peri = sum(
        0.5 * (c_peri_list[i] + c_peri_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_endo = sum(
        0.5 * (c_endo_list[i] + c_endo_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    t_half_peri = math.log(2) / k_peri_out

    notes = (
        f"Cochlear PK for {drug_name}, route={route}, "
        f"BLB factor={blb_factor:.4f}, t1/2 perilymph={t_half_peri:.2f} h"
    )

    return CochlearPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_perilymph_mg_L=c_peri_list,
        c_endolymph_mg_L=c_endo_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_perilymph_mg_L=cmax_peri,
        cmax_endolymph_mg_L=cmax_endo,
        auc_perilymph_mg_h_per_L=auc_peri,
        auc_endolymph_mg_h_per_L=auc_endo,
        blood_labyrinth_barrier_factor=blb_factor,
        t_half_perilymph_h=t_half_peri,
        notes=notes,
    )
