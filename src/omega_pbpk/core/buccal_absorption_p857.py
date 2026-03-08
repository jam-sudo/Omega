"""Phase 857 — Buccal Mucosa Drug Absorption."""

import math
from dataclasses import dataclass, field


@dataclass
class BuccalAbsorptionResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_buccal_depot_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    kp_buccal: float = 0.0
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    bioavailability_pct: float = 0.0
    t_half_buccal_h: float = 0.0
    permeability_class: str = ""
    notes: str = ""


def simulate_buccal_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    pka: float,
    is_acid: bool,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    saliva_ph: float = 6.8,
    t_end_h: float = 4.0,
    dt_h: float = 0.02,
) -> BuccalAbsorptionResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    k_wash = 0.2  # /h

    # pH-dependent unionized fraction
    if is_acid:
        f_unionized_buccal = 1.0 / (1.0 + 10.0 ** (saliva_ph - pka))
    else:
        f_unionized_buccal = 1.0 / (1.0 + 10.0 ** (pka - saliva_ph))
    f_unionized_buccal = max(0.001, min(1.0, f_unionized_buccal))

    # Permeability
    P_buccal = 0.002 * f_unionized_buccal * max(0.1, logP + 1.0) / max(1.0, mw_Da / 300.0)
    P_buccal = max(0.001, min(0.5, P_buccal))

    kp_buccal = P_buccal / (P_buccal + k_wash)
    ke = cl_sys_L_per_h / vd_sys_L
    t_half_buccal = math.log(2) / (P_buccal + k_wash)

    if P_buccal > 0.05:
        permeability_class = "high"
    elif P_buccal > 0.01:
        permeability_class = "moderate"
    else:
        permeability_class = "low"

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(P_buccal + k_wash, ke))

    # Forward Euler
    A_buccal = dose_mg
    C_plasma = 0.0
    t = 0.0

    times_h = [t]
    c_buccal_depot_mg = [A_buccal]
    c_plasma_mg_L = [C_plasma]

    n_steps = int(math.ceil(t_end_h / dt_int))
    for _ in range(n_steps):
        dA = -(P_buccal + k_wash) * A_buccal
        dC = P_buccal * A_buccal / vd_sys_L - ke * C_plasma

        A_buccal += dA * dt_int
        C_plasma += dC * dt_int

        A_buccal = max(0.0, A_buccal)
        C_plasma = max(0.0, C_plasma)

        t += dt_int
        times_h.append(t)
        c_buccal_depot_mg.append(A_buccal)
        c_plasma_mg_L.append(C_plasma)

    # Cmax, Tmax
    cmax_mg_L = 0.0
    tmax_h = 0.0
    for i, c in enumerate(c_plasma_mg_L):
        if c > cmax_mg_L:
            cmax_mg_L = c
            tmax_h = times_h[i]

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    bioavailability_pct = kp_buccal * 100.0

    notes = (
        f"Buccal absorption: P_buccal={P_buccal:.4f}/h, "
        f"f_unionized={f_unionized_buccal:.4f}, "
        f"permeability={permeability_class}"
    )

    return BuccalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_buccal_depot_mg=c_buccal_depot_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        kp_buccal=kp_buccal,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        t_half_buccal_h=t_half_buccal,
        permeability_class=permeability_class,
        notes=notes,
    )
