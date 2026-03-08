"""Phase 858 — Vaginal Drug Absorption."""

import math
from dataclasses import dataclass, field

VALID_FORMULATIONS = {"gel", "tablet", "ring", "cream"}

KA_BY_FORMULATION = {
    "gel": 0.3,
    "tablet": 0.15,
    "ring": 0.02,
    "cream": 0.25,
}


@dataclass
class VaginalAbsorptionResult:
    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list = field(default_factory=list)
    c_vaginal_depot_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    bioavailability_pct: float = 0.0
    local_conc_mg_mL: float = 0.0
    t_half_absorption_h: float = 0.0
    notes: str = ""


def simulate_vaginal_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    formulation: str = "gel",
    vaginal_ph: float = 4.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> VaginalAbsorptionResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if formulation not in VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {sorted(VALID_FORMULATIONS)}, got '{formulation}'"
        )

    k_drain = 0.1  # /h
    ka_vag = KA_BY_FORMULATION[formulation]

    F_sys = min(0.8, max(0.1, 0.3 + logP * 0.05 - mw_Da / 3000.0))
    ke = cl_sys_L_per_h / vd_sys_L

    local_conc_mg_mL = dose_mg / 2.0
    bioavailability_pct = F_sys * ka_vag / (ka_vag + k_drain) * 100.0
    t_half_absorption = math.log(2) / (ka_vag + k_drain)

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(ka_vag + k_drain, ke))

    # Forward Euler
    A_depot = dose_mg
    C_plasma = 0.0
    t = 0.0

    times_h = [t]
    c_vaginal_depot_mg = [A_depot]
    c_plasma_mg_L = [C_plasma]

    n_steps = int(math.ceil(t_end_h / dt_int))
    for _ in range(n_steps):
        dA = -(ka_vag + k_drain) * A_depot
        dC = ka_vag * F_sys * A_depot / vd_sys_L - ke * C_plasma

        A_depot += dA * dt_int
        C_plasma += dC * dt_int

        A_depot = max(0.0, A_depot)
        C_plasma = max(0.0, C_plasma)

        t += dt_int
        times_h.append(t)
        c_vaginal_depot_mg.append(A_depot)
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

    notes = (
        f"Vaginal absorption: formulation={formulation}, "
        f"ka_vag={ka_vag}/h, F_sys={F_sys:.3f}, "
        f"bioavailability={bioavailability_pct:.1f}%"
    )

    return VaginalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times_h,
        c_vaginal_depot_mg=c_vaginal_depot_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        local_conc_mg_mL=local_conc_mg_mL,
        t_half_absorption_h=t_half_absorption,
        notes=notes,
    )
