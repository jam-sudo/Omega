"""Phase 682 — Intramuscular Depot PK model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IMDepotPKResult:
    """Result of intramuscular depot PK simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list = field(default_factory=list)
    c_depot_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_absorption_h: float = 0.0
    t_half_plasma_h: float = 0.0
    flip_flop: bool = False
    bioavailability_pct: float = 0.0
    notes: str = ""


_FORMULATION_PARAMS = {
    "aqueous": {"ka": 1.5},
    "oil": {"ka": 0.05},
    "microsphere": {"ka": 0.02},
    "suspension": {"ka": 0.3},
}

_VALID_FORMULATIONS = set(_FORMULATION_PARAMS.keys())


def simulate_im_depot_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    logP: float = 2.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> IMDepotPKResult:
    """Simulate intramuscular depot injection PK."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}")

    ka = _FORMULATION_PARAMS[formulation_type]["ka"]
    ke = cl_sys_L_per_h / vd_sys_L

    # Initial conditions
    A_depot = dose_mg
    C_plasma = 0.0

    times_h = []
    c_depot_mg = []
    c_plasma_mg_L = []

    cmax = 0.0
    tmax = 0.0

    n_steps = int(t_end_h / dt_h) + 1

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_depot_mg.append(A_depot)
        c_plasma_mg_L.append(C_plasma)

        if C_plasma > cmax:
            cmax = C_plasma
            tmax = t

        # Forward Euler
        dA_depot = -ka * A_depot
        dC_plasma = ka * A_depot / vd_sys_L - ke * C_plasma

        A_depot = max(0.0, A_depot + dA_depot * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_plasma_mg_L[j] + c_plasma_mg_L[j - 1]) * (times_h[j] - times_h[j - 1])
        for j in range(1, len(times_h))
    )

    t_half_absorption = math.log(2) / ka
    t_half_plasma = math.log(2) / ke
    flip_flop = t_half_absorption > t_half_plasma

    return IMDepotPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        c_depot_mg=c_depot_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_absorption_h=t_half_absorption,
        t_half_plasma_h=t_half_plasma,
        flip_flop=flip_flop,
        bioavailability_pct=85.0,
        notes=f"IM {formulation_type} depot; logP={logP}",
    )
