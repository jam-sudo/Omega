"""Phase 681 — Nasal Drug Absorption model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class NasalAbsorptionResult:
    """Result of nasal drug absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_nasal_depot_mg: list = field(default_factory=list)
    mucociliary_cleared_mg: list = field(default_factory=list)
    bioavailability_nasal: float = 0.0
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_plasma_h: float = 0.0
    notes: str = ""


_FORMULATION_PARAMS = {
    "solution": {"ka": 1.5, "k_mc": 0.3},
    "powder": {"ka": 0.8, "k_mc": 0.15},
    "gel": {"ka": 0.4, "k_mc": 0.05},
}

_VALID_FORMULATIONS = set(_FORMULATION_PARAMS.keys())


def simulate_nasal_absorption(
    drug_name: str,
    dose_mg: float,
    formulation: str,
    mw_Da: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> NasalAbsorptionResult:
    """Simulate nasal drug absorption with mucociliary clearance."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {sorted(_VALID_FORMULATIONS)}")

    params = _FORMULATION_PARAMS[formulation]
    ka_nasal = params["ka"]
    k_mc = params["k_mc"]

    # Systemic bioavailability factor
    f_sys = min(1.0, max(0.3, 1.0 - mw_Da / 2000.0 + logP * 0.05))

    # Elimination rate
    ke = cl_sys_L_per_h / vd_sys_L

    # Initial conditions
    A_nasal = dose_mg
    C_plasma = 0.0
    total_mc_cleared = 0.0

    times_h = []
    c_plasma_mg_L = []
    c_nasal_depot_mg = []
    mucociliary_cleared_mg = []

    cmax = 0.0
    tmax = 0.0

    n_steps = int(t_end_h / dt_h) + 1

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_plasma_mg_L.append(C_plasma)
        c_nasal_depot_mg.append(A_nasal)
        mucociliary_cleared_mg.append(total_mc_cleared)

        if C_plasma > cmax:
            cmax = C_plasma
            tmax = t

        # Forward Euler
        dA_nasal = -(ka_nasal + k_mc) * A_nasal
        dC_plasma = ka_nasal * A_nasal * f_sys / vd_sys_L - ke * C_plasma
        d_mc = k_mc * A_nasal

        A_nasal = max(0.0, A_nasal + dA_nasal * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        total_mc_cleared = total_mc_cleared + d_mc * dt_h

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_plasma_mg_L[j] + c_plasma_mg_L[j - 1]) * (times_h[j] - times_h[j - 1])
        for j in range(1, len(times_h))
    )

    # Absorption fraction and bioavailability
    absorption_fraction = ka_nasal / (ka_nasal + k_mc)
    bioavailability_nasal = absorption_fraction * f_sys

    # Half-lives
    t_half_plasma = math.log(2) / ke

    return NasalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_nasal_depot_mg=c_nasal_depot_mg,
        mucociliary_cleared_mg=mucociliary_cleared_mg,
        bioavailability_nasal=bioavailability_nasal,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_plasma_h=t_half_plasma,
        notes=f"Nasal {formulation} absorption; MW={mw_Da} Da, logP={logP}",
    )
