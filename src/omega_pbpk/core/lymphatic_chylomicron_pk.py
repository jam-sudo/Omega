"""Phase 356 — Drug Absorption via Lymphatics (Chylomicron Route).

Models lymphatic drug absorption via chylomicron association for lipophilic
drugs using a 3-compartment Forward Euler approach:
gut → (portal_vein [1-f_chylo] + lymphatics [f_chylo]) → systemic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Lymphatic first-order release rate constant (1/h)
_K_LYMPH_RELEASE_PER_H = 0.5


def _logistic(x: float) -> float:
    """Compute logistic (sigmoid) function: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Numerically stable for negative x
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_f_chylomicron(logP: float) -> float:
    """Compute chylomicron association fraction from logP.

    f_chylo = logistic(logP - 3.0)
    High logP → more chylomicron associated → lymphatic route.
    """
    return _logistic(logP - 3.0)


def _food_effect_class(f_chylo: float) -> str:
    """Classify food effect based on chylomicron association fraction."""
    if f_chylo > 0.3:
        return "significant"
    if f_chylo >= 0.1:
        return "minor"
    return "negligible"


@dataclass
class LymphaticChylomicronResult:
    """Result of lymphatic chylomicron absorption simulation."""

    times_h: list
    c_gut_mg_mL: list
    c_lymph_depot_mg: list
    c_systemic_mg_L: list
    f_chylomicron: float
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    lymphatic_contribution_pct: float
    food_effect_class: str
    notes: str


def simulate_lymphatic_chylomicron(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ka_per_h: float,
    f_oral: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> LymphaticChylomicronResult:
    """Simulate lymphatic chylomicron-mediated drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose (mg).
    logP:
        Octanol-water partition coefficient (log scale).
    ka_per_h:
        First-order gut absorption rate constant (1/h).
    f_oral:
        Fraction of dose available for absorption (0 < f_oral <= 1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for Forward Euler (h).

    Returns
    -------
    LymphaticChylomicronResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be positive, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be positive, got {vd_sys_L}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be positive, got {ka_per_h}")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")

    f_chylo = compute_f_chylomicron(logP)
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Volumes for concentration tracking
    v_gut_mL = 1000.0  # representative gut volume (mL)

    # State variables
    a_gut = f_oral * dose_mg  # amount in gut (mg)
    a_lymph = 0.0  # amount in lymph depot (mg)
    c_sys = 0.0  # systemic concentration (mg/L)

    times_h: list = []
    c_gut_list: list = []
    c_lymph_list: list = []
    c_sys_list: list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        # Record state
        times_h.append(t)
        c_gut_list.append(a_gut / v_gut_mL)  # mg/mL
        c_lymph_list.append(a_lymph)  # mg
        c_sys_list.append(c_sys)

        if t >= t_end_h:
            break

        # Fluxes
        gut_absorption_flux = ka_per_h * a_gut  # mg/h
        portal_flux = gut_absorption_flux * (1.0 - f_chylo)
        lymph_input_flux = gut_absorption_flux * f_chylo
        lymph_release_flux = _K_LYMPH_RELEASE_PER_H * a_lymph

        # ODEs
        da_gut_dt = -gut_absorption_flux
        da_lymph_dt = lymph_input_flux - lymph_release_flux
        dc_sys_dt = (portal_flux + lymph_release_flux) / vd_sys_L - ke_sys * c_sys

        # Forward Euler update
        a_gut += da_gut_dt * dt_h
        a_lymph += da_lymph_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Clamp to non-negative
        if a_gut < 0.0:
            a_gut = 0.0
        if a_lymph < 0.0:
            a_lymph = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t = round(t + dt_h, 10)

    # AUC via trapezoidal rule
    auc_systemic = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    cmax_systemic = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax_systemic)
    tmax_systemic = times_h[tmax_idx]

    lymphatic_contribution_pct = f_chylo * 100.0
    food_effect = _food_effect_class(f_chylo)

    notes = (
        f"Lymphatic chylomicron absorption model for {drug_name}. "
        f"logP={logP:.2f}, f_chylomicron={f_chylo:.3f}, "
        f"lymphatic_contribution={lymphatic_contribution_pct:.1f}%, "
        f"food_effect={food_effect}."
    )

    return LymphaticChylomicronResult(
        times_h=times_h,
        c_gut_mg_mL=c_gut_list,
        c_lymph_depot_mg=c_lymph_list,
        c_systemic_mg_L=c_sys_list,
        f_chylomicron=f_chylo,
        auc_systemic_mg_h_per_L=auc_systemic,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        lymphatic_contribution_pct=lymphatic_contribution_pct,
        food_effect_class=food_effect,
        notes=notes,
    )
