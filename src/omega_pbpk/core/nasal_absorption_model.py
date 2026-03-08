"""Nasal drug absorption model with anterior/posterior compartments and mucociliary clearance."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Formulation lookup: (anterior_fraction, posterior_fraction, k_mcc_per_h)
_FORMULATIONS: dict[str, tuple[float, float, float]] = {
    "solution": (0.3, 0.7, 12.0),
    "spray": (0.5, 0.5, 6.0),
    "powder": (0.2, 0.8, 3.0),
}

_NASAL_SURFACE_AREA_CM2 = 150.0
_NASAL_VOLUME_ML = 15.0


def _calc_k_abs(mw_da: float, logp: float) -> float:
    """Effective absorption rate constant from nasal cavity (1/h)."""
    kp_nasal = 10.0 ** (-1.5 + 0.5 * logp - 0.003 * mw_da)
    k_abs = kp_nasal * _NASAL_SURFACE_AREA_CM2 / _NASAL_VOLUME_ML
    return k_abs


@dataclass
class NasalAbsorptionResult:
    """Result of a nasal drug absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list[float]
    c_systemic_mg_L: list[float]
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    nasal_bioavailability_pct: float
    k_abs_per_h: float
    notes: str


def simulate_nasal_absorption(
    drug_name: str,
    dose_mg: float,
    formulation: str,
    mw_da: float,
    logp: float,
    cl_sys_l_per_h: float = 5.0,
    vd_sys_l: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> NasalAbsorptionResult:
    """Simulate nasal drug absorption using a 3-compartment ODE model.

    Compartments:
        A_ant  - drug mass in anterior nasal cavity (mg)
        A_post - drug mass in posterior nasal cavity (mg)
        C_sys  - drug concentration in systemic circulation (mg/L)
    """
    if formulation not in _FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {set(_FORMULATIONS.keys())}, got '{formulation}'"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if cl_sys_l_per_h <= 0:
        raise ValueError(f"cl_sys_l_per_h must be > 0, got {cl_sys_l_per_h}")
    if vd_sys_l <= 0:
        raise ValueError(f"vd_sys_l must be > 0, got {vd_sys_l}")

    ant_frac, post_frac, k_mcc = _FORMULATIONS[formulation]
    k_abs = _calc_k_abs(mw_da, logp)
    k_abs_ant = k_abs * 0.1
    k_abs_post = k_abs
    k_mcc_post = k_mcc * 0.3

    # Initial conditions
    a_ant = dose_mg * ant_frac
    a_post = dose_mg * post_frac
    c_sys = 0.0

    # Elimination rate from systemic compartment
    k_el = cl_sys_l_per_h / vd_sys_l

    times: list[float] = []
    c_sys_traj: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        c_sys_traj.append(c_sys)

        # Forward Euler derivatives
        d_ant = -(k_mcc + k_abs_ant) * a_ant
        d_post = -(k_mcc_post + k_abs_post) * a_post
        # Absorption into systemic (mg/h) / Vd -> concentration rate (mg/L/h)
        absorption_rate_sys = (k_abs_ant * a_ant + k_abs_post * a_post) / vd_sys_l
        d_csys = absorption_rate_sys - k_el * c_sys

        a_ant = max(0.0, a_ant + d_ant * dt_h)
        a_post = max(0.0, a_post + d_post * dt_h)
        c_sys = max(0.0, c_sys + d_csys * dt_h)
        t = round(t + dt_h, 10)

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_sys_traj[i] + c_sys_traj[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    cmax = max(c_sys_traj)
    tmax = times[c_sys_traj.index(cmax)]

    # Nasal bioavailability: total absorbed / dose
    # Absorbed = dose - remaining in nasal at end (approx via AUC * CL)
    total_absorbed_mg = auc * cl_sys_l_per_h
    nasal_ba_pct = min(100.0, (total_absorbed_mg / dose_mg) * 100.0)

    notes = (
        f"k_mcc={k_mcc:.2f}/h, k_abs_ant={k_abs_ant:.4f}/h, "
        f"k_abs_post={k_abs_post:.4f}/h, k_mcc_post={k_mcc_post:.3f}/h"
    )

    return NasalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times,
        c_systemic_mg_L=c_sys_traj,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        auc_systemic_mg_h_per_L=auc,
        nasal_bioavailability_pct=nasal_ba_pct,
        k_abs_per_h=k_abs,
        notes=notes,
    )


def compare_nasal_formulations(
    drug_name: str,
    dose_mg: float,
    mw_da: float,
    logp: float,
    cl_sys_l_per_h: float = 5.0,
    vd_sys_l: float = 50.0,
) -> list[NasalAbsorptionResult]:
    """Compare all 3 nasal formulations, sorted by AUC descending."""
    results = [
        simulate_nasal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=f,
            mw_da=mw_da,
            logp=logp,
            cl_sys_l_per_h=cl_sys_l_per_h,
            vd_sys_l=vd_sys_l,
        )
        for f in _FORMULATIONS
    ]
    return sorted(results, key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
