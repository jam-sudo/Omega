"""Transdermal drug delivery PBPK — Phase 165.

Three-compartment model (patch/SC depot -> dermis -> plasma) for
simulating drug absorption through skin layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TransdermalPKResult:
    """Result of transdermal PK simulation."""

    drug_name: str
    dose_mg: float
    area_cm2: float
    times_h: list[float]
    c_sc_mg_L: list[float]
    c_dermis_mg_L: list[float]
    c_plasma_mg_L: list[float]
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    skin_to_plasma_ratio: float
    transdermal_bioavailability: float
    lag_time_h: float
    notes: list[str]


def simulate_transdermal_pk(
    drug_name: str,
    dose_mg_cm2: float,
    area_cm2: float,
    k_sc_per_h: float = 0.01,
    k_dermis_per_h: float = 0.5,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    v_sc_L: float | None = None,
    v_dermis_L: float | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> TransdermalPKResult:
    """Simulate transdermal drug delivery through skin layers.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg_cm2 : float
        Dose per unit area (mg/cm²). Must be > 0.
    area_cm2 : float
        Patch/application area (cm²). Must be > 0.
    k_sc_per_h : float
        Stratum corneum permeability rate (1/h). Must be > 0.
    k_dermis_per_h : float
        Dermis to systemic transfer rate (1/h).
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Systemic volume of distribution (L).
    v_sc_L : float or None
        Stratum corneum volume (L). Defaults to 0.001 * area_cm2.
    v_dermis_L : float or None
        Dermis volume (L). Defaults to 0.005 * area_cm2.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    TransdermalPKResult
    """
    if dose_mg_cm2 <= 0:
        raise ValueError("dose_mg_cm2 must be > 0")
    if area_cm2 <= 0:
        raise ValueError("area_cm2 must be > 0")
    if k_sc_per_h <= 0:
        raise ValueError("k_sc_per_h must be > 0")

    if v_sc_L is None:
        v_sc_L = 0.001 * area_cm2
    if v_dermis_L is None:
        v_dermis_L = 0.005 * area_cm2

    dose_mg = dose_mg_cm2 * area_cm2
    notes: list[str] = []

    n_steps = int(math.ceil(t_end_h / dt_h))
    times = [i * dt_h for i in range(n_steps + 1)]

    # State: amounts (mg) in each compartment
    a_sc = dose_mg  # SC depot starts with full dose
    a_dermis = 0.0
    a_plasma = 0.0
    a_eliminated = 0.0

    ke = cl_sys_L_per_h / vd_sys_L  # elimination rate

    c_sc_list: list[float] = []
    c_dermis_list: list[float] = []
    c_plasma_list: list[float] = []

    for _ in times:
        c_sc_list.append(a_sc / v_sc_L if v_sc_L > 0 else 0.0)
        c_dermis_list.append(a_dermis / v_dermis_L if v_dermis_L > 0 else 0.0)
        c_plasma_list.append(a_plasma / vd_sys_L if vd_sys_L > 0 else 0.0)

        # Forward Euler
        d_sc_to_dermis = k_sc_per_h * a_sc * dt_h
        d_dermis_to_plasma = k_dermis_per_h * a_dermis * dt_h
        d_elimination = ke * a_plasma * dt_h

        a_sc = a_sc - d_sc_to_dermis
        a_dermis = a_dermis + d_sc_to_dermis - d_dermis_to_plasma
        a_plasma = a_plasma + d_dermis_to_plasma - d_elimination
        a_eliminated = a_eliminated + d_elimination

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma_h = times[tmax_idx]

    cmax_dermis = max(c_dermis_list)
    skin_to_plasma_ratio = cmax_dermis / cmax_plasma if cmax_plasma > 0 else 0.0

    auc_plasma = float(np_trapz(c_plasma_list, times))

    # Bioavailability: fraction of dose that was absorbed systemically
    total_absorbed = a_eliminated + a_plasma
    transdermal_bioavailability = total_absorbed / dose_mg if dose_mg > 0 else 0.0

    # Lag time: time to reach 5% of Cmax
    threshold = 0.05 * cmax_plasma
    lag_time_h = 0.0
    for i, c in enumerate(c_plasma_list):
        if c >= threshold:
            lag_time_h = times[i]
            break

    if transdermal_bioavailability < 0.3:
        notes.append("Low transdermal bioavailability (<30%)")
    if lag_time_h > 4.0:
        notes.append("Extended lag time (>4 h)")

    return TransdermalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        area_cm2=area_cm2,
        times_h=times,
        c_sc_mg_L=c_sc_list,
        c_dermis_mg_L=c_dermis_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        skin_to_plasma_ratio=skin_to_plasma_ratio,
        transdermal_bioavailability=transdermal_bioavailability,
        lag_time_h=lag_time_h,
        notes=notes,
    )


def compare_formulations_transdermal(
    drug_name: str,
    dose_mg_cm2: float,
    area_cm2: float,
    formulations: list[dict],
    **kwargs,
) -> list[TransdermalPKResult]:
    """Compare multiple transdermal formulations.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg_cm2 : float
        Dose per unit area (mg/cm²).
    area_cm2 : float
        Patch area (cm²).
    formulations : list[dict]
        Each dict contains formulation-specific overrides
        (e.g. k_sc_per_h, k_dermis_per_h).

    Returns
    -------
    list[TransdermalPKResult]
    """
    if not formulations:
        raise ValueError("formulations list must not be empty")

    results: list[TransdermalPKResult] = []
    for form in formulations:
        merged = {**kwargs, **form}
        result = simulate_transdermal_pk(
            drug_name=drug_name,
            dose_mg_cm2=dose_mg_cm2,
            area_cm2=area_cm2,
            **merged,
        )
        results.append(result)
    return results
