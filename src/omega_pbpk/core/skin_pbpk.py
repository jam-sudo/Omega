"""Multi-layer skin PBPK model for topically applied drugs.

Four-compartment forward-Euler model:
  Stratum Corneum → Viable Epidermis → Dermis → Systemic
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass
class SkinPBPKResult:
    """Result of a multi-layer skin PBPK simulation."""

    drug_name: str
    dose_mg_cm2: float
    application_area_cm2: float
    times_h: list[float] = field(default_factory=list)
    a_sc_mg: list[float] = field(default_factory=list)
    a_vep_mg: list[float] = field(default_factory=list)
    a_derm_mg: list[float] = field(default_factory=list)
    c_sys_mg_L: list[float] = field(default_factory=list)
    cmax_sys: float = 0.0
    auc_sys: float = 0.0
    f_systemic: float = 0.0
    notes: str = ""


def simulate_skin_pbpk(
    drug_name: str,
    dose_mg_cm2: float,
    application_area_cm2: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    k_sc_per_h: float = 0.1,
    k_vep_per_h: float = 0.5,
    k_dermis_per_h: float = 2.0,
    k_sys_per_h: float = 3.0,
) -> SkinPBPKResult:
    """Simulate topical drug absorption through a 4-layer skin model.

    Layers (forward Euler ODEs):
      SC  : dA_sc/dt  = -k_sc * A_sc
      VEP : dA_vep/dt = k_sc * A_sc  - k_vep * A_vep
      Derm: dA_derm/dt= k_vep * A_vep - k_derm * A_derm
      Sys : dC_sys/dt = k_derm * A_derm / Vd - ke * C_sys

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    dose_mg_cm2 : float
        Applied dose per unit area (mg/cm²), must be > 0.
    application_area_cm2 : float
        Application area (cm²), must be > 0.
    cl_sys_L_per_h : float
        Systemic clearance (L/h), must be >= 0.
    vd_sys_L : float
        Systemic volume of distribution (L), must be > 0.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step size (h).
    k_sc_per_h : float
        First-order transfer rate from stratum corneum (1/h).
    k_vep_per_h : float
        First-order transfer rate from viable epidermis (1/h).
    k_dermis_per_h : float
        First-order transfer rate from dermis to systemic (1/h).
    k_sys_per_h : float
        Systemic elimination rate constant (1/h); used as ke when cl/vd not provided.
        Note: ke is computed as cl_sys / vd_sys; k_sys_per_h is kept for backward compat.

    Returns
    -------
    SkinPBPKResult
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg_cm2 <= 0:
        raise ValueError("dose_mg_cm2 must be > 0")
    if application_area_cm2 <= 0:
        raise ValueError("application_area_cm2 must be > 0")
    if cl_sys_L_per_h < 0:
        raise ValueError("cl_sys_L_per_h must be >= 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if k_sc_per_h < 0:
        raise ValueError("k_sc_per_h must be >= 0")
    if k_vep_per_h < 0:
        raise ValueError("k_vep_per_h must be >= 0")
    if k_dermis_per_h < 0:
        raise ValueError("k_dermis_per_h must be >= 0")
    if k_sys_per_h < 0:
        raise ValueError("k_sys_per_h must be >= 0")

    total_dose_mg = dose_mg_cm2 * application_area_cm2

    # Eliminate using actual CL/Vd; k_sys_per_h is a fallback for ke
    ke = cl_sys_L_per_h / vd_sys_L if cl_sys_L_per_h > 0 else k_sys_per_h

    n_steps = max(1, int(math.ceil(t_end_h / dt_h)))
    times = [0.0] * (n_steps + 1)
    a_sc = [0.0] * (n_steps + 1)
    a_vep = [0.0] * (n_steps + 1)
    a_derm = [0.0] * (n_steps + 1)
    c_sys = [0.0] * (n_steps + 1)

    # Initial conditions
    a_sc[0] = total_dose_mg

    for i in range(n_steps):
        t_now = i * dt_h
        times[i] = t_now

        sc_i = a_sc[i]
        vep_i = a_vep[i]
        derm_i = a_derm[i]
        csys_i = c_sys[i]

        da_sc = -k_sc_per_h * sc_i
        da_vep = k_sc_per_h * sc_i - k_vep_per_h * vep_i
        da_derm = k_vep_per_h * vep_i - k_dermis_per_h * derm_i
        dc_sys = k_dermis_per_h * derm_i / vd_sys_L - ke * csys_i

        a_sc[i + 1] = max(0.0, sc_i + dt_h * da_sc)
        a_vep[i + 1] = max(0.0, vep_i + dt_h * da_vep)
        a_derm[i + 1] = max(0.0, derm_i + dt_h * da_derm)
        c_sys[i + 1] = max(0.0, csys_i + dt_h * dc_sys)

    times[n_steps] = n_steps * dt_h

    # AUC via trapezoidal rule
    t_arr = np.array(times)
    c_arr = np.array(c_sys)
    auc = float(np_trapz(c_arr, t_arr))
    cmax = float(max(c_sys))

    # Fraction reaching systemic (based on cumulative absorbed amount reaching systemic)
    # Approximated as AUC * CL / total_dose
    amount_sys = auc * cl_sys_L_per_h if cl_sys_L_per_h > 0 else cmax * vd_sys_L
    f_systemic = float(np.clip(amount_sys / total_dose_mg, 0.0, 1.0)) if total_dose_mg > 0 else 0.0

    return SkinPBPKResult(
        drug_name=drug_name,
        dose_mg_cm2=dose_mg_cm2,
        application_area_cm2=application_area_cm2,
        times_h=times,
        a_sc_mg=a_sc,
        a_vep_mg=a_vep,
        a_derm_mg=a_derm,
        c_sys_mg_L=c_sys,
        cmax_sys=cmax,
        auc_sys=auc,
        f_systemic=f_systemic,
        notes="Forward-Euler 4-layer skin PBPK: SC→VEP→Dermis→Systemic",
    )


def compare_penetration_enhancers(
    drug_name: str,
    dose_mg_cm2: float,
    area_cm2: float,
    cl_sys: float,
    vd_sys: float,
    enhancer_factors: list[dict],
) -> list[dict]:
    """Compare penetration enhancers by systemic AUC.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg_cm2 : float
        Applied dose per unit area (mg/cm²).
    area_cm2 : float
        Application area (cm²).
    cl_sys : float
        Systemic clearance (L/h).
    vd_sys : float
        Systemic volume of distribution (L).
    enhancer_factors : list[dict]
        Each entry must have ``name`` (str) and ``k_sc_mult`` (float >= 0).
        Example: [{"name": "baseline", "k_sc_mult": 1.0},
                  {"name": "DMSO", "k_sc_mult": 3.0}]

    Returns
    -------
    list[dict]
        One dict per enhancer with keys: name, k_sc_mult, auc_sys, cmax_sys, f_systemic.
        Sorted by auc_sys descending.
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg_cm2 <= 0:
        raise ValueError("dose_mg_cm2 must be > 0")
    if area_cm2 <= 0:
        raise ValueError("area_cm2 must be > 0")
    if cl_sys < 0:
        raise ValueError("cl_sys must be >= 0")
    if vd_sys <= 0:
        raise ValueError("vd_sys must be > 0")
    if not enhancer_factors:
        raise ValueError("enhancer_factors must be a non-empty list")

    results: list[dict] = []
    for ef in enhancer_factors:
        name = ef.get("name", "unknown")
        k_sc_mult = float(ef.get("k_sc_mult", 1.0))
        if k_sc_mult < 0:
            raise ValueError(f"k_sc_mult must be >= 0 (got {k_sc_mult} for '{name}')")

        sim = simulate_skin_pbpk(
            drug_name=drug_name,
            dose_mg_cm2=dose_mg_cm2,
            application_area_cm2=area_cm2,
            cl_sys_L_per_h=cl_sys,
            vd_sys_L=vd_sys,
            k_sc_per_h=0.1 * k_sc_mult,
        )
        results.append(
            {
                "name": name,
                "k_sc_mult": k_sc_mult,
                "auc_sys": sim.auc_sys,
                "cmax_sys": sim.cmax_sys,
                "f_systemic": sim.f_systemic,
            }
        )

    results.sort(key=lambda r: r["auc_sys"], reverse=True)
    return results


__all__ = ["SkinPBPKResult", "simulate_skin_pbpk", "compare_penetration_enhancers"]
