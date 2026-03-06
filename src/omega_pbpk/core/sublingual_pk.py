"""Sublingual/Buccal absorption model.

Models drug absorption through oral mucosal membranes:
  - Sublingual: under the tongue (faster)
  - Buccal: cheek (slightly slower)

Key feature: bypasses hepatic first-pass metabolism — higher bioavailability
for high-extraction drugs compared to swallowed oral dosing.

Two-compartment Forward Euler ODE:
  dM_mucosal/dt = -(ka_mucosal + k_swallow) * M_mucosal
  dC_plasma/dt  = (ka_mucosal * M_mucosal) / Vd - (CL/Vd) * C_plasma
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SublingualPKResult:
    """Result of sublingual/buccal PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    m_mucosal_mg: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed_mucosal: float
    onset_time_h: float
    notes: list[str]


def simulate_sublingual_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_mucosal_per_h: float = 2.0,
    k_swallow_per_h: float = 0.5,
    route: str = "sublingual",
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> SublingualPKResult:
    """Simulate sublingual or buccal drug absorption.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    ka_mucosal_per_h:
        Mucosal absorption rate constant (1/h). Default 2.0.
    k_swallow_per_h:
        Swallowing/saliva washout rate constant (1/h). Default 0.5.
    route:
        "sublingual" or "buccal". Buccal uses 1.5x slower ka.
    t_end_h:
        Simulation duration in hours.
    dt_h:
        ODE integration step size in hours.

    Returns
    -------
    SublingualPKResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ka_mucosal_per_h <= 0:
        raise ValueError(f"ka_mucosal_per_h must be > 0, got {ka_mucosal_per_h}")
    if k_swallow_per_h <= 0:
        raise ValueError(f"k_swallow_per_h must be > 0, got {k_swallow_per_h}")
    if route not in ("sublingual", "buccal"):
        raise ValueError(f"route must be 'sublingual' or 'buccal', got {route!r}")

    # Buccal is slightly slower than sublingual (more saliva contact, thicker mucosa)
    ka_eff = ka_mucosal_per_h if route == "sublingual" else ka_mucosal_per_h / 1.5

    ke = cl_L_per_h / vd_L  # elimination rate constant

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    m_mucosal = np.zeros(n_steps)
    c_plasma = np.zeros(n_steps)

    m_mucosal[0] = dose_mg
    c_plasma[0] = 0.0

    for i in range(1, n_steps):
        dt = times[i] - times[i - 1]
        dm = -(ka_eff + k_swallow_per_h) * m_mucosal[i - 1]
        dc = (ka_eff * m_mucosal[i - 1]) / vd_L - ke * c_plasma[i - 1]
        m_mucosal[i] = max(m_mucosal[i - 1] + dm * dt, 0.0)
        c_plasma[i] = max(c_plasma[i - 1] + dc * dt, 0.0)

    # --- Derived PK metrics ---
    cmax = float(np.max(c_plasma))
    tmax_idx = int(np.argmax(c_plasma))
    tmax = float(times[tmax_idx])
    auc = float(np_trapz(c_plasma, times))

    # Fraction absorbed through mucosa (vs swallowed)
    f_absorbed_mucosal = ka_eff / (ka_eff + k_swallow_per_h)

    # Onset time: time to reach 20% of Cmax
    threshold_20 = 0.20 * cmax
    onset_indices = np.where(c_plasma >= threshold_20)[0]
    onset_time = float(times[onset_indices[0]]) if len(onset_indices) > 0 else tmax

    notes: list[str] = []
    notes.append(f"Route: {route}; effective ka = {ka_eff:.2f} /h")
    notes.append(
        f"No hepatic first-pass; f_mucosal = {f_absorbed_mucosal:.2%} absorbed through mucosa"
    )
    if f_absorbed_mucosal < 0.5:
        notes.append(
            "High swallowing rate — substantial fraction swallowed; consider slower formulation"
        )

    return SublingualPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times.tolist(),
        m_mucosal_mg=m_mucosal.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed_mucosal=f_absorbed_mucosal,
        onset_time_h=onset_time,
        notes=notes,
    )


def compare_mucosal_oral(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_hepatic: float,
    ka_mucosal_per_h: float = 2.0,
    k_swallow_per_h: float = 0.5,
    route: str = "sublingual",
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> dict:
    """Compare sublingual/buccal vs swallowed oral PK.

    The oral route applies hepatic first-pass:  F_oral = 1 - f_hepatic
    The mucosal route bypasses hepatic first-pass (only systemic CL applies).

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    f_hepatic:
        Hepatic extraction fraction (0-1).  f_hepatic=0 → no first-pass.
    ka_mucosal_per_h:
        Mucosal absorption rate constant for sublingual/buccal route.
    k_swallow_per_h:
        Swallowing rate constant.
    route:
        "sublingual" or "buccal".
    t_end_h:
        Simulation duration in hours.
    dt_h:
        ODE step size in hours.

    Returns
    -------
    dict with keys:
        sublingual_result : SublingualPKResult
        oral_result       : SublingualPKResult (simulates oral with F_oral)
        relative_bioavailability : float  (sublingual AUC / oral AUC)
        f_hepatic         : float
        f_oral            : float
    """
    if not 0.0 <= f_hepatic <= 1.0:
        raise ValueError(f"f_hepatic must be in [0, 1], got {f_hepatic}")

    sublingual_result = simulate_sublingual_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_mucosal_per_h=ka_mucosal_per_h,
        k_swallow_per_h=k_swallow_per_h,
        route=route,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Oral route: first-pass reduces effective bioavailable dose
    f_oral = 1.0 - f_hepatic
    effective_oral_dose = dose_mg * f_oral

    # Use a typical oral ka (faster than mucosal but with first-pass applied to dose)
    ka_oral = 1.0  # typical oral ka 1/h; first-pass captured via dose scaling

    oral_result = simulate_sublingual_pk(
        drug_name=f"{drug_name}_oral",
        dose_mg=effective_oral_dose,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_mucosal_per_h=ka_oral,
        k_swallow_per_h=0.01,  # negligible swallowing loss (already swallowed)
        route="sublingual",  # use sublingual model mechanics, dose already scaled
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    sl_auc = sublingual_result.auc_mg_h_per_L
    oral_auc = oral_result.auc_mg_h_per_L
    relative_ba = sl_auc / oral_auc if oral_auc > 0 else float("inf")

    return {
        "sublingual_result": sublingual_result,
        "oral_result": oral_result,
        "relative_bioavailability": relative_ba,
        "f_hepatic": f_hepatic,
        "f_oral": f_oral,
    }
