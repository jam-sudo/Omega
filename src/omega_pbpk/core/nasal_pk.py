"""Nasal drug delivery pharmacokinetics (Phase 297).

Models intranasal drug absorption (nasal spray) using a 2-compartment
forward-Euler ODE system:
  - Nasal depot compartment: drug deposited on nasal mucosa
  - Systemic plasma compartment: absorbs from nasal depot

Key features:
  - Fast absorption via rich nasal blood supply
  - Bypasses hepatic first-pass metabolism
  - Mucociliary clearance (MCC) competes with absorption, removing drug from nasal depot
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class NasalPKResult:
    """Result of a nasal drug delivery PK simulation."""

    drug_name: str
    dose_mg: float
    fraction_absorbed: float
    ka_nasal_per_h: float
    k_mucociliary_per_h: float
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_bioavailable: float  # fraction reaching systemic circulation vs lost to MCC
    notes: str


def simulate_nasal_pk(
    drug_name: str,
    dose_mg: float,
    fraction_absorbed: float = 0.8,
    ka_nasal_per_h: float = 6.0,
    k_mucociliary_per_h: float = 20.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    lag_time_h: float = 0.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> NasalPKResult:
    """Simulate nasal drug delivery pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg:
        Total intranasal dose in mg.
    fraction_absorbed:
        Fraction of dose that deposits on nasal mucosa (rest swallowed/lost).
        Must be in (0, 1].
    ka_nasal_per_h:
        First-order absorption rate constant from nasal depot to plasma (1/h).
    k_mucociliary_per_h:
        Mucociliary clearance rate constant removing drug from nasal depot (1/h).
        Drug cleared by MCC is removed from the system (swallowed or expelled).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    lag_time_h:
        Lag time before absorption begins (h). 0 = no lag.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler integration step size (h).

    Returns
    -------
    NasalPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < fraction_absorbed <= 1):
        raise ValueError(f"fraction_absorbed must be in (0, 1], got {fraction_absorbed}")
    if ka_nasal_per_h <= 0:
        raise ValueError(f"ka_nasal_per_h must be > 0, got {ka_nasal_per_h}")
    if k_mucociliary_per_h < 0:
        raise ValueError(f"k_mucociliary_per_h must be >= 0, got {k_mucociliary_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if lag_time_h < 0:
        raise ValueError(f"lag_time_h must be >= 0, got {lag_time_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # Initial conditions
    # fraction_absorbed defines nasal depot; rest is immediately lost
    a_nasal_0 = dose_mg * fraction_absorbed  # mg in nasal depot
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    k_out_nasal = ka_nasal_per_h + k_mucociliary_per_h  # total nasal loss rate

    a_nasal = np.zeros(n_steps + 1)  # mg in nasal depot
    c_plasma = np.zeros(n_steps + 1)  # mg/L in plasma

    a_nasal[0] = a_nasal_0

    for i in range(n_steps):
        t_cur = times[i]
        # Apply lag time: no absorption before lag_time_h
        if t_cur < lag_time_h:
            # No change in nasal depot or plasma during lag phase
            a_nasal[i + 1] = a_nasal[i]
            c_plasma[i + 1] = c_plasma[i]
            continue

        # dA_nasal/dt = -(ka_nasal + k_mc) * A_nasal
        da_nasal = -k_out_nasal * a_nasal[i]
        # dC_plasma/dt = ka_nasal * A_nasal / Vd - (CL/Vd) * C_plasma
        dc_plasma = ka_nasal_per_h * a_nasal[i] / vd_L - ke * c_plasma[i]

        a_nasal[i + 1] = max(a_nasal[i] + da_nasal * dt_h, 0.0)
        c_plasma[i + 1] = max(c_plasma[i] + dc_plasma * dt_h, 0.0)

    # --- PK metrics ---
    auc = float(np_trapz(c_plasma, times))
    cmax = float(np.max(c_plasma))
    tmax_idx = int(np.argmax(c_plasma))
    tmax_h = float(times[tmax_idx])

    # f_bioavailable: fraction of nasal depot that reached systemic circulation
    # = ka / (ka + k_mc) * (1 - fraction lost before absorption)
    # Using analytical ratio of rate constants
    if k_out_nasal > 0:
        f_bioavailable = ka_nasal_per_h / k_out_nasal
    else:
        f_bioavailable = 1.0

    # Build notes
    note_parts: list[str] = []
    if k_mucociliary_per_h > ka_nasal_per_h:
        note_parts.append("MCC rate > absorption rate; majority of nasal dose lost to clearance")
    if lag_time_h > 0:
        note_parts.append(f"Lag time of {lag_time_h:.2f} h applied before absorption onset")
    if fraction_absorbed < 1.0:
        note_parts.append(
            f"{(1 - fraction_absorbed) * 100:.0f}% of dose swallowed/lost before nasal deposition"
        )
    notes = "; ".join(note_parts) if note_parts else "Normal nasal absorption"

    return NasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fraction_absorbed=fraction_absorbed,
        ka_nasal_per_h=ka_nasal_per_h,
        k_mucociliary_per_h=k_mucociliary_per_h,
        times_h=times.tolist(),
        c_plasma=c_plasma.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_bioavailable=f_bioavailable,
        notes=notes,
    )


def compare_nasal_vs_oral(
    drug_name: str,
    dose_mg: float,
    ka_oral_per_h: float,
    ka_nasal_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 0.3,
    f_nasal: float = 0.8,
    k_mucociliary_per_h: float = 20.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare nasal vs oral route PK for the same drug and dose.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg (same for both routes).
    ka_oral_per_h:
        Oral absorption rate constant (1/h).
    ka_nasal_per_h:
        Nasal absorption rate constant (1/h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    f_oral:
        Oral bioavailability fraction (0-1); accounts for first-pass.
    f_nasal:
        Nasal fraction absorbed onto mucosa (0-1]; no first-pass.
    k_mucociliary_per_h:
        Mucociliary clearance rate (1/h) for nasal route.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    dict with keys:
        nasal : NasalPKResult
        oral_cmax : float
        oral_tmax_h : float
        oral_auc : float
        nasal_vs_oral_cmax_ratio : float
        nasal_vs_oral_auc_ratio : float
    """
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # Nasal simulation
    nasal_result = simulate_nasal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fraction_absorbed=f_nasal,
        ka_nasal_per_h=ka_nasal_per_h,
        k_mucociliary_per_h=k_mucociliary_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Oral 1-compartment simulation (accounts for f_oral bioavailability)
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    ke = cl_L_per_h / vd_L

    a_gut = dose_mg * f_oral  # bioavailable oral dose
    c_oral = np.zeros(n_steps + 1)

    for i in range(n_steps):
        da_gut = -ka_oral_per_h * a_gut
        dc = ka_oral_per_h * a_gut / vd_L - ke * c_oral[i]
        a_gut = max(a_gut + da_gut * dt_h, 0.0)
        c_oral[i + 1] = max(c_oral[i] + dc * dt_h, 0.0)

    oral_auc = float(np_trapz(c_oral, times))
    oral_cmax = float(np.max(c_oral))
    oral_tmax_h = float(times[int(np.argmax(c_oral))])

    cmax_ratio = nasal_result.cmax / oral_cmax if oral_cmax > 0 else float("inf")
    auc_ratio = nasal_result.auc / oral_auc if oral_auc > 0 else float("inf")

    return {
        "nasal": nasal_result,
        "oral_cmax": oral_cmax,
        "oral_tmax_h": oral_tmax_h,
        "oral_auc": oral_auc,
        "nasal_vs_oral_cmax_ratio": cmax_ratio,
        "nasal_vs_oral_auc_ratio": auc_ratio,
    }


__all__ = [
    "NasalPKResult",
    "simulate_nasal_pk",
    "compare_nasal_vs_oral",
]
