"""
Phase 890 — Enterohepatic Circulation (EHC)

Models enterohepatic recirculation of drugs causing multiple plasma peaks.
Pure Python / stdlib only (math, dataclasses). No numpy, no scipy.
Forward Euler ODE integration. Manual trapezoidal AUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EHCResult:
    """Result of an enterohepatic circulation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    n_secondary_peaks: int
    secondary_peak_times_h: list
    secondary_peak_concs: list
    f_ehc: float
    bile_recycling_amplification: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC via the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def _find_cmax_tmax(times: list, concs: list) -> tuple:
    """Return (cmax, tmax_h) for the given concentration profile."""
    cmax = 0.0
    tmax_h = 0.0
    for t, c in zip(times, concs):
        if c > cmax:
            cmax = c
            tmax_h = t
    return cmax, tmax_h


def _detect_secondary_peaks(times: list, concs: list, tmax_idx: int) -> tuple:
    """
    Detect local maxima after the first peak.

    Returns (secondary_peak_times_h, secondary_peak_concs).
    """
    sec_times = []
    sec_concs = []
    n = len(concs)
    for i in range(tmax_idx + 1, n - 1):
        if concs[i] > concs[i - 1] and concs[i] > concs[i + 1]:
            sec_times.append(times[i])
            sec_concs.append(concs[i])
    return sec_times, sec_concs


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    k_abs_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    f_bile: float = 0.3,
    k_bile_per_h: float = 0.2,
    t_bile_transit_h: float = 4.0,
    f_reabs: float = 0.6,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EHCResult:
    """
    Simulate enterohepatic circulation using Forward Euler.

    Compartments:
        a_gut          -- amount in GI / oral absorption depot (mg)
        A_plasma       -- amount in plasma (mg)
        A_bile         -- amount in bile duct (mg)
        A_intestine    -- amount in intestinal lumen awaiting re-absorption (mg)

    Parameters
    ----------
    drug_name         : drug name
    dose_mg           : oral dose (mg)
    k_abs_per_h       : first-order absorption rate (1/h)
    cl_L_per_h        : total plasma clearance (L/h)
    vd_L              : volume of distribution (L)
    f_bile            : fraction of plasma drug excreted into bile per unit time (dimensionless rate modifier)
    k_bile_per_h      : rate constant for biliary excretion from plasma (1/h)
    t_bile_transit_h  : bile duct transit time (h); k_release_bile = 1/t_bile_transit_h
    f_reabs           : fraction of intestinal lumen drug re-absorbed
    t_end_h           : simulation end time (h)
    dt_h              : Forward Euler time step (h)

    Returns
    -------
    EHCResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= f_bile <= 1.0):
        raise ValueError("f_bile must be in [0, 1]")
    if not (0.0 <= f_reabs <= 1.0):
        raise ValueError("f_reabs must be in [0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # --- Derived rate constants ---
    ke = cl_L_per_h / vd_L  # plasma elimination (1/h)
    k_release_bile = 1.0 / t_bile_transit_h  # bile duct emptying (1/h)
    k_reabs = 0.5  # intestinal re-absorption transit (1/h)

    # --- Initial conditions ---
    a_gut = dose_mg
    A_plasma = 0.0
    A_bile = 0.0
    A_intestine = 0.0

    times = []
    c_plasma_list = []

    n_steps = int(math.ceil(t_end_h / dt_h))

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_plasma_list.append(A_plasma / vd_L)

        if i < n_steps:
            # da_gut/dt = -k_abs * a_gut + f_reabs * k_reabs * A_intestine
            d_gut = -k_abs_per_h * a_gut + f_reabs * k_reabs * A_intestine
            # dA_plasma/dt = k_abs * a_gut - (ke + f_bile * k_bile_per_h) * A_plasma
            d_plasma = k_abs_per_h * a_gut - (ke + f_bile * k_bile_per_h) * A_plasma
            # dA_bile/dt = f_bile * k_bile_per_h * A_plasma - k_release_bile * A_bile
            d_bile = f_bile * k_bile_per_h * A_plasma - k_release_bile * A_bile
            # dA_intestine/dt = k_release_bile * A_bile - k_reabs * A_intestine
            d_intestine = k_release_bile * A_bile - k_reabs * A_intestine

            a_gut += d_gut * dt_h
            A_plasma += d_plasma * dt_h
            A_bile += d_bile * dt_h
            A_intestine += d_intestine * dt_h

            # Clamp negatives (numerical noise)
            if a_gut < 0.0:
                a_gut = 0.0
            if A_plasma < 0.0:
                A_plasma = 0.0
            if A_bile < 0.0:
                A_bile = 0.0
            if A_intestine < 0.0:
                A_intestine = 0.0

    # --- PK metrics ---
    cmax_plasma, tmax_plasma_h = _find_cmax_tmax(times, c_plasma_list)
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)

    # Find tmax index for secondary peak detection
    tmax_idx = 0
    for idx, c in enumerate(c_plasma_list):
        if c >= cmax_plasma:
            tmax_idx = idx
            break

    sec_times, sec_concs = _detect_secondary_peaks(times, c_plasma_list, tmax_idx)
    n_secondary_peaks = len(sec_times)

    # --- EHC metrics ---
    f_ehc = f_bile * f_reabs
    if f_ehc < 1.0:
        bile_recycling_amplification = 1.0 / (1.0 - f_ehc)
    else:
        bile_recycling_amplification = 10.0

    # --- Notes ---
    if n_secondary_peaks > 1:
        notes = "Multiple EHC cycles detected — prolonged effective half-life"
    elif n_secondary_peaks == 1:
        notes = "Single EHC recycling event detected"
    else:
        notes = "No significant EHC observed"

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        n_secondary_peaks=n_secondary_peaks,
        secondary_peak_times_h=sec_times,
        secondary_peak_concs=sec_concs,
        f_ehc=f_ehc,
        bile_recycling_amplification=bile_recycling_amplification,
        notes=notes,
    )


def compare_ehc_scenarios(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """
    Simulate EHC for multiple parameter scenarios.

    Each scenario dict may contain optional keys:
    "f_bile", "f_reabs", "t_bile_transit_h", "name".

    Returns list of EHCResult sorted by auc_plasma descending.
    """
    results = []
    for scenario in scenarios:
        kwargs = {k: v for k, v in scenario.items() if k != "name"}
        result = simulate_ehc(drug_name=drug_name, dose_mg=dose_mg, **kwargs)
        results.append(result)
    results.sort(key=lambda r: r.auc_plasma, reverse=True)
    return results
