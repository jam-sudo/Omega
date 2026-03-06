"""Therapeutic window analysis — MEC/MTC, time-in-range, risk assessment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TherapeuticWindowResult:
    drug_name: str
    mec_mg_L: float          # minimum effective concentration
    mtc_mg_L: float          # minimum toxic concentration
    cmax_mg_L: float
    cmin_mg_L: float
    time_in_range_h: float   # time with MEC <= C <= MTC
    time_above_mtc_h: float  # time above toxic level
    time_below_mec_h: float  # time below effective level
    pct_time_in_range: float
    therapeutic_ratio: float # MTC/MEC
    auc_therapeutic: float   # AUC while in range
    risk_level: str          # safe / caution / toxic
    recommendation: str


def analyze_therapeutic_window(
    drug_name: str,
    times_h: list[float],
    conc_mg_L: list[float],
    mec_mg_L: float,
    mtc_mg_L: float,
) -> TherapeuticWindowResult:
    """Analyze drug concentration profile vs therapeutic window.

    Parameters
    ----------
    mec_mg_L : float
        Minimum effective concentration (mg/L).
    mtc_mg_L : float
        Minimum toxic concentration (mg/L).
    """
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be > mec_mg_L")

    t = np.asarray(times_h, dtype=float)
    c = np.asarray(conc_mg_L, dtype=float)
    n = len(t)

    cmax = float(np.max(c))
    cmin = float(np.min(c))

    # Compute times in each zone via trapezoidal integration of indicator
    in_range = ((c >= mec_mg_L) & (c <= mtc_mg_L)).astype(float)
    above_mtc = (c > mtc_mg_L).astype(float)
    below_mec = (c < mec_mg_L).astype(float)

    total_time = float(t[-1] - t[0]) if n > 1 else 1.0
    time_in_range = float(np_trapz(in_range, t))
    time_above = float(np_trapz(above_mtc, t))
    time_below = float(np_trapz(below_mec, t))

    pct_in_range = 100.0 * time_in_range / max(total_time, 1e-9)
    therapeutic_ratio = mtc_mg_L / mec_mg_L

    # AUC while in range
    c_in_range = np.where(in_range > 0.5, c, 0.0)
    auc_therapeutic = float(np_trapz(c_in_range, t))

    # Risk classification
    if time_above > 0.5 * total_time:
        risk = "toxic"
        recommendation = "Reduce dose or extend dosing interval"
    elif cmax > mtc_mg_L:
        risk = "caution"
        recommendation = "Monitor closely; consider dose reduction"
    elif cmin < mec_mg_L and pct_in_range < 50:
        risk = "caution"
        recommendation = "Consider more frequent dosing or sustained-release formulation"
    else:
        risk = "safe"
        recommendation = "Continue current regimen"

    return TherapeuticWindowResult(
        drug_name=drug_name,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        cmax_mg_L=cmax,
        cmin_mg_L=cmin,
        time_in_range_h=time_in_range,
        time_above_mtc_h=time_above,
        time_below_mec_h=time_below,
        pct_time_in_range=pct_in_range,
        therapeutic_ratio=therapeutic_ratio,
        auc_therapeutic=auc_therapeutic,
        risk_level=risk,
        recommendation=recommendation,
    )


def optimal_dose_for_window(
    drug_name: str,
    times_h: list[float],
    conc_per_mg: list[float],  # concentration per mg dose (linear PK)
    mec_mg_L: float,
    mtc_mg_L: float,
    dose_min_mg: float = 1.0,
    dose_max_mg: float = 1000.0,
    n_steps: int = 50,
) -> dict:
    """Find dose that maximises time-in-therapeutic-range.

    Assumes linear PK: conc = dose * conc_per_mg.
    """
    doses = np.linspace(dose_min_mg, dose_max_mg, n_steps)
    best_dose = dose_min_mg
    best_pct = -1.0

    for dose in doses:
        conc = [cp * dose for cp in conc_per_mg]
        r = analyze_therapeutic_window(drug_name, times_h, conc, mec_mg_L, mtc_mg_L)
        if r.pct_time_in_range > best_pct:
            best_pct = r.pct_time_in_range
            best_dose = float(dose)

    return {
        "optimal_dose_mg": best_dose,
        "pct_time_in_range": best_pct,
        "drug_name": drug_name,
    }


__all__ = [
    "TherapeuticWindowResult",
    "analyze_therapeutic_window",
    "optimal_dose_for_window",
]
