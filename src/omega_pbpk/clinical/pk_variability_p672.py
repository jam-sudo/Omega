"""Population pharmacokinetic variability simulator.

Monte Carlo simulation of inter-individual pharmacokinetic variability
using log-normal distributions for clearance and volume of distribution.
Uses only the Python standard library (random, math).

References
----------
- Rowland M, Tozer TN. Clinical Pharmacokinetics (population PK concepts)
- Ette EI, Williams PJ. Pharmacometrics (Monte Carlo simulation in PK)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PKVariabilityResult:
    """Result of population PK variability simulation."""

    drug_name: str
    n_subjects: int
    mean_cl_L_per_h: float
    cv_cl_pct: float
    mean_vd_L: float
    cv_vd_pct: float
    dose_mg: float
    cmax_mean_mg_L: float
    cmax_cv_pct: float
    cmax_5th_pct_mg_L: float
    cmax_95th_pct_mg_L: float
    auc_mean_mg_h_per_L: float
    auc_cv_pct: float
    auc_5th_pct_mg_h_per_L: float
    auc_95th_pct_mg_h_per_L: float
    n_below_mec: int
    n_above_mtc: int
    therapeutic_window_coverage_pct: float
    notes: str


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mu: float) -> float:
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _cv_pct(values: list[float]) -> float:
    mu = _mean(values)
    if mu == 0:
        return 0.0
    return _std(values, mu) / mu * 100


def _percentile(sorted_values: list[float], p: float) -> float:
    idx = int(p * len(sorted_values))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def simulate_pk_variability(
    drug_name: str,
    dose_mg: float,
    mean_cl_L_per_h: float,
    cv_cl_pct: float,
    mean_vd_L: float,
    cv_vd_pct: float,
    ka_per_h: float = 1.5,
    n_subjects: int = 1000,
    mec_mg_L: float = 0.1,
    mtc_mg_L: float = 2.0,
    seed: int = 42,
) -> PKVariabilityResult:
    """Simulate population PK variability via Monte Carlo.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg (must be > 0).
    mean_cl_L_per_h : float
        Population mean clearance in L/h (must be > 0).
    cv_cl_pct : float
        Coefficient of variation for clearance in % (>= 0).
    mean_vd_L : float
        Population mean volume of distribution in L (must be > 0).
    cv_vd_pct : float
        Coefficient of variation for Vd in % (>= 0).
    ka_per_h : float
        Absorption rate constant (1/h).
    n_subjects : int
        Number of virtual subjects (>= 10).
    mec_mg_L : float
        Minimum effective concentration in mg/L (must be > 0).
    mtc_mg_L : float
        Maximum tolerated concentration in mg/L (must be > mec).
    seed : int
        Random seed for reproducibility.
    """
    # --- validation ---
    if n_subjects < 10:
        raise ValueError("n_subjects must be >= 10")
    if cv_cl_pct < 0:
        raise ValueError("cv_cl_pct must be >= 0")
    if cv_vd_pct < 0:
        raise ValueError("cv_vd_pct must be >= 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mean_cl_L_per_h <= 0:
        raise ValueError("mean_cl_L_per_h must be > 0")
    if mean_vd_L <= 0:
        raise ValueError("mean_vd_L must be > 0")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be > mec_mg_L")

    rng = random.Random(seed)

    omega_cl = cv_cl_pct / 100.0
    omega_vd = cv_vd_pct / 100.0

    cmax_values: list[float] = []
    auc_values: list[float] = []

    ka = ka_per_h
    bioavail = 0.85

    for _ in range(n_subjects):
        eta_cl = rng.gauss(0, omega_cl) if omega_cl > 0 else 0.0
        eta_vd = rng.gauss(0, omega_vd) if omega_vd > 0 else 0.0
        cl_i = mean_cl_L_per_h * math.exp(eta_cl)
        vd_i = mean_vd_L * math.exp(eta_vd)

        k_el = cl_i / vd_i

        if abs(ka - k_el) < 1e-6:
            tmax = 1.0 / ka
        else:
            tmax = math.log(ka / k_el) / (ka - k_el)

        cmax = (
            (
                dose_mg
                * bioavail
                / vd_i
                * ka
                / (ka - k_el)
                * (math.exp(-k_el * tmax) - math.exp(-ka * tmax))
            )
            if abs(ka - k_el) >= 1e-6
            else (dose_mg * bioavail / vd_i * ka * tmax * math.exp(-ka * tmax))
        )
        cmax = max(0.0, cmax)

        auc = dose_mg * bioavail / cl_i

        cmax_values.append(cmax)
        auc_values.append(auc)

    # --- statistics ---
    cmax_mean = _mean(cmax_values)
    cmax_cv = _cv_pct(cmax_values)
    auc_mean = _mean(auc_values)
    auc_cv = _cv_pct(auc_values)

    sorted_cmax = sorted(cmax_values)
    sorted_auc = sorted(auc_values)

    cmax_5th = _percentile(sorted_cmax, 0.05)
    cmax_95th = _percentile(sorted_cmax, 0.95)
    auc_5th = _percentile(sorted_auc, 0.05)
    auc_95th = _percentile(sorted_auc, 0.95)

    # --- therapeutic window ---
    n_below_mec = sum(1 for c in cmax_values if c < mec_mg_L)
    n_above_mtc = sum(1 for c in cmax_values if c > mtc_mg_L)
    n_in_window = sum(1 for c in cmax_values if mec_mg_L <= c <= mtc_mg_L)
    therapeutic_window_coverage_pct = n_in_window / n_subjects * 100

    return PKVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        mean_cl_L_per_h=mean_cl_L_per_h,
        cv_cl_pct=cv_cl_pct,
        mean_vd_L=mean_vd_L,
        cv_vd_pct=cv_vd_pct,
        dose_mg=dose_mg,
        cmax_mean_mg_L=cmax_mean,
        cmax_cv_pct=cmax_cv,
        cmax_5th_pct_mg_L=cmax_5th,
        cmax_95th_pct_mg_L=cmax_95th,
        auc_mean_mg_h_per_L=auc_mean,
        auc_cv_pct=auc_cv,
        auc_5th_pct_mg_h_per_L=auc_5th,
        auc_95th_pct_mg_h_per_L=auc_95th,
        n_below_mec=n_below_mec,
        n_above_mtc=n_above_mtc,
        therapeutic_window_coverage_pct=therapeutic_window_coverage_pct,
        notes=f"Monte Carlo PK simulation for {drug_name} with {n_subjects} subjects.",
    )
