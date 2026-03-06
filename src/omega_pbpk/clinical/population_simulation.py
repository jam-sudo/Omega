"""Monte Carlo population PK simulation with inter-individual variability.

Distinct from population_pk.py (Standard Two-Stage popPK fitting).
This module performs forward simulation of a virtual population using
log-normal BSV on PK parameters.
"""

from __future__ import annotations

__all__ = [
    "PopulationSimResult",
    "simulate_population_pk",
    "summarize_population_pk",
]

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PopulationSimResult:
    """Summary statistics from a population PK simulation."""

    drug_name: str
    n_subjects: int
    dose_mg: float
    route: str
    cmax_mean: float
    cmax_cv_pct: float
    cmax_p5: float
    cmax_p95: float
    auc_mean: float
    auc_cv_pct: float
    auc_p5: float
    auc_p95: float
    t_half_mean_h: float
    cl_mean_L_per_h: float
    vd_mean_L: float
    notes: str


def simulate_population_pk(
    drug_name: str,
    dose_mg: float,
    cl_typical_L_per_h: float,
    vd_typical_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    omega_cl: float = 0.3,  # BSV on CL (CV, not variance)
    omega_vd: float = 0.2,  # BSV on Vd
    omega_ka: float = 0.4,  # BSV on Ka (if oral)
    n_subjects: int = 100,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    seed: int = 42,
) -> PopulationSimResult:
    """Simulate population PK using 1-compartment analytical solutions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    cl_typical_L_per_h:
        Typical (population mean) clearance in L/h.
    vd_typical_L:
        Typical (population mean) volume of distribution in L.
    route:
        Dosing route: "oral" or "iv".
    ka_per_h:
        Typical first-order absorption rate constant (oral only) in 1/h.
    f_oral:
        Oral bioavailability fraction (0–1).
    omega_cl:
        Between-subject variability (CV) on CL (log-normal).
    omega_vd:
        Between-subject variability (CV) on Vd (log-normal).
    omega_ka:
        Between-subject variability (CV) on Ka (log-normal, oral only).
    n_subjects:
        Number of simulated subjects.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours (not used in analytical solution, kept for API compatibility).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationSimResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_typical_L_per_h <= 0:
        raise ValueError(f"cl_typical_L_per_h must be > 0, got {cl_typical_L_per_h}")
    if vd_typical_L <= 0:
        raise ValueError(f"vd_typical_L must be > 0, got {vd_typical_L}")
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    if omega_cl < 0:
        raise ValueError(f"omega_cl must be >= 0, got {omega_cl}")
    if omega_vd < 0:
        raise ValueError(f"omega_vd must be >= 0, got {omega_vd}")
    if omega_ka < 0:
        raise ValueError(f"omega_ka must be >= 0, got {omega_ka}")

    rng = np.random.default_rng(seed)

    # Log-normal BSV: individual_param = typical * exp(eta), eta ~ N(0, omega^2)
    cl_i = cl_typical_L_per_h * np.exp(rng.normal(0.0, omega_cl, n_subjects))
    vd_i = vd_typical_L * np.exp(rng.normal(0.0, omega_vd, n_subjects))
    ka_i = ka_per_h * np.exp(rng.normal(0.0, omega_ka, n_subjects))

    cmaxes: list[float] = []
    aucs: list[float] = []
    t_halfs: list[float] = []

    for cl, vd, ka in zip(cl_i, vd_i, ka_i, strict=False):
        ke = cl / vd
        t_half = 0.693 / ke

        if route == "iv":
            cmax = dose_mg / vd
            auc = dose_mg / cl
        else:
            # Oral 1-cpt: C(t) = (D*F*ka) / (Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
            if abs(ka - ke) < 1e-6:
                ka = ka + 1e-6
            tmax = np.log(ka / ke) / (ka - ke)
            # Ensure tmax is positive
            if tmax < 0:
                tmax = 0.0
            cmax = (
                (dose_mg * f_oral * ka)
                / (vd * (ka - ke))
                * (np.exp(-ke * tmax) - np.exp(-ka * tmax))
            )
            cmax = abs(cmax)
            auc = dose_mg * f_oral / cl

        cmaxes.append(cmax)
        aucs.append(auc)
        t_halfs.append(t_half)

    cmax_arr = np.array(cmaxes)
    auc_arr = np.array(aucs)
    t_half_arr = np.array(t_halfs)

    cmax_mean = float(np.mean(cmax_arr))
    cmax_cv = float(np.std(cmax_arr) / np.mean(cmax_arr) * 100.0)
    cmax_p5 = float(np.percentile(cmax_arr, 5))
    cmax_p95 = float(np.percentile(cmax_arr, 95))

    auc_mean = float(np.mean(auc_arr))
    auc_cv = float(np.std(auc_arr) / np.mean(auc_arr) * 100.0)
    auc_p5 = float(np.percentile(auc_arr, 5))
    auc_p95 = float(np.percentile(auc_arr, 95))

    t_half_mean = float(np.mean(t_half_arr))

    notes = (
        f"n={n_subjects} subjects; route={route}; "
        f"omega_cl={omega_cl:.2f}, omega_vd={omega_vd:.2f}, omega_ka={omega_ka:.2f}; "
        f"seed={seed}"
    )

    return PopulationSimResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        route=route,
        cmax_mean=cmax_mean,
        cmax_cv_pct=cmax_cv,
        cmax_p5=cmax_p5,
        cmax_p95=cmax_p95,
        auc_mean=auc_mean,
        auc_cv_pct=auc_cv,
        auc_p5=auc_p5,
        auc_p95=auc_p95,
        t_half_mean_h=t_half_mean,
        cl_mean_L_per_h=float(np.mean(cl_i)),
        vd_mean_L=float(np.mean(vd_i)),
        notes=notes,
    )


def summarize_population_pk(results: list[dict[str, Any]]) -> PopulationSimResult:
    """Summarize pre-computed Cmax/AUC list into a PopulationSimResult.

    Each dict in results must have keys: "cmax" and "auc".
    Optional keys: "drug_name", "dose_mg", "route", "cl", "vd", "t_half".

    Parameters
    ----------
    results:
        List of per-subject PK result dicts.

    Returns
    -------
    PopulationSimResult
    """
    if not results:
        raise ValueError("results list must not be empty")

    cmax_arr = np.array([r["cmax"] for r in results])
    auc_arr = np.array([r["auc"] for r in results])

    drug_name = str(results[0].get("drug_name", "Unknown"))
    dose_mg = float(results[0].get("dose_mg", 0.0))
    route = str(results[0].get("route", "unknown"))

    cl_vals = [r["cl"] for r in results if "cl" in r]
    vd_vals = [r["vd"] for r in results if "vd" in r]
    t_half_vals = [r["t_half"] for r in results if "t_half" in r]

    cl_mean = float(np.mean(cl_vals)) if cl_vals else 0.0
    vd_mean = float(np.mean(vd_vals)) if vd_vals else 0.0
    t_half_mean = float(np.mean(t_half_vals)) if t_half_vals else 0.0

    n = len(results)
    cmax_mean = float(np.mean(cmax_arr))
    cmax_cv = float(np.std(cmax_arr) / cmax_mean * 100.0) if cmax_mean > 0 else 0.0
    auc_mean = float(np.mean(auc_arr))
    auc_cv = float(np.std(auc_arr) / auc_mean * 100.0) if auc_mean > 0 else 0.0

    return PopulationSimResult(
        drug_name=drug_name,
        n_subjects=n,
        dose_mg=dose_mg,
        route=route,
        cmax_mean=cmax_mean,
        cmax_cv_pct=cmax_cv,
        cmax_p5=float(np.percentile(cmax_arr, 5)),
        cmax_p95=float(np.percentile(cmax_arr, 95)),
        auc_mean=auc_mean,
        auc_cv_pct=auc_cv,
        auc_p5=float(np.percentile(auc_arr, 5)),
        auc_p95=float(np.percentile(auc_arr, 95)),
        t_half_mean_h=t_half_mean,
        cl_mean_L_per_h=cl_mean,
        vd_mean_L=vd_mean,
        notes=f"Summarized from {n} pre-computed subjects",
    )
