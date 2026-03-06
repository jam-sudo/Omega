"""Population PK simulation — between-subject variability (BSV) and covariate effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = ["PopPKResult", "simulate_pop_pk", "eta_shrinkage"]


@dataclass(frozen=True)
class PopPKResult:
    n_subjects: int
    times_h: list[float]
    median_conc: list[float]
    p5_conc: list[float]
    p95_conc: list[float]
    individual_auc: list[float]
    median_auc: float
    cv_auc_pct: float
    median_cmax: float
    cv_cmax_pct: float
    n_above_therapeutic: int
    n_below_therapeutic: int
    covariate_impacts: dict


def simulate_pop_pk(
    n_subjects: int = 100,
    dose_mg: float = 100.0,
    cl_typical_L_per_h: float = 5.0,
    vd_typical_L: float = 50.0,
    ka_typical_per_h: float = 1.0,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    omega_ka: float = 0.4,
    weight_covariate: bool = True,
    age_covariate: bool = True,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    seed: int = 42,
    therapeutic_low_mg_L: float | None = None,
    therapeutic_high_mg_L: float | None = None,
) -> PopPKResult:
    rng = np.random.default_rng(seed)

    # Time grid
    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.array([i * dt_h for i in range(n_steps + 1)])

    # Random effects (etas)
    eta_cl = rng.standard_normal(n_subjects)
    eta_vd = rng.standard_normal(n_subjects)
    eta_ka = rng.standard_normal(n_subjects)

    # Covariates
    if weight_covariate:
        weights = rng.normal(75.0, 15.0, n_subjects)
        weights = np.clip(weights, 40.0, 150.0)
    else:
        weights = np.full(n_subjects, 70.0)

    if age_covariate:
        ages = rng.normal(50.0, 15.0, n_subjects)
        ages = np.clip(ages, 18.0, 80.0)
    else:
        ages = np.full(n_subjects, 50.0)

    # Individual parameters
    cl_i = cl_typical_L_per_h * (weights / 70.0) ** 0.75 * np.exp(omega_cl * eta_cl)
    vd_i = vd_typical_L * np.exp(-0.01 * (ages - 50.0)) * np.exp(omega_vd * eta_vd)
    ka_i = ka_typical_per_h * np.exp(omega_ka * eta_ka)

    # Simulate each subject (forward Euler, 1-compartment)
    conc_matrix = np.zeros((n_subjects, len(times)))
    cmax_vals = np.zeros(n_subjects)
    auc_vals = np.zeros(n_subjects)

    for i in range(n_subjects):
        ke = cl_i[i] / vd_i[i]
        gut = dose_mg if route == "oral" else 0.0
        central = 0.0 if route == "oral" else dose_mg

        for j, _t in enumerate(times):
            conc_matrix[i, j] = central / vd_i[i]
            if j < len(times) - 1:
                if route == "oral":
                    d_gut = -ka_i[i] * gut * dt_h
                    d_central = (ka_i[i] * gut - ke * central) * dt_h
                    gut += d_gut
                    central += d_central
                else:
                    d_central = -ke * central * dt_h
                    central += d_central

        profile = conc_matrix[i, :]
        cmax_vals[i] = float(np.max(profile))
        auc_vals[i] = float(np_trapz(profile, times))

    # Percentiles across subjects at each time point
    median_conc = np.percentile(conc_matrix, 50, axis=0)
    p5_conc = np.percentile(conc_matrix, 5, axis=0)
    p95_conc = np.percentile(conc_matrix, 95, axis=0)

    median_auc = float(np.median(auc_vals))
    cv_auc_pct = (
        float(np.std(auc_vals, ddof=1) / np.mean(auc_vals) * 100.0)
        if np.mean(auc_vals) > 0
        else 0.0
    )
    median_cmax = float(np.median(cmax_vals))
    cv_cmax_pct = (
        float(np.std(cmax_vals, ddof=1) / np.mean(cmax_vals) * 100.0)
        if np.mean(cmax_vals) > 0
        else 0.0
    )

    # Therapeutic window assessment
    n_above = 0
    n_below = 0
    if therapeutic_high_mg_L is not None:
        n_above = int(np.sum(cmax_vals > therapeutic_high_mg_L))
    if therapeutic_low_mg_L is not None:
        # Check if any subject's Cmax is below the lower bound
        n_below = int(np.sum(cmax_vals < therapeutic_low_mg_L))

    # Covariate impact (Pearson correlation between log(AUC) and covariate)
    covariate_impacts: dict[str, float] = {}
    log_auc = np.log(auc_vals + 1e-30)
    if weight_covariate:
        corr_w = float(np.corrcoef(log_auc, weights)[0, 1])
        covariate_impacts["weight"] = corr_w
    if age_covariate:
        corr_a = float(np.corrcoef(log_auc, ages)[0, 1])
        covariate_impacts["age"] = corr_a

    return PopPKResult(
        n_subjects=n_subjects,
        times_h=times.tolist(),
        median_conc=median_conc.tolist(),
        p5_conc=p5_conc.tolist(),
        p95_conc=p95_conc.tolist(),
        individual_auc=auc_vals.tolist(),
        median_auc=median_auc,
        cv_auc_pct=cv_auc_pct,
        median_cmax=median_cmax,
        cv_cmax_pct=cv_cmax_pct,
        n_above_therapeutic=n_above,
        n_below_therapeutic=n_below,
        covariate_impacts=covariate_impacts,
    )


def eta_shrinkage(
    individual_etas: list[float],
    omega: float,
) -> float:
    """Compute eta shrinkage: 1 - SD(eta) / omega."""
    if omega == 0:
        return 1.0
    sd = float(np.std(individual_etas, ddof=1)) if len(individual_etas) > 1 else 0.0
    return 1.0 - sd / omega
