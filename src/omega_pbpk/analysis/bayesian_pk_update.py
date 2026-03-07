"""Bayesian MAP PK parameter updating from observed concentrations (grid search, no scipy)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BayesianPKResult:
    drug_name: str
    n_observations: int
    prior_cl_L_per_h: float
    prior_vd_L: float
    posterior_cl_L_per_h: float
    posterior_vd_L: float
    posterior_ka_per_h: float | None
    cl_shrinkage_pct: float
    vd_shrinkage_pct: float
    predicted_concentrations: tuple
    residuals: tuple
    rmse: float
    individual_auc_mg_h_L: float
    individual_t_half_h: float
    notes: str


def predict_1cpt(
    times_h: list,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
) -> list:
    """Predict 1-compartment concentrations at given times."""
    ke = cl_L_per_h / vd_L
    concs = []
    for t in times_h:
        if route == "iv":
            c = (dose_mg / vd_L) * math.exp(-ke * t)
        else:
            # Oral: avoid ka == ke singularity
            ka = ka_per_h
            if abs(ka - ke) < 1e-6:
                ka = ke * 1.001
            c = (f_oral * dose_mg * ka / (vd_L * (ka - ke))) * (
                math.exp(-ke * t) - math.exp(-ka * t)
            )
            c = max(0.0, c)
        concs.append(c)
    return concs


def bayesian_update(
    drug_name: str,
    obs_times_h: list,
    obs_conc_mg_L: list,
    prior_cl_L_per_h: float,
    prior_vd_L: float,
    dose_mg: float,
    route: str = "oral",
    prior_ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    omega_cl: float = 0.3,
    omega_vd: float = 0.2,
    sigma_prop: float = 0.15,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> BayesianPKResult:
    """Bayesian MAP estimation of CL and Vd from observed PK data (grid search)."""
    # Validation
    if len(obs_times_h) != len(obs_conc_mg_L):
        raise ValueError("obs_times_h and obs_conc_mg_L must have the same length")
    if len(obs_times_h) < 1:
        raise ValueError("At least 1 observation is required")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if prior_cl_L_per_h <= 0:
        raise ValueError("prior_cl_L_per_h must be > 0")
    if prior_vd_L <= 0:
        raise ValueError("prior_vd_L must be > 0")
    if route not in {"oral", "iv"}:
        raise ValueError("route must be 'oral' or 'iv'")
    if omega_cl <= 0:
        raise ValueError("omega_cl must be > 0")
    if omega_vd <= 0:
        raise ValueError("omega_vd must be > 0")

    # Grid definition
    cl_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
    vd_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
    ka_factors = [0.5, 1.0, 2.0] if route == "oral" else [1.0]

    cl_candidates = [prior_cl_L_per_h * f for f in cl_factors]
    vd_candidates = [prior_vd_L * f for f in vd_factors]
    ka_candidates = [prior_ka_per_h * f for f in ka_factors]

    best_log_post = float("-inf")
    best_cl = prior_cl_L_per_h
    best_vd = prior_vd_L
    best_ka = prior_ka_per_h if route == "oral" else None

    for cl in cl_candidates:
        for vd in vd_candidates:
            for ka in ka_candidates:
                # Predict at observation times
                preds = predict_1cpt(
                    list(obs_times_h),
                    dose_mg,
                    cl,
                    vd,
                    route=route,
                    ka_per_h=ka,
                    f_oral=f_oral,
                )

                # Log likelihood (proportional error)
                log_like = 0.0
                for obs, pred in zip(obs_conc_mg_L, preds):
                    if pred <= 0:
                        continue
                    err_sd = sigma_prop * pred
                    log_like -= 0.5 * ((obs - pred) / err_sd) ** 2

                # Log priors (lognormal)
                log_prior_cl = -0.5 * (math.log(cl / prior_cl_L_per_h)) ** 2 / omega_cl**2
                log_prior_vd = -0.5 * (math.log(vd / prior_vd_L)) ** 2 / omega_vd**2

                log_post = log_like + log_prior_cl + log_prior_vd

                if log_post > best_log_post:
                    best_log_post = log_post
                    best_cl = cl
                    best_vd = vd
                    best_ka = ka if route == "oral" else None

    # Posterior predictions and residuals
    predicted = predict_1cpt(
        list(obs_times_h),
        dose_mg,
        best_cl,
        best_vd,
        route=route,
        ka_per_h=best_ka if best_ka is not None else prior_ka_per_h,
        f_oral=f_oral,
    )
    residuals = [o - p for o, p in zip(obs_conc_mg_L, predicted)]
    rmse = math.sqrt(sum(r**2 for r in residuals) / len(residuals))

    # AUC (analytical)
    if route == "iv":
        auc = dose_mg / best_cl
    else:
        auc = f_oral * dose_mg / best_cl

    # t_half
    t_half = math.log(2) * best_vd / best_cl

    # Shrinkage
    cl_shrinkage = abs(best_cl - prior_cl_L_per_h) / prior_cl_L_per_h * 100.0
    vd_shrinkage = abs(best_vd - prior_vd_L) / prior_vd_L * 100.0

    # Notes
    notes_parts = [
        f"MAP via 5x5{'x3' if route == 'oral' else ''} grid search",
        f"Posterior CL={best_cl:.3f} L/h, Vd={best_vd:.2f} L",
        f"RMSE={rmse:.4f} mg/L",
    ]
    if route == "oral" and best_ka is not None:
        notes_parts.append(f"ka={best_ka:.3f} h-1")

    return BayesianPKResult(
        drug_name=drug_name,
        n_observations=len(obs_times_h),
        prior_cl_L_per_h=prior_cl_L_per_h,
        prior_vd_L=prior_vd_L,
        posterior_cl_L_per_h=best_cl,
        posterior_vd_L=best_vd,
        posterior_ka_per_h=best_ka,
        cl_shrinkage_pct=cl_shrinkage,
        vd_shrinkage_pct=vd_shrinkage,
        predicted_concentrations=tuple(predicted),
        residuals=tuple(residuals),
        rmse=rmse,
        individual_auc_mg_h_L=auc,
        individual_t_half_h=t_half,
        notes="; ".join(notes_parts),
    )
