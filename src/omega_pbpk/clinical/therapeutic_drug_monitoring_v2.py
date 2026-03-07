"""
Therapeutic Drug Monitoring v2 — Enhanced TDM with Bayesian MAP estimation.

MAP estimation via 20x20 grid search over CL and Vd parameter space.
Uses 1-compartment PK model (oral or IV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BayesianTDMResult:
    drug_name: str
    measured_conc_mg_L: float
    sample_time_h: float
    prior_cl_L_per_h: float
    prior_vd_L: float
    map_cl_L_per_h: float
    map_vd_L: float
    map_t_half_h: float
    predicted_trough_mg_L: float
    predicted_peak_mg_L: float
    target_trough_mg_L: float
    target_peak_mg_L: float
    recommended_dose_mg: float
    recommended_interval_h: float
    dose_adjustment_needed: bool
    posterior_cv_cl_pct: float
    notes: str


def _predict_conc_1cpt(
    dose_mg: float,
    dose_time_h: float,
    cl_L_per_h: float,
    vd_L: float,
    t_h: float,
    f_oral: float = 1.0,
    ka_per_h: float | None = None,
) -> float:
    """Predict concentration at time t using 1-compartment model."""
    elapsed = t_h - dose_time_h
    if elapsed <= 0.0:
        return 0.0

    ke = cl_L_per_h / vd_L
    if ke <= 0.0:
        return 0.0

    if ka_per_h is None:
        # IV bolus: C(t) = (Dose/Vd) * exp(-ke * elapsed)
        c = (dose_mg / vd_L) * math.exp(-ke * elapsed)
    else:
        # Oral 1-cpt: C(t) = (F*Dose*ka) / (Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
        if abs(ka_per_h - ke) < 1e-6:
            # Avoid division by zero; flip-flop case
            ka_use = ke + 1e-4
        else:
            ka_use = ka_per_h
        denom = vd_L * (ka_use - ke)
        if abs(denom) < 1e-12:
            return 0.0
        c = (f_oral * dose_mg * ka_use / denom) * (
            math.exp(-ke * elapsed) - math.exp(-ka_use * elapsed)
        )
    return max(c, 0.0)


def _log_posterior(
    cl: float,
    vd: float,
    measured_conc: float,
    predicted_conc: float,
    prior_cl: float,
    prior_vd: float,
    bsv_cl_cv: float,
    bsv_vd_cv: float,
    sigma: float = 0.2,
) -> float:
    """Compute log-posterior = log-likelihood + log-prior."""
    if predicted_conc <= 0.0 or cl <= 0.0 or vd <= 0.0:
        return -1e18

    # Log-likelihood: log-normal error on concentration
    log_ratio_obs = math.log(measured_conc / predicted_conc)
    log_lik = -0.5 * (log_ratio_obs / sigma) ** 2

    # Log-prior: normal on log(CL/prior_CL) with CV=bsv_cl_cv
    log_cl_ratio = math.log(cl / prior_cl)
    log_prior_cl = -0.5 * (log_cl_ratio / bsv_cl_cv) ** 2

    log_vd_ratio = math.log(vd / prior_vd)
    log_prior_vd = -0.5 * (log_vd_ratio / bsv_vd_cv) ** 2

    return log_lik + log_prior_cl + log_prior_vd


def bayesian_tdm_optimization(
    drug_name: str,
    measured_conc_mg_L: float,
    sample_time_h: float,
    dose_mg: float,
    dose_time_h: float = 0.0,
    prior_cl_L_per_h: float = 5.0,
    prior_vd_L: float = 50.0,
    bsv_cl_cv: float = 0.3,
    bsv_vd_cv: float = 0.2,
    target_trough_mg_L: float = 1.0,
    target_peak_mg_L: float = 8.0,
    tau_h: float = 12.0,
    f_oral: float = 1.0,
    ka_per_h: float | None = None,
) -> BayesianTDMResult:
    """
    Bayesian MAP estimation for TDM via 20x20 grid search.

    Returns MAP-estimated PK parameters and recommended dose/interval.
    """
    if measured_conc_mg_L < 0.0:
        raise ValueError(f"measured_conc_mg_L must be non-negative, got {measured_conc_mg_L}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if prior_cl_L_per_h <= 0.0 or prior_vd_L <= 0.0:
        raise ValueError("Prior CL and Vd must be positive")

    # Grid search: 20x20 grid over [prior*0.2, prior*3.0]
    n_grid = 20
    cl_min = prior_cl_L_per_h * 0.2
    cl_max = prior_cl_L_per_h * 3.0
    vd_min = prior_vd_L * 0.2
    vd_max = prior_vd_L * 3.0

    best_log_post = -1e18
    best_cl = prior_cl_L_per_h
    best_vd = prior_vd_L

    # Store all log-posterior values for posterior CV estimation
    log_posts = []
    cl_vals = []

    for i in range(n_grid):
        # Log-space sampling
        cl = cl_min * math.exp(math.log(cl_max / cl_min) * (i + 0.5) / n_grid)
        for j in range(n_grid):
            vd = vd_min * math.exp(math.log(vd_max / vd_min) * (j + 0.5) / n_grid)

            pred = _predict_conc_1cpt(dose_mg, dose_time_h, cl, vd, sample_time_h, f_oral, ka_per_h)
            lp = _log_posterior(
                cl, vd, measured_conc_mg_L, pred, prior_cl_L_per_h, prior_vd_L, bsv_cl_cv, bsv_vd_cv
            )

            log_posts.append(lp)
            cl_vals.append(cl)

            if lp > best_log_post:
                best_log_post = lp
                best_cl = cl
                best_vd = vd

    map_cl = best_cl
    map_vd = best_vd
    ke_map = map_cl / map_vd
    t_half = 0.693147 / ke_map

    # Estimate posterior CV for CL using weighted variance
    # Use exp(log_post - max) as unnormalized weights
    max_lp = max(log_posts)
    weights = [math.exp(lp - max_lp) for lp in log_posts]
    sum_w = sum(weights)
    if sum_w > 0.0:
        mean_cl = sum(w * c for w, c in zip(weights, cl_vals)) / sum_w
        var_cl = sum(w * (c - mean_cl) ** 2 for w, c in zip(weights, cl_vals)) / sum_w
        posterior_cv_cl = math.sqrt(var_cl) / mean_cl * 100.0 if mean_cl > 0 else 0.0
    else:
        posterior_cv_cl = bsv_cl_cv * 100.0

    # Predict trough and peak at steady state using MAP parameters
    # Trough: concentration just before next dose (at tau_h)
    # For simplicity, use single-dose prediction at tau_h
    predicted_trough = _predict_conc_1cpt(
        dose_mg, dose_time_h, map_cl, map_vd, dose_time_h + tau_h, f_oral, ka_per_h
    )

    # Peak: find max concentration within [0, tau_h]
    n_peak_points = 100
    predicted_peak = 0.0
    for k in range(n_peak_points + 1):
        t_check = dose_time_h + tau_h * k / n_peak_points
        c_check = _predict_conc_1cpt(
            dose_mg, dose_time_h, map_cl, map_vd, t_check, f_oral, ka_per_h
        )
        if c_check > predicted_peak:
            predicted_peak = c_check

    # Back-calculate recommended dose from target trough
    # C_trough = (recommended_dose / original_dose) * predicted_trough_at_tau
    # So recommended_dose = original_dose * (target_trough / predicted_trough)
    if predicted_trough > 1e-12:
        recommended_dose = dose_mg * (target_trough_mg_L / predicted_trough)
    else:
        recommended_dose = dose_mg

    # Clamp to reasonable range
    recommended_dose = max(recommended_dose, 1.0)

    # Check if dose adjustment needed
    # Is current predicted trough within target range?
    dose_adjustment_needed = not (
        target_trough_mg_L * 0.9 <= predicted_trough <= target_peak_mg_L * 1.1
    )

    notes_parts = []
    if predicted_trough < target_trough_mg_L * 0.8:
        notes_parts.append("trough below target — consider dose increase")
    elif predicted_trough > target_peak_mg_L:
        notes_parts.append(
            "trough above target peak — consider dose reduction or interval extension"
        )
    else:
        notes_parts.append("PK parameters within acceptable range")

    notes = "; ".join(notes_parts) if notes_parts else "No action needed"

    return BayesianTDMResult(
        drug_name=drug_name,
        measured_conc_mg_L=measured_conc_mg_L,
        sample_time_h=sample_time_h,
        prior_cl_L_per_h=prior_cl_L_per_h,
        prior_vd_L=prior_vd_L,
        map_cl_L_per_h=map_cl,
        map_vd_L=map_vd,
        map_t_half_h=t_half,
        predicted_trough_mg_L=predicted_trough,
        predicted_peak_mg_L=predicted_peak,
        target_trough_mg_L=target_trough_mg_L,
        target_peak_mg_L=target_peak_mg_L,
        recommended_dose_mg=recommended_dose,
        recommended_interval_h=tau_h,
        dose_adjustment_needed=dose_adjustment_needed,
        posterior_cv_cl_pct=posterior_cv_cl,
        notes=notes,
    )


def quick_tdm_assessment(
    drug_name: str,
    measured_conc_mg_L: float,
    target_min: float,
    target_max: float,
) -> dict:
    """
    Quick TDM assessment without full Bayesian optimization.

    Returns dict with status and adjustment suggestion.
    """
    if measured_conc_mg_L < 0.0:
        raise ValueError(f"measured_conc_mg_L must be non-negative, got {measured_conc_mg_L}")

    if measured_conc_mg_L < target_min:
        status = "subtherapeutic"
        ratio = target_min / measured_conc_mg_L if measured_conc_mg_L > 0 else float("inf")
        suggestion = f"Increase dose by approximately {(ratio - 1.0) * 100:.0f}%"
        action = "increase_dose"
    elif measured_conc_mg_L > target_max:
        status = "supratherapeutic"
        ratio = target_max / measured_conc_mg_L if measured_conc_mg_L > 0 else 0.0
        suggestion = f"Reduce dose by approximately {(1.0 - ratio) * 100:.0f}%"
        action = "reduce_dose"
    else:
        status = "therapeutic"
        suggestion = "No dose adjustment required"
        action = "maintain_dose"

    return {
        "drug_name": drug_name,
        "measured_conc_mg_L": measured_conc_mg_L,
        "target_min_mg_L": target_min,
        "target_max_mg_L": target_max,
        "status": status,
        "action": action,
        "adjustment_suggestion": suggestion,
    }
