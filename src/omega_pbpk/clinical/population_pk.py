"""Population PK fitting — Standard Two-Stage method."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SubjectData:
    subject_id: str
    times: list[float]
    conc: list[float]
    dose_mg: float
    route: str  # "oral" | "iv"
    body_weight: float = 70.0
    covariates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IndividualEstimate:
    subject_id: str
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float  # math.nan for IV
    auc_pred: float
    cmax_pred: float
    rmse: float
    converged: bool


@dataclass(frozen=True)
class PopulationPKResult:
    cl_pop_L_per_h: float
    vd_pop_L: float
    ka_pop_per_h: float  # math.nan for IV
    omega_cl_pct: float
    omega_vd_pct: float
    omega_ka_pct: float  # math.nan for IV
    sigma_prop_pct: float
    individual_estimates: list[IndividualEstimate]
    n_subjects: int
    n_observations: int
    aic: float
    bic: float
    bw_cl_exponent: float | None
    method: str


def _cp_iv(t: np.ndarray, dose: float, cl: float, vd: float) -> np.ndarray:
    ke = cl / vd
    return (dose / vd) * np.exp(-ke * t)


def _cp_oral(t: np.ndarray, dose: float, cl: float, vd: float, ka: float) -> np.ndarray:
    ke = cl / vd
    cp = np.zeros_like(t, dtype=float)
    if abs(ka - ke) < 1e-3 * ke:
        # Limit form when ka ≈ ke
        cp = (dose / vd) * ke * t * np.exp(-ke * t)
    else:
        cp = dose * ka / (vd * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.maximum(cp, 0.0)


def _fit_subject(subj: SubjectData, is_oral: bool) -> IndividualEstimate:
    times = np.asarray(subj.times, dtype=float)
    conc = np.asarray(subj.conc, dtype=float)

    if len(times) < 3:
        return IndividualEstimate(
            subject_id=subj.subject_id,
            cl_L_per_h=math.nan,
            vd_L=math.nan,
            ka_per_h=math.nan,
            auc_pred=math.nan,
            cmax_pred=math.nan,
            rmse=math.nan,
            converged=False,
        )

    dose = subj.dose_mg

    def objective(params: np.ndarray) -> float:
        cl, vd = params[0], params[1]
        if is_oral:
            ka = params[2]
            pred = _cp_oral(times, dose, cl, vd, ka)
        else:
            pred = _cp_iv(times, dose, cl, vd)
        return float(np.sum((np.log(conc + 1e-6) - np.log(pred + 1e-6)) ** 2))

    if is_oral:
        x0 = np.array([5.0, 50.0, 1.0])
        bounds = [(0.01, 500.0), (0.1, 2000.0), (0.01, 50.0)]
    else:
        x0 = np.array([5.0, 50.0])
        bounds = [(0.01, 500.0), (0.1, 2000.0)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    if not result.success:
        return IndividualEstimate(
            subject_id=subj.subject_id,
            cl_L_per_h=math.nan,
            vd_L=math.nan,
            ka_per_h=math.nan,
            auc_pred=math.nan,
            cmax_pred=math.nan,
            rmse=math.nan,
            converged=False,
        )

    cl_fit = float(result.x[0])
    vd_fit = float(result.x[1])
    ka_fit = float(result.x[2]) if is_oral else math.nan

    # AUC0-inf
    auc_pred = dose / cl_fit

    # Cmax from dense evaluation
    t_dense = np.linspace(0.0, float(np.max(times)), 200)
    if is_oral:
        cp_dense = _cp_oral(t_dense, dose, cl_fit, vd_fit, ka_fit)
    else:
        cp_dense = _cp_iv(t_dense, dose, cl_fit, vd_fit)
    cmax_pred = float(np.max(cp_dense))

    # RMSE in original space
    if is_oral:
        pred_at_obs = _cp_oral(times, dose, cl_fit, vd_fit, ka_fit)
    else:
        pred_at_obs = _cp_iv(times, dose, cl_fit, vd_fit)
    rmse = float(np.sqrt(np.mean((conc - pred_at_obs) ** 2)))

    return IndividualEstimate(
        subject_id=subj.subject_id,
        cl_L_per_h=cl_fit,
        vd_L=vd_fit,
        ka_per_h=ka_fit,
        auc_pred=auc_pred,
        cmax_pred=cmax_pred,
        rmse=rmse,
        converged=True,
    )


def fit_population_pk(
    subjects: list[SubjectData],
    model: str = "1cpt_oral",
    cl_init: float = 5.0,
    vd_init: float = 50.0,
    ka_init: float = 1.0,
    fit_covariates: bool = False,
) -> PopulationPKResult:
    """Fit population PK using Standard Two-Stage method."""
    is_oral = model == "1cpt_oral"

    # Stage 1: fit each subject individually
    estimates = [_fit_subject(subj, is_oral) for subj in subjects]
    converged = [e for e in estimates if e.converged]

    if not converged:
        raise ValueError("No subjects converged")

    # Stage 2: population summary (geometric means and variability)
    log_cl = np.array([math.log(e.cl_L_per_h) for e in converged])
    log_vd = np.array([math.log(e.vd_L) for e in converged])

    cl_pop = math.exp(float(np.mean(log_cl)))
    vd_pop = math.exp(float(np.mean(log_vd)))

    omega_cl_pct = math.sqrt(max(0.0, math.exp(float(np.var(log_cl, ddof=1))) - 1.0)) * 100.0 if len(converged) > 1 else 0.0
    omega_vd_pct = math.sqrt(max(0.0, math.exp(float(np.var(log_vd, ddof=1))) - 1.0)) * 100.0 if len(converged) > 1 else 0.0

    if is_oral:
        log_ka = np.array([math.log(e.ka_per_h) for e in converged])
        ka_pop = math.exp(float(np.mean(log_ka)))
        omega_ka_pct = math.sqrt(max(0.0, math.exp(float(np.var(log_ka, ddof=1))) - 1.0)) * 100.0 if len(converged) > 1 else 0.0
    else:
        ka_pop = math.nan
        omega_ka_pct = math.nan

    # Compute pooled RSS and sigma in original space
    n_obs = 0
    rss = 0.0
    all_obs: list[float] = []
    for subj, est in zip(subjects, estimates):
        if not est.converged:
            continue
        times = np.asarray(subj.times, dtype=float)
        conc = np.asarray(subj.conc, dtype=float)
        if is_oral:
            pred = _cp_oral(times, subj.dose_mg, est.cl_L_per_h, est.vd_L, est.ka_per_h)
        else:
            pred = _cp_iv(times, subj.dose_mg, est.cl_L_per_h, est.vd_L)
        rss += float(np.sum((conc - pred) ** 2))
        n_obs += len(times)
        all_obs.extend(conc.tolist())

    sigma2 = rss / n_obs if n_obs > 0 else 0.0
    mean_obs = float(np.mean(all_obs)) if all_obs else 1.0
    sigma_prop_pct = math.sqrt(sigma2) / mean_obs * 100.0 if mean_obs > 0 else 0.0

    # AIC / BIC
    n_params = 3 if is_oral else 2
    if fit_covariates:
        n_params += 1

    if n_obs > 0 and sigma2 > 0:
        log_lik = -n_obs / 2.0 * (math.log(2.0 * math.pi * sigma2) + 1.0)
    else:
        log_lik = 0.0
    aic = -2.0 * log_lik + 2.0 * n_params
    bic = -2.0 * log_lik + n_params * math.log(n_obs) if n_obs > 0 else aic

    # Covariate analysis
    bw_cl_exponent: float | None = None
    if fit_covariates and len(converged) >= 2:
        subj_map = {s.subject_id: s for s in subjects}
        bw_arr = np.array([subj_map[e.subject_id].body_weight for e in converged])
        cl_arr = np.array([e.cl_L_per_h for e in converged])
        coeffs = np.polyfit(np.log(bw_arr / 70.0), np.log(cl_arr), 1)
        bw_cl_exponent = float(coeffs[0])

    return PopulationPKResult(
        cl_pop_L_per_h=cl_pop,
        vd_pop_L=vd_pop,
        ka_pop_per_h=ka_pop,
        omega_cl_pct=omega_cl_pct,
        omega_vd_pct=omega_vd_pct,
        omega_ka_pct=omega_ka_pct,
        sigma_prop_pct=sigma_prop_pct,
        individual_estimates=estimates,
        n_subjects=len(subjects),
        n_observations=n_obs,
        aic=aic,
        bic=bic,
        bw_cl_exponent=bw_cl_exponent,
        method="standard_two_stage",
    )


__all__ = ["SubjectData", "IndividualEstimate", "PopulationPKResult", "fit_population_pk"]
