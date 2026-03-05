"""Parameter Calibration Engine — fit PBPK model parameters to observed PK data.

Performs maximum-likelihood parameter estimation by minimizing the sum of
squared log-fold-errors between simulated and observed plasma concentrations.

Objective function (equivalent to log-space RMSE):
    f(CLint) = Σ_i (log10(Csim(ti; CLint) / Cobs(ti)))^2

This objective is scale-invariant (treats 2× over and 0.5× under equally),
appropriate for pharmacokinetic data spanning orders of magnitude.

Calibrated parameters:
    - CLint_hepatic (L/h): primary driver of systemic clearance
    - fup (fraction unbound): affects both distribution and clearance
    - Kp_scale: global multiplicative scale on all tissue Kp values

Each parameter can be independently enabled via the `params_to_fit` argument.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug
from omega_pbpk.validation.pk_validator import ValidationResult, compute_validation_metrics

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of parameter calibration against observed PK data."""

    # Parameters
    original_clint_L_per_h: float
    calibrated_clint_L_per_h: float

    original_fup: float
    calibrated_fup: float

    original_kp_scale: float
    calibrated_kp_scale: float

    # Performance metrics before/after calibration
    pre_calibration: ValidationResult
    post_calibration: ValidationResult

    # Optimization metadata
    n_iterations: int
    converged: bool
    optimizer_message: str

    # Calibrated drug (ready to use)
    calibrated_drug: Drug

    @property
    def aafe_improvement(self) -> float:
        """Fold improvement in AAFE (pre/post). > 1 means calibration helped."""
        return self.pre_calibration.aafe / max(self.post_calibration.aafe, 1e-12)

    def summary(self) -> dict[str, object]:
        """Return flat dict suitable for JSON serialization."""
        return {
            "original_clint_L_per_h": self.original_clint_L_per_h,
            "calibrated_clint_L_per_h": self.calibrated_clint_L_per_h,
            "original_fup": self.original_fup,
            "calibrated_fup": self.calibrated_fup,
            "original_kp_scale": self.original_kp_scale,
            "calibrated_kp_scale": self.calibrated_kp_scale,
            "pre_aafe": self.pre_calibration.aafe,
            "post_aafe": self.post_calibration.aafe,
            "aafe_improvement": self.aafe_improvement,
            "pre_within_2fold_pct": self.pre_calibration.within_2fold_pct,
            "post_within_2fold_pct": self.post_calibration.within_2fold_pct,
            "pre_rmse": self.pre_calibration.rmse_mg_per_L,
            "post_rmse": self.post_calibration.rmse_mg_per_L,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "optimizer_message": self.optimizer_message,
        }


def _apply_params(
    base_drug: Drug,
    log_clint: float,
    log_fup: float,
    log_kp_scale: float,
) -> Drug:
    """Return a Drug copy with updated CLint, fup, and Kp values."""
    drug = copy.copy(base_drug)

    clint_new = float(np.exp(log_clint))
    fup_new = float(np.clip(np.exp(log_fup), 1e-4, 1.0))
    kp_scale = float(np.exp(log_kp_scale))

    object.__setattr__(drug, "clint_hepatic_L_per_h", clint_new)
    object.__setattr__(drug, "fup", fup_new)

    # Scale all Kp values
    if drug.kp and kp_scale != 1.0:
        new_kp = {k: v * kp_scale for k, v in drug.kp.items()}
        object.__setattr__(drug, "kp", new_kp)
    if drug.permeability_limited and kp_scale != 1.0:
        new_pl: dict[str, dict[str, float]] = {}
        for tname, tparams in drug.permeability_limited.items():
            new_pl[tname] = dict(tparams)
            if "kp" in new_pl[tname]:
                new_pl[tname]["kp"] = tparams["kp"] * kp_scale
        object.__setattr__(drug, "permeability_limited", new_pl)

    return drug


def _simulate_at_obs(
    drug: Drug,
    obs_time: np.ndarray,
    dose_mg: float,
    route: str,
    body_weight: float,
    t_end_h: float,
) -> np.ndarray:
    """Run simulation and return concentrations interpolated at obs_time."""
    try:
        model = WholeBodyPBPK(drug, body_weight=body_weight)
        if route == "iv":
            model.setup_iv(dose_mg=dose_mg)
        else:
            model.setup_oral(dose_mg=dose_mg)
        result = model.simulate(t_end_h=t_end_h, dt_h=0.1)
        sim_c = np.interp(obs_time, result.time_h, result.plasma_concentration())
        return np.clip(sim_c, 1e-12, None)
    except Exception:
        return np.full(len(obs_time), 1e-12)


def calibrate_drug(
    drug: Drug,
    observed_times: list[float] | np.ndarray,
    observed_conc: list[float] | np.ndarray,
    dose_mg: float,
    route: str = "oral",
    body_weight: float = 70.0,
    params_to_fit: list[str] | None = None,
    clint_bounds: tuple[float, float] | None = None,
    fup_bounds: tuple[float, float] | None = None,
    kp_scale_bounds: tuple[float, float] | None = None,
    max_iter: int = 200,
) -> CalibrationResult:
    """Calibrate PBPK model parameters against observed plasma PK data.

    Minimizes the sum of squared log10-fold-errors between simulated and
    observed concentrations using scipy.optimize.minimize (L-BFGS-B).

    Args:
        drug: Base Drug object (starting point for optimization).
        observed_times: Observation timepoints (h). All values must be ≥ 0.
        observed_conc: Observed plasma concentrations (mg/L). All must be > 0.
        dose_mg: Administered dose (mg).
        route: "oral" or "iv".
        body_weight: Subject body weight (kg).
        params_to_fit: List of parameter names to optimize. Subset of
            ["clint", "fup", "kp_scale"]. Default: ["clint"].
        clint_bounds: (min, max) CLint (L/h). Default: (0.01×original, 100×original).
        fup_bounds: (min, max) fup. Default: (0.001, 1.0).
        kp_scale_bounds: (min, max) Kp scale factor. Default: (0.1, 10.0).
        max_iter: Maximum optimizer iterations.

    Returns:
        CalibrationResult with original/calibrated parameters and pre/post metrics.
    """
    if params_to_fit is None:
        params_to_fit = ["clint"]

    obs_t = np.asarray(observed_times, dtype=np.float64)
    obs_c = np.asarray(observed_conc, dtype=np.float64)

    if np.any(obs_c <= 0):
        raise ValueError("All observed concentrations must be > 0")

    t_end_h = float(np.max(obs_t)) * 1.2
    t_end_h = max(t_end_h, 8.0)

    # --- initial parameters in log space ---
    clint0 = max(drug.clint_hepatic_L_per_h, 0.01)
    fup0 = float(np.clip(drug.fup, 1e-4, 1.0))
    kp0 = 1.0  # multiplicative scale, start at 1

    log_clint0 = np.log(clint0)
    log_fup0 = np.log(fup0)
    log_kp0 = np.log(kp0)

    # --- bounds ---
    cl_lo, cl_hi = clint_bounds if clint_bounds else (0.01 * clint0, 100.0 * clint0)
    fp_lo, fp_hi = fup_bounds if fup_bounds else (0.001, 1.0)
    kp_lo, kp_hi = kp_scale_bounds if kp_scale_bounds else (0.1, 10.0)

    # Build parameter vector (only include fitted params)
    x0 = []
    bounds = []
    _fit_clint = "clint" in params_to_fit
    _fit_fup = "fup" in params_to_fit
    _fit_kp = "kp_scale" in params_to_fit

    if _fit_clint:
        x0.append(log_clint0)
        bounds.append((np.log(cl_lo), np.log(cl_hi)))
    if _fit_fup:
        x0.append(log_fup0)
        bounds.append((np.log(fp_lo), np.log(fp_hi)))
    if _fit_kp:
        x0.append(log_kp0)
        bounds.append((np.log(kp_lo), np.log(kp_hi)))

    if not x0:
        raise ValueError("params_to_fit must contain at least one of: clint, fup, kp_scale")

    # Precompute pre-calibration metrics
    pre_sim = _simulate_at_obs(drug, obs_t, dose_mg, route, body_weight, t_end_h)
    # Build full simulation for pre metrics
    try:
        _m0 = WholeBodyPBPK(drug, body_weight=body_weight)
        if route == "iv":
            _m0.setup_iv(dose_mg=dose_mg)
        else:
            _m0.setup_oral(dose_mg=dose_mg)
        _r0 = _m0.simulate(t_end_h=t_end_h, dt_h=0.1)
        pre_metrics = compute_validation_metrics(obs_t, obs_c, _r0.time_h, _r0.plasma_concentration())
    except Exception:
        pre_metrics = compute_validation_metrics(obs_t, obs_c, obs_t, pre_sim)

    call_count = [0]

    def objective(x: np.ndarray) -> float:
        call_count[0] += 1
        idx = 0
        lc = x[idx] if _fit_clint else log_clint0
        if _fit_clint:
            idx += 1
        lf = x[idx] if _fit_fup else log_fup0
        if _fit_fup:
            idx += 1
        lk = x[idx] if _fit_kp else log_kp0

        d = _apply_params(drug, lc, lf, lk)
        sim_c = _simulate_at_obs(d, obs_t, dose_mg, route, body_weight, t_end_h)
        log_fe = np.log10(sim_c / obs_c)
        return float(np.sum(log_fe ** 2))

    opt = minimize(
        objective,
        x0=np.array(x0),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
    )

    # Extract calibrated values
    x_opt = opt.x
    idx = 0
    lc_opt = x_opt[idx] if _fit_clint else log_clint0
    if _fit_clint:
        idx += 1
    lf_opt = x_opt[idx] if _fit_fup else log_fup0
    if _fit_fup:
        idx += 1
    lk_opt = x_opt[idx] if _fit_kp else log_kp0

    cal_drug = _apply_params(drug, lc_opt, lf_opt, lk_opt)

    # Post-calibration metrics
    try:
        _mc = WholeBodyPBPK(cal_drug, body_weight=body_weight)
        if route == "iv":
            _mc.setup_iv(dose_mg=dose_mg)
        else:
            _mc.setup_oral(dose_mg=dose_mg)
        _rc = _mc.simulate(t_end_h=t_end_h, dt_h=0.1)
        post_metrics = compute_validation_metrics(obs_t, obs_c, _rc.time_h, _rc.plasma_concentration())
    except Exception:
        post_metrics = pre_metrics

    logger.info(
        "Calibration: CLint %.3f→%.3f L/h, fup %.3f→%.3f, AAFE %.2f→%.2f, iters=%d, converged=%s",
        clint0,
        float(np.exp(lc_opt)),
        fup0,
        float(np.clip(np.exp(lf_opt), 1e-4, 1.0)),
        pre_metrics.aafe,
        post_metrics.aafe,
        call_count[0],
        opt.success,
    )

    return CalibrationResult(
        original_clint_L_per_h=clint0,
        calibrated_clint_L_per_h=float(np.exp(lc_opt)),
        original_fup=fup0,
        calibrated_fup=float(np.clip(np.exp(lf_opt), 1e-4, 1.0)),
        original_kp_scale=1.0,
        calibrated_kp_scale=float(np.exp(lk_opt)),
        pre_calibration=pre_metrics,
        post_calibration=post_metrics,
        n_iterations=call_count[0],
        converged=bool(opt.success),
        optimizer_message=str(opt.message),
        calibrated_drug=cal_drug,
    )


__all__ = ["CalibrationResult", "calibrate_drug"]
