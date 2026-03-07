"""Phase 320 — Concentration-Time Profile Fitter.

Fit observed concentration-time data to a 1-compartment oral or IV PK model
using grid search. Estimates CL, Vd (and ka for oral) from observed data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_ROUTES = {"oral", "iv_bolus"}


@dataclass(frozen=True)
class PKFitResult:
    """Immutable result from a 1-compartment PK model fit."""

    drug_name: str
    dose_mg: float
    route: str
    fitted_cl_L_per_h: float
    fitted_vd_L: float
    fitted_ka_per_h: float | None
    fitted_ke_per_h: float
    fitted_t_half_h: float
    r_squared: float
    ssr: float
    auc_obs_mg_h_per_L: float
    auc_pred_mg_h_per_L: float
    n_observations: int
    notes: str


def _validate(
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    route: str,
) -> None:
    if len(times_h) < 3:
        raise ValueError(f"Need at least 3 observations, got {len(times_h)}")
    if len(concentrations_mg_L) != len(times_h):
        raise ValueError(
            f"len(concentrations_mg_L)={len(concentrations_mg_L)} != len(times_h)={len(times_h)}"
        )
    if any(t < 0 for t in times_h):
        raise ValueError("All times must be >= 0")
    if any(c < 0 for c in concentrations_mg_L):
        raise ValueError("All concentrations must be >= 0")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")


def _logspace(start: float, stop: float, n: int) -> list[float]:
    """Generate n log-spaced values from start to stop inclusive."""
    if n == 1:
        return [start]
    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / (n - 1)
    return [math.exp(log_start + i * step) for i in range(n)]


def _predict_iv(times_h: list[float], dose_mg: float, cl: float, vd: float) -> list[float]:
    """Analytical 1-cpt IV bolus: C(t) = (dose/Vd) * exp(-ke*t)."""
    ke = cl / vd
    c0 = dose_mg / vd
    return [c0 * math.exp(-ke * t) for t in times_h]


def _predict_oral(
    times_h: list[float], dose_mg: float, cl: float, vd: float, ka: float
) -> list[float]:
    """Analytical 1-cpt oral: C(t) = dose*ka/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))."""
    ke = cl / vd
    # avoid singularity when ka ~ ke
    if abs(ka - ke) < 1e-6:
        ka = ke * 1.001
    denom = vd * (ka - ke)
    scale = dose_mg * ka / denom
    return [max(0.0, scale * (math.exp(-ke * t) - math.exp(-ka * t))) for t in times_h]


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _ssr(obs: list[float], pred: list[float]) -> float:
    return sum((o - p) ** 2 for o, p in zip(obs, pred))


def fit_one_compartment(
    drug_name: str,
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    route: str,
) -> PKFitResult:
    """Fit a 1-compartment PK model to observed concentration-time data.

    Uses grid search to minimise sum of squared residuals (SSR).

    Parameters
    ----------
    drug_name:
        Name / identifier of the drug.
    times_h:
        Observed time points in hours (>= 0, at least 3 points).
    concentrations_mg_L:
        Observed plasma concentrations in mg/L (same length as times_h, >= 0).
    dose_mg:
        Administered dose in mg (> 0).
    route:
        "oral" or "iv_bolus".

    Returns
    -------
    PKFitResult
    """
    _validate(times_h, concentrations_mg_L, dose_mg, route)

    obs = concentrations_mg_L
    t = times_h
    n = len(obs)

    mean_obs = sum(obs) / n
    ss_tot = sum((o - mean_obs) ** 2 for o in obs)

    # Grid definitions
    cl_grid = _logspace(0.1, 100.0, 20)
    vd_grid = _logspace(1.0, 1000.0, 20)
    ka_grid = _logspace(0.1, 10.0, 10) if route == "oral" else [None]

    best_ssr = float("inf")
    best_cl = cl_grid[0]
    best_vd = vd_grid[0]
    best_ka: float | None = None
    best_pred: list[float] = []

    for cl in cl_grid:
        for vd in vd_grid:
            for ka in ka_grid:
                if route == "iv_bolus":
                    pred = _predict_iv(t, dose_mg, cl, vd)
                else:
                    pred = _predict_oral(t, dose_mg, cl, vd, ka)  # type: ignore[arg-type]
                s = _ssr(obs, pred)
                if s < best_ssr:
                    best_ssr = s
                    best_cl = cl
                    best_vd = vd
                    best_ka = ka
                    best_pred = pred

    ke = best_cl / best_vd
    t_half = 0.693 / max(ke, 1e-10)
    r2 = 1.0 - best_ssr / max(ss_tot, 1e-10)

    auc_obs = _trapezoidal_auc(t, obs)
    auc_pred = _trapezoidal_auc(t, best_pred)

    notes = f"Grid search: CL={best_cl:.3f} L/h, Vd={best_vd:.1f} L"
    if route == "oral" and best_ka is not None:
        notes += f", ka={best_ka:.3f}/h"

    return PKFitResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        fitted_cl_L_per_h=best_cl,
        fitted_vd_L=best_vd,
        fitted_ka_per_h=best_ka,
        fitted_ke_per_h=ke,
        fitted_t_half_h=t_half,
        r_squared=r2,
        ssr=best_ssr,
        auc_obs_mg_h_per_L=auc_obs,
        auc_pred_mg_h_per_L=auc_pred,
        n_observations=n,
        notes=notes,
    )


def predict_from_fitted(result: PKFitResult, times_h: list[float]) -> list[float]:
    """Generate predicted concentrations from a fitted PKFitResult.

    Parameters
    ----------
    result:
        A PKFitResult from :func:`fit_one_compartment`.
    times_h:
        Time points at which to predict concentrations (in hours).

    Returns
    -------
    list of predicted concentrations in mg/L.
    """
    if result.route == "iv_bolus":
        return _predict_iv(times_h, result.dose_mg, result.fitted_cl_L_per_h, result.fitted_vd_L)
    else:
        ka = result.fitted_ka_per_h
        if ka is None:
            ka = 1.0  # fallback; should not happen for oral
        return _predict_oral(
            times_h, result.dose_mg, result.fitted_cl_L_per_h, result.fitted_vd_L, ka
        )
