"""PK concentration-time curve fitting using compartmental models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "OneCptFitResult",
    "TwoCptFitResult",
    "fit_one_compartment_iv",
    "fit_one_compartment_oral",
    "fit_two_compartment_iv",
]


def _validate_data(times: list[float], concentrations: list[float]) -> None:
    """Validate input data for curve fitting."""
    if len(times) < 3:
        raise ValueError("At least 3 data points are required for fitting.")
    if len(times) != len(concentrations):
        raise ValueError("times and concentrations must have the same length.")
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            raise ValueError("times must be strictly increasing.")
    if any(c <= 0 for c in concentrations):
        raise ValueError("All concentrations must be positive (> 0).")


def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, 0.0, 1.0))


def _rmse(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_obs - y_fit) ** 2)))


def _aic(y_obs: np.ndarray, y_fit: np.ndarray, n_params: int) -> float:
    """AIC = n * log(SSR/n) + 2*k."""
    n = len(y_obs)
    ssr = float(np.sum((y_obs - y_fit) ** 2))
    if ssr <= 0.0:
        ssr = 1e-12
    return n * math.log(ssr / n) + 2 * n_params


@dataclass(frozen=True)
class OneCptFitResult:
    """Result from one-compartment PK model fitting."""

    model: str
    c0_or_cmax: float
    ke_per_h: float
    ka_per_h: float | None
    t_half_h: float
    r_squared: float
    rmse: float
    aic: float
    n_params: int
    times_fit: list[float]
    c_fitted: list[float]
    residuals: list[float]


@dataclass(frozen=True)
class TwoCptFitResult:
    """Result from two-compartment PK model fitting."""

    model: str
    A: float
    alpha: float
    B: float
    beta: float
    t_half_alpha: float
    t_half_beta: float
    r_squared: float
    rmse: float
    aic: float
    n_params: int
    times_fit: list[float]
    c_fitted: list[float]
    residuals: list[float]


def fit_one_compartment_iv(
    times: list[float],
    concentrations: list[float],
    dose_mg: float | None = None,
) -> OneCptFitResult:
    """Fit one-compartment IV model: C(t) = C0 * exp(-ke * t).

    Parameters
    ----------
    times:
        Observation times in hours.
    concentrations:
        Observed plasma concentrations (same units as dose_mg/volume).
    dose_mg:
        Optional dose in mg used to derive CL and Vd.

    Returns
    -------
    OneCptFitResult
    """
    _validate_data(times, concentrations)
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    # Initial guesses
    c0_guess = float(np.max(c))
    ke_guess = max(
        1e-4,
        -math.log(max(c[-1], 1e-12) / max(c[0], 1e-12)) / max(t[-1] - t[0], 1e-6),
    )

    def model(t_arr: np.ndarray, c0: float, ke: float) -> np.ndarray:
        return c0 * np.exp(-ke * t_arr)

    popt, _ = curve_fit(
        model,
        t,
        c,
        p0=[c0_guess, ke_guess],
        bounds=([0.0, 1e-6], [np.inf, np.inf]),
        maxfev=10000,
    )
    c0_fit, ke_fit = float(popt[0]), float(popt[1])
    c_fit = model(t, c0_fit, ke_fit)
    residuals = (c - c_fit).tolist()

    t_half = math.log(2.0) / ke_fit

    return OneCptFitResult(
        model="one_compartment_iv",
        c0_or_cmax=c0_fit,
        ke_per_h=ke_fit,
        ka_per_h=None,
        t_half_h=t_half,
        r_squared=_r_squared(c, c_fit),
        rmse=_rmse(c, c_fit),
        aic=_aic(c, c_fit, n_params=2),
        n_params=2,
        times_fit=t.tolist(),
        c_fitted=c_fit.tolist(),
        residuals=residuals,
    )


def fit_one_compartment_oral(
    times: list[float],
    concentrations: list[float],
    dose_mg: float,
) -> OneCptFitResult:
    """Fit one-compartment oral model (F=1 assumed).

    Model: C(t) = (dose/Vd) * ka/(ka-ke) * (exp(-ke*t) - exp(-ka*t))

    Parameters
    ----------
    times:
        Observation times in hours.
    concentrations:
        Observed plasma concentrations.
    dose_mg:
        Dose in mg.

    Returns
    -------
    OneCptFitResult
    """
    _validate_data(times, concentrations)
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    # Initial guesses
    cmax = float(np.max(c))
    tmax_idx = int(np.argmax(c))
    tmax = float(t[tmax_idx])
    ke_guess = max(1e-4, math.log(2.0) / max(tmax * 2, 1.0))
    ka_guess = max(ke_guess * 2, 1.0 / max(tmax, 0.1))
    vd_guess = dose_mg / max(cmax, 1e-12)

    def model(t_arr: np.ndarray, ka: float, ke: float, vd: float) -> np.ndarray:
        denom = ka - ke
        if abs(denom) < 1e-9:
            denom = 1e-9
        return (dose_mg / vd) * (ka / denom) * (np.exp(-ke * t_arr) - np.exp(-ka * t_arr))

    try:
        popt, _ = curve_fit(
            model,
            t,
            c,
            p0=[ka_guess, ke_guess, vd_guess],
            bounds=([1e-6, 1e-6, 1e-3], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        ka_fit, ke_fit, vd_fit = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        # Fallback: use guesses
        ka_fit, ke_fit, vd_fit = ka_guess, ke_guess, vd_guess

    c_fit = model(t, ka_fit, ke_fit, vd_fit)
    # Clip negative predictions
    c_fit = np.maximum(c_fit, 0.0)
    residuals = (c - c_fit).tolist()
    cmax_fit = float(np.max(c_fit)) if len(c_fit) > 0 else 0.0
    t_half = math.log(2.0) / ke_fit

    return OneCptFitResult(
        model="one_compartment_oral",
        c0_or_cmax=cmax_fit,
        ke_per_h=ke_fit,
        ka_per_h=ka_fit,
        t_half_h=t_half,
        r_squared=_r_squared(c, c_fit),
        rmse=_rmse(c, c_fit),
        aic=_aic(c, c_fit, n_params=3),
        n_params=3,
        times_fit=t.tolist(),
        c_fitted=c_fit.tolist(),
        residuals=residuals,
    )


def fit_two_compartment_iv(
    times: list[float],
    concentrations: list[float],
) -> TwoCptFitResult:
    """Fit two-compartment IV model: C(t) = A*exp(-alpha*t) + B*exp(-beta*t).

    Parameters
    ----------
    times:
        Observation times in hours.
    concentrations:
        Observed plasma concentrations.

    Returns
    -------
    TwoCptFitResult
    """
    _validate_data(times, concentrations)
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    # Initial guesses using log-linear regression on early/late segments
    n = len(t)
    mid = max(1, n // 3)
    c0_guess = float(np.max(c))

    # Late phase: fit beta from last third
    late_t = t[2 * mid :]
    late_c = c[2 * mid :]
    if len(late_t) >= 2 and np.all(late_c > 0):
        beta_guess = max(
            1e-4,
            (math.log(float(late_c[0])) - math.log(float(late_c[-1])))
            / max(float(late_t[-1] - late_t[0]), 1e-6),
        )
    else:
        beta_guess = 0.1

    alpha_guess = max(beta_guess * 5, 1.0)
    B_guess = c0_guess * 0.3
    A_guess = c0_guess * 0.7

    def model(t_arr: np.ndarray, A: float, alpha: float, B: float, beta: float) -> np.ndarray:
        return A * np.exp(-alpha * t_arr) + B * np.exp(-beta * t_arr)

    try:
        popt, _ = curve_fit(
            model,
            t,
            c,
            p0=[A_guess, alpha_guess, B_guess, beta_guess],
            bounds=([0.0, 1e-6, 0.0, 1e-6], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        A_fit, alpha_fit, B_fit, beta_fit = (
            float(popt[0]),
            float(popt[1]),
            float(popt[2]),
            float(popt[3]),
        )
        # Ensure alpha >= beta (distribution phase faster)
        if alpha_fit < beta_fit:
            A_fit, alpha_fit, B_fit, beta_fit = B_fit, beta_fit, A_fit, alpha_fit
    except RuntimeError:
        A_fit, alpha_fit, B_fit, beta_fit = A_guess, alpha_guess, B_guess, beta_guess

    c_fit = model(t, A_fit, alpha_fit, B_fit, beta_fit)
    residuals = (c - c_fit).tolist()

    t_half_alpha = math.log(2.0) / alpha_fit
    t_half_beta = math.log(2.0) / beta_fit

    return TwoCptFitResult(
        model="two_compartment_iv",
        A=A_fit,
        alpha=alpha_fit,
        B=B_fit,
        beta=beta_fit,
        t_half_alpha=t_half_alpha,
        t_half_beta=t_half_beta,
        r_squared=_r_squared(c, c_fit),
        rmse=_rmse(c, c_fit),
        aic=_aic(c, c_fit, n_params=4),
        n_params=4,
        times_fit=t.tolist(),
        c_fitted=c_fit.tolist(),
        residuals=residuals,
    )
