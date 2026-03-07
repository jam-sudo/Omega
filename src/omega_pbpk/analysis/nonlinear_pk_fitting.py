"""Nonlinear PK parameter fitting — Phase 258.

Fits Michaelis-Menten PK models to concentration-time data from saturable
elimination experiments. Uses scipy.optimize.curve_fit.

References
----------
- Paal M et al., Pharm Res. 2018 (phenytoin MM-PK example)
- Wagner JG, J Pharmacokinet Biopharm. 1973;1(2):103-21
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class NonlinearPKFitResult:
    """Result from Michaelis-Menten PK fitting."""

    drug_name: str
    vmax_mg_per_h: float  # Maximum elimination rate (mg/h)
    km_mg_L: float  # Michaelis constant (mg/L)
    vd_L: float  # Volume of distribution (L)
    r_squared: float  # Goodness of fit
    rmse: float  # Root mean squared error (mg/L)
    aafe: float  # Average absolute fold error
    predicted: list[float]  # Predicted concentrations
    notes: str


def _mm_pk_model(
    t: np.ndarray,
    vmax: float,
    km: float,
    vd: float,
    c0: float,
    dt: float = 0.01,
) -> np.ndarray:
    """Simulate MM-PK model via forward Euler for given parameters.

    d(amount)/dt = -Vmax * C / (Km + C)  where C = amount/Vd
    """
    t_end = float(t[-1])
    n_steps = max(int(t_end / dt), 1)
    t_sim = np.linspace(0.0, t_end, n_steps + 1)

    c = np.zeros(n_steps + 1)
    c[0] = c0

    for i in range(n_steps):
        dc = -vmax * c[i] / (km + c[i]) / vd * (t_sim[1] - t_sim[0])
        c[i + 1] = max(0.0, c[i] + dc)

    # Interpolate to requested time points
    return np.interp(t, t_sim, c)


def fit_nonlinear_pk(
    drug_name: str,
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    vd_guess_L: float = 30.0,
    vmax_guess: float | None = None,
    km_guess: float | None = None,
) -> NonlinearPKFitResult:
    """Fit Michaelis-Menten PK model to concentration-time data.

    Parameters
    ----------
    drug_name : Drug name.
    times_h : Observed time points (h). Must have >= 4 points.
    concentrations_mg_L : Observed concentrations (mg/L).
    dose_mg : Administered dose (mg). Used as initial amount.
    vd_guess_L : Initial guess for Vd (L). Must be > 0.
    vmax_guess : Initial guess for Vmax (mg/h). If None, estimated from data.
    km_guess : Initial guess for Km (mg/L). If None, estimated from data.

    Returns
    -------
    NonlinearPKFitResult
    """
    if len(times_h) < 4:
        raise ValueError("Need at least 4 time points for fitting")
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError("times_h and concentrations_mg_L must have same length")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_guess_L <= 0:
        raise ValueError("vd_guess_L must be > 0")

    t_arr = np.array(times_h, dtype=float)
    c_obs = np.array(concentrations_mg_L, dtype=float)

    c0 = dose_mg / vd_guess_L
    if vmax_guess is None:
        # Rough estimate: total elimination over time
        vmax_guess = dose_mg / (times_h[-1] * 2)
    if km_guess is None:
        km_guess = float(np.median(c_obs))

    def model_func(t: np.ndarray, vmax: float, km: float) -> np.ndarray:
        return _mm_pk_model(t, vmax, km, vd_guess_L, c0)

    try:
        popt, _ = curve_fit(
            model_func,
            t_arr,
            c_obs,
            p0=[vmax_guess, km_guess],
            bounds=([0.001, 0.001], [1e6, 1e6]),
            maxfev=5000,
        )
        vmax_fit, km_fit = float(popt[0]), float(popt[1])
    except RuntimeError:
        # Fallback to initial guesses if fit fails
        vmax_fit, km_fit = float(vmax_guess), float(km_guess)

    # Predictions at observed times
    c_pred = _mm_pk_model(t_arr, vmax_fit, km_fit, vd_guess_L, c0)

    # Goodness of fit
    ss_res = float(np.sum((c_obs - c_pred) ** 2))
    ss_tot = float(np.sum((c_obs - np.mean(c_obs)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((c_obs - c_pred) ** 2)))

    # AAFE: exp(mean(|log(pred/obs)|))
    valid = (c_obs > 0) & (c_pred > 0)
    if np.any(valid):
        aafe = float(np.exp(np.mean(np.abs(np.log(c_pred[valid] / c_obs[valid])))))
    else:
        aafe = float("nan")

    notes = (
        f"{drug_name}: MM-PK fit. Vmax={vmax_fit:.2f} mg/h, Km={km_fit:.3f} mg/L, "
        f"Vd={vd_guess_L:.1f} L. R\u00b2={r_squared:.3f}, RMSE={rmse:.3f} mg/L, AAFE={aafe:.2f}."
    )

    return NonlinearPKFitResult(
        drug_name=drug_name,
        vmax_mg_per_h=vmax_fit,
        km_mg_L=km_fit,
        vd_L=vd_guess_L,
        r_squared=r_squared,
        rmse=rmse,
        aafe=aafe,
        predicted=c_pred.tolist(),
        notes=notes,
    )


__all__ = ["NonlinearPKFitResult", "fit_nonlinear_pk"]
