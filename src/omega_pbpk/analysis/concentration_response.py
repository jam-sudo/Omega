"""Concentration-response modeling for toxicology and pharmacodynamics.

Fits and predicts concentration-response relationships using standard
Emax and Hill models via scipy.optimize.curve_fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "EmaxFitResult",
    "fit_emax_model",
    "fit_hill_model",
    "predict_response",
    "ic50_from_fit",
    "therapeutic_index",
]


@dataclass(frozen=True)
class EmaxFitResult:
    model: str
    e0: float
    emax: float
    ec50: float
    hill_n: float
    r_squared: float
    rmse: float
    aic: float
    c_range: tuple[float, float]
    concentrations_fit: list[float]
    responses_fit: list[float]
    responses_predicted: list[float]
    slope_at_ec50: float  # n * emax / (4 * ec50) for Hill model
    notes: str


def _emax_function(c: np.ndarray, e0: float, emax: float, ec50: float, n: float) -> np.ndarray:
    """Emax/Hill sigmoid model: E = E0 + Emax * C^n / (EC50^n + C^n)."""
    c_arr = np.asarray(c, dtype=float)
    numerator = emax * np.power(c_arr, n)
    denominator = np.power(ec50, n) + np.power(c_arr, n)
    return e0 + numerator / denominator


def _compute_r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2."""
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _compute_aic(y_obs: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Compute AIC assuming normal errors (AIC = n*ln(RSS/n) + 2k)."""
    n = len(y_obs)
    rss = float(np.sum((y_obs - y_pred) ** 2))
    if rss <= 0:
        rss = 1e-12
    log_likelihood = -n / 2.0 * math.log(rss / n)
    return 2 * n_params - 2 * log_likelihood


def _validate_inputs(concentrations: list[float], responses: list[float]) -> None:
    """Validate concentration-response inputs."""
    if len(concentrations) < 4:
        raise ValueError(f"At least 4 data points required for fitting, got {len(concentrations)}")
    if len(concentrations) != len(responses):
        raise ValueError(
            f"concentrations and responses must have the same length, "
            f"got {len(concentrations)} vs {len(responses)}"
        )
    if any(c <= 0 for c in concentrations):
        raise ValueError("All concentrations must be > 0")


def fit_emax_model(
    concentrations: list[float],
    responses: list[float],
) -> EmaxFitResult:
    """Fit Emax/Hill model to concentration-response data.

    Model: E = E0 + Emax * C^n / (EC50^n + C^n)

    Parameters
    ----------
    concentrations:
        Concentration values (must be > 0, at least 4 points).
    responses:
        Corresponding response values.

    Returns
    -------
    EmaxFitResult
        Fitted model parameters and goodness-of-fit statistics.

    Raises
    ------
    ValueError
        If fewer than 4 data points, mismatched lengths, or non-positive concentrations.
    """
    _validate_inputs(concentrations, responses)

    c_arr = np.array(concentrations, dtype=float)
    r_arr = np.array(responses, dtype=float)

    # Initial guesses
    e0_guess = float(np.min(r_arr))
    emax_guess = float(np.max(r_arr) - np.min(r_arr))
    ec50_guess = float(np.median(c_arr))
    n_guess = 1.0

    try:
        popt, _ = curve_fit(
            _emax_function,
            c_arr,
            r_arr,
            p0=[e0_guess, emax_guess, ec50_guess, n_guess],
            bounds=(
                [-np.inf, 0, 0, 0.1],
                [np.inf, np.inf, np.inf, 10.0],
            ),
            maxfev=10000,
        )
        e0, emax, ec50, hill_n = popt
    except RuntimeError:
        # Fall back to initial guesses if fitting fails
        e0, emax, ec50, hill_n = e0_guess, emax_guess, ec50_guess, n_guess

    y_pred = _emax_function(c_arr, e0, emax, ec50, hill_n)
    r2 = _compute_r_squared(r_arr, y_pred)
    rmse = float(np.sqrt(np.mean((r_arr - y_pred) ** 2)))
    aic = _compute_aic(r_arr, y_pred, n_params=4)

    # slope at EC50: dE/dC at C=EC50 = n * Emax / (4 * EC50)
    slope = hill_n * emax / (4.0 * ec50) if ec50 > 0 else 0.0

    notes = (
        f"Emax model fit: E0={e0:.4g}, Emax={emax:.4g}, EC50={ec50:.4g}, n={hill_n:.4g}. "
        f"R²={r2:.4f}."
    )

    return EmaxFitResult(
        model="emax",
        e0=float(e0),
        emax=float(emax),
        ec50=float(ec50),
        hill_n=float(hill_n),
        r_squared=r2,
        rmse=rmse,
        aic=aic,
        c_range=(float(np.min(c_arr)), float(np.max(c_arr))),
        concentrations_fit=list(c_arr),
        responses_fit=list(r_arr),
        responses_predicted=list(y_pred),
        slope_at_ec50=slope,
        notes=notes,
    )


def fit_hill_model(
    concentrations: list[float],
    responses: list[float],
) -> EmaxFitResult:
    """Fit Hill model to concentration-response data with response normalized to 0-100%.

    Same as Emax model but response is normalized. The EC50 and Hill coefficient
    are computed from the normalized data.

    Parameters
    ----------
    concentrations:
        Concentration values (must be > 0, at least 4 points).
    responses:
        Corresponding response values (will be normalized to 0-100% internally).

    Returns
    -------
    EmaxFitResult
        Fitted model parameters with normalized response scale.

    Raises
    ------
    ValueError
        If fewer than 4 data points, mismatched lengths, or non-positive concentrations.
    """
    _validate_inputs(concentrations, responses)

    c_arr = np.array(concentrations, dtype=float)
    r_arr = np.array(responses, dtype=float)

    # Normalize to 0-100%
    r_min = float(np.min(r_arr))
    r_max = float(np.max(r_arr))
    r_range = r_max - r_min
    if r_range == 0.0:
        r_norm = np.zeros_like(r_arr)
    else:
        r_norm = (r_arr - r_min) / r_range * 100.0

    # Fit on normalized data: E0 ~0, Emax ~100
    ec50_guess = float(np.median(c_arr))
    n_guess = 1.0

    try:
        popt, _ = curve_fit(
            _emax_function,
            c_arr,
            r_norm,
            p0=[0.0, 100.0, ec50_guess, n_guess],
            bounds=(
                [-10.0, 50.0, 0, 0.1],
                [50.0, 200.0, np.inf, 10.0],
            ),
            maxfev=10000,
        )
        e0, emax, ec50, hill_n = popt
    except RuntimeError:
        e0, emax, ec50, hill_n = 0.0, 100.0, ec50_guess, n_guess

    y_pred_norm = _emax_function(c_arr, e0, emax, ec50, hill_n)
    r2 = _compute_r_squared(r_norm, y_pred_norm)
    rmse = float(np.sqrt(np.mean((r_norm - y_pred_norm) ** 2)))
    aic = _compute_aic(r_norm, y_pred_norm, n_params=4)

    slope = hill_n * emax / (4.0 * ec50) if ec50 > 0 else 0.0

    notes = (
        f"Hill model fit (normalized 0-100%): E0={e0:.4g}, Emax={emax:.4g}, "
        f"EC50={ec50:.4g}, n={hill_n:.4g}. R²={r2:.4f}."
    )

    return EmaxFitResult(
        model="hill",
        e0=float(e0),
        emax=float(emax),
        ec50=float(ec50),
        hill_n=float(hill_n),
        r_squared=r2,
        rmse=rmse,
        aic=aic,
        c_range=(float(np.min(c_arr)), float(np.max(c_arr))),
        concentrations_fit=list(c_arr),
        responses_fit=list(r_norm),
        responses_predicted=list(y_pred_norm),
        slope_at_ec50=slope,
        notes=notes,
    )


def predict_response(fit: EmaxFitResult, concentrations: list[float]) -> list[float]:
    """Predict responses for given concentrations using a fitted model.

    Parameters
    ----------
    fit:
        Fitted EmaxFitResult from fit_emax_model or fit_hill_model.
    concentrations:
        Concentration values for prediction (must be > 0).

    Returns
    -------
    list[float]
        Predicted response values.

    Raises
    ------
    ValueError
        If any concentration is non-positive.
    """
    if any(c <= 0 for c in concentrations):
        raise ValueError("All concentrations must be > 0")
    c_arr = np.array(concentrations, dtype=float)
    y_pred = _emax_function(c_arr, fit.e0, fit.emax, fit.ec50, fit.hill_n)
    return [float(v) for v in y_pred]


def ic50_from_fit(fit: EmaxFitResult) -> float:
    """Return EC50/IC50 from a fitted model.

    For inhibition models, EC50 equals IC50 by definition.

    Parameters
    ----------
    fit:
        Fitted EmaxFitResult.

    Returns
    -------
    float
        EC50 (= IC50 for inhibition models).
    """
    return fit.ec50


def therapeutic_index(ec50_efficacy: float, ec50_toxicity: float) -> float:
    """Compute the therapeutic index (TI).

    TI = EC50_toxicity / EC50_efficacy

    A TI > 1 indicates a favorable safety margin (toxicity requires higher
    concentrations than efficacy).

    Parameters
    ----------
    ec50_efficacy:
        EC50 for the desired therapeutic effect.
    ec50_toxicity:
        EC50 for the toxic effect.

    Returns
    -------
    float
        Therapeutic index.

    Raises
    ------
    ValueError
        If either EC50 is non-positive.
    """
    if ec50_efficacy <= 0:
        raise ValueError(f"ec50_efficacy must be > 0, got {ec50_efficacy}")
    if ec50_toxicity <= 0:
        raise ValueError(f"ec50_toxicity must be > 0, got {ec50_toxicity}")
    return ec50_toxicity / ec50_efficacy
