"""Dose-response modeling for efficacy and toxicity endpoints.

Fits dose-response relationships using Hill/Emax sigmoid models.
Supports computation of ED10/ED50/ED90, therapeutic index, and
compound screening by potency.

References
----------
- Hill AV. J Physiol 1910;40:190-224
- FDA Guidance: Estimating the Maximum Safe Starting Dose (2005)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "DoseResponseResult",
    "fit_dose_response",
    "compute_therapeutic_index",
    "screen_dose_response_compounds",
]


@dataclass(frozen=True)
class DoseResponseResult:
    """Results from dose-response model fitting.

    Attributes
    ----------
    drug_name : Drug or compound name.
    endpoint : Type of endpoint ("efficacy", "toxicity", "biomarker").
    model_type : Model fitted ("hill", "sigmoid_emax", "biphasic").
    doses_mg : Input doses (mg).
    responses : Observed response values.
    responses_fitted : Model-fitted response values at input doses.
    emax : Maximum effect (above baseline).
    ec50_mg : Dose producing 50% of Emax.
    hill_n : Hill coefficient (steepness of the curve).
    e0 : Baseline response.
    r_squared : Coefficient of determination.
    rmse : Root mean squared error.
    ed10_mg : Effective dose for 10% of Emax.
    ed50_mg : Effective dose for 50% of Emax (= ec50_mg).
    ed90_mg : Effective dose for 90% of Emax.
    therapeutic_index : ED50_tox / ED50_eff if both available; 0.0 otherwise.
    d_min_effective : Lowest dose where response > 10% Emax.
    d_max_useful : Dose at 90% Emax (= ed90_mg).
    selectivity_window_fold : ED90_eff / ED10_tox if applicable; 0.0 otherwise.
    notes : Fit summary.
    """

    drug_name: str
    endpoint: str
    model_type: str

    doses_mg: list[float]
    responses: list[float]
    responses_fitted: list[float]

    emax: float
    ec50_mg: float
    hill_n: float
    e0: float

    r_squared: float
    rmse: float

    ed10_mg: float
    ed50_mg: float
    ed90_mg: float
    therapeutic_index: float

    d_min_effective: float
    d_max_useful: float
    selectivity_window_fold: float

    notes: str


def _hill_model(d: np.ndarray, e0: float, emax: float, ec50: float, n: float) -> np.ndarray:
    """Hill/Emax sigmoid: E(D) = E0 + Emax * D^n / (EC50^n + D^n)."""
    d_arr = np.asarray(d, dtype=float)
    numerator = emax * np.power(d_arr, n)
    denominator = np.power(ec50, n) + np.power(d_arr, n)
    return e0 + numerator / denominator


def _r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    r2 = 1.0 - ss_res / ss_tot
    # Clamp to [0, 1] — negative R2 from bad fits is reported as 0
    return float(max(0.0, min(1.0, r2)))


def _edx_from_params(emax: float, ec50: float, hill_n: float, fraction: float) -> float:
    """Compute EDx from Hill model parameters.

    EDx = EC50 * (f / (1 - f))^(1/n), where f = fraction of Emax.
    """
    if emax <= 0 or ec50 <= 0 or hill_n <= 0:
        return float("nan")
    if fraction <= 0 or fraction >= 1:
        return float("nan")
    ratio = fraction / (1.0 - fraction)
    return float(ec50 * (ratio ** (1.0 / hill_n)))


def _validate_inputs(doses_mg: list[float], responses: list[float]) -> None:
    """Validate dose-response inputs."""
    if len(doses_mg) != len(responses):
        raise ValueError(
            f"doses_mg and responses must have the same length, "
            f"got {len(doses_mg)} vs {len(responses)}"
        )
    if len(doses_mg) < 4:
        raise ValueError(f"At least 4 data points required for fitting, got {len(doses_mg)}")
    if any(d <= 0 for d in doses_mg):
        raise ValueError("All doses_mg must be > 0")
    if any(not math.isfinite(r) for r in responses):
        raise ValueError("All responses must be finite")


def fit_dose_response(
    drug_name: str,
    doses_mg: list[float],
    responses: list[float],
    endpoint: str = "efficacy",
    model_type: str = "hill",
    e0: float | None = None,
    emax_fixed: float | None = None,
) -> DoseResponseResult:
    """Fit a Hill/Emax dose-response model to dose-response data.

    Model: E(D) = E0 + Emax * D^n / (EC50^n + D^n)

    Parameters
    ----------
    drug_name : Drug or compound name.
    doses_mg : Dose values (mg). Must all be > 0. At least 4 points required.
    responses : Observed response values. Must be finite.
    endpoint : Endpoint type ("efficacy", "toxicity", "biomarker"). Default "efficacy".
    model_type : Model type label ("hill", "sigmoid_emax", "biphasic"). Default "hill".
    e0 : Optional fixed baseline response. If None, E0 is fitted.
    emax_fixed : Optional fixed maximum effect. If None, Emax is fitted.

    Returns
    -------
    DoseResponseResult with fitted parameters and derived metrics.

    Raises
    ------
    ValueError
        If fewer than 4 data points, mismatched lengths, or invalid doses/responses.
    """
    _validate_inputs(doses_mg, responses)

    d_arr = np.array(doses_mg, dtype=float)
    r_arr = np.array(responses, dtype=float)

    # Initial guesses
    e0_guess = float(np.min(r_arr)) if e0 is None else float(e0)
    emax_guess = float(np.max(r_arr) - np.min(r_arr)) if emax_fixed is None else float(emax_fixed)
    if emax_guess <= 0:
        emax_guess = max(1.0, float(np.ptp(r_arr)))
    ec50_guess = float(np.median(d_arr))
    n_guess = 1.0

    # Determine which parameters are free
    fix_e0 = e0 is not None
    fix_emax = emax_fixed is not None

    fitted_e0 = e0_guess
    fitted_emax = emax_guess
    fitted_ec50 = ec50_guess
    fitted_n = n_guess

    fit_success = False

    if not fix_e0 and not fix_emax:
        # Fit all 4 parameters
        try:
            popt, _ = curve_fit(
                _hill_model,
                d_arr,
                r_arr,
                p0=[e0_guess, emax_guess, ec50_guess, n_guess],
                bounds=(
                    [-np.inf, 1e-9, 1e-9, 0.3],
                    [np.inf, np.inf, np.inf, 5.0],
                ),
                maxfev=10000,
            )
            fitted_e0, fitted_emax, fitted_ec50, fitted_n = popt
            fit_success = True
        except (RuntimeError, ValueError):
            fit_success = False

    elif fix_e0 and not fix_emax:
        # Fix E0, fit Emax/EC50/n
        fixed_e0_val = float(e0)

        def _model_fixed_e0(d: np.ndarray, emax: float, ec50: float, n: float) -> np.ndarray:
            return fixed_e0_val + emax * np.power(d, n) / (np.power(ec50, n) + np.power(d, n))

        try:
            popt, _ = curve_fit(
                _model_fixed_e0,
                d_arr,
                r_arr,
                p0=[emax_guess, ec50_guess, n_guess],
                bounds=([1e-9, 1e-9, 0.3], [np.inf, np.inf, 5.0]),
                maxfev=10000,
            )
            fitted_e0 = fixed_e0_val
            fitted_emax, fitted_ec50, fitted_n = popt
            fit_success = True
        except (RuntimeError, ValueError):
            fitted_e0 = fixed_e0_val
            fit_success = False

    elif not fix_e0 and fix_emax:
        # Fix Emax, fit E0/EC50/n
        fixed_emax_val = float(emax_fixed)

        def _model_fixed_emax(d: np.ndarray, e0_p: float, ec50: float, n: float) -> np.ndarray:
            return e0_p + fixed_emax_val * np.power(d, n) / (np.power(ec50, n) + np.power(d, n))

        try:
            popt, _ = curve_fit(
                _model_fixed_emax,
                d_arr,
                r_arr,
                p0=[e0_guess, ec50_guess, n_guess],
                bounds=([-np.inf, 1e-9, 0.3], [np.inf, np.inf, 5.0]),
                maxfev=10000,
            )
            fitted_e0, fitted_ec50, fitted_n = popt
            fitted_emax = fixed_emax_val
            fit_success = True
        except (RuntimeError, ValueError):
            fitted_emax = fixed_emax_val
            fit_success = False

    else:
        # Both fixed: only fit EC50 and n
        fixed_e0_val = float(e0)
        fixed_emax_val = float(emax_fixed)

        def _model_both_fixed(d: np.ndarray, ec50: float, n: float) -> np.ndarray:
            return fixed_e0_val + fixed_emax_val * np.power(d, n) / (
                np.power(ec50, n) + np.power(d, n)
            )

        try:
            popt, _ = curve_fit(
                _model_both_fixed,
                d_arr,
                r_arr,
                p0=[ec50_guess, n_guess],
                bounds=([1e-9, 0.3], [np.inf, 5.0]),
                maxfev=10000,
            )
            fitted_e0 = fixed_e0_val
            fitted_emax = fixed_emax_val
            fitted_ec50, fitted_n = popt
            fit_success = True
        except (RuntimeError, ValueError):
            fitted_e0 = fixed_e0_val
            fitted_emax = fixed_emax_val
            fit_success = False

    # Ensure positive emax and ec50
    fitted_emax = max(1e-9, float(fitted_emax))
    fitted_ec50 = max(1e-9, float(fitted_ec50))
    fitted_n = float(np.clip(fitted_n, 0.3, 5.0))

    # Compute fitted values and statistics
    y_pred = _hill_model(d_arr, fitted_e0, fitted_emax, fitted_ec50, fitted_n)
    r2 = _r_squared(r_arr, y_pred)
    rmse = float(np.sqrt(np.mean((r_arr - y_pred) ** 2)))

    # Derived dose metrics
    ed10 = _edx_from_params(fitted_emax, fitted_ec50, fitted_n, 0.10)
    ed50 = fitted_ec50  # by definition
    ed90 = _edx_from_params(fitted_emax, fitted_ec50, fitted_n, 0.90)

    # d_min_effective: lowest dose where fitted response > E0 + 0.1 * Emax
    threshold = float(fitted_e0) + 0.1 * fitted_emax
    effective_mask = y_pred > threshold
    if np.any(effective_mask):
        d_min_effective = float(d_arr[effective_mask][0])
    else:
        d_min_effective = float(np.min(d_arr))

    d_max_useful = ed90 if math.isfinite(ed90) else float(np.max(d_arr))

    fit_note = "successful" if fit_success else "failed (using approximate parameters)"
    notes = (
        f"{drug_name} {endpoint} dose-response ({model_type}): "
        f"E0={fitted_e0:.4g}, Emax={fitted_emax:.4g}, EC50={fitted_ec50:.4g} mg, "
        f"n={fitted_n:.4g}. R²={r2:.4f}. Fit: {fit_note}."
    )

    return DoseResponseResult(
        drug_name=drug_name,
        endpoint=endpoint,
        model_type=model_type,
        doses_mg=list(doses_mg),
        responses=list(responses),
        responses_fitted=[float(v) for v in y_pred],
        emax=float(fitted_emax),
        ec50_mg=float(fitted_ec50),
        hill_n=float(fitted_n),
        e0=float(fitted_e0),
        r_squared=r2,
        rmse=rmse,
        ed10_mg=ed10,
        ed50_mg=ed50,
        ed90_mg=ed90,
        therapeutic_index=0.0,
        d_min_effective=d_min_effective,
        d_max_useful=d_max_useful,
        selectivity_window_fold=0.0,
        notes=notes,
    )


def compute_therapeutic_index(
    efficacy_result: DoseResponseResult,
    toxicity_result: DoseResponseResult,
) -> dict:
    """Compute therapeutic index from efficacy and toxicity dose-response results.

    Therapeutic index (TI) = ED50_tox / ED50_eff
    Selectivity window = ED10_tox / ED90_eff

    Parameters
    ----------
    efficacy_result : Fitted efficacy DoseResponseResult.
    toxicity_result : Fitted toxicity DoseResponseResult.

    Returns
    -------
    dict with keys:
        - "therapeutic_index": float
        - "selectivity_window": float
        - "risk_assessment": "favorable", "marginal", or "unfavorable"
    """
    ed50_eff = efficacy_result.ec50_mg
    ed50_tox = toxicity_result.ec50_mg

    ti = ed50_tox / ed50_eff if ed50_eff > 0 else float("inf")

    ed10_tox = toxicity_result.ed10_mg
    ed90_eff = efficacy_result.ed90_mg

    if math.isfinite(ed10_tox) and math.isfinite(ed90_eff) and ed90_eff > 0:
        selectivity_window = ed10_tox / ed90_eff
    else:
        selectivity_window = 0.0

    if ti >= 10.0:
        risk_assessment = "favorable"
    elif ti >= 2.0:
        risk_assessment = "marginal"
    else:
        risk_assessment = "unfavorable"

    return {
        "therapeutic_index": float(ti),
        "selectivity_window": float(selectivity_window),
        "risk_assessment": risk_assessment,
    }


def screen_dose_response_compounds(
    compounds_data: list[dict],
) -> list[DoseResponseResult]:
    """Screen a list of compounds by dose-response potency.

    Each compound dict must have keys: "name", "doses_mg", "responses".
    Optional keys: "endpoint", "model_type".

    Returns results sorted by ec50_mg ascending (most potent first).

    Parameters
    ----------
    compounds_data : List of compound dicts with dose-response data.

    Returns
    -------
    list[DoseResponseResult]
        Sorted by EC50 ascending (most potent first).

    Raises
    ------
    ValueError
        If any compound data is missing required keys.
    """
    results: list[DoseResponseResult] = []
    for compound in compounds_data:
        if "name" not in compound:
            raise ValueError("Each compound dict must have 'name' key")
        if "doses_mg" not in compound:
            raise ValueError("Each compound dict must have 'doses_mg' key")
        if "responses" not in compound:
            raise ValueError("Each compound dict must have 'responses' key")

        endpoint = compound.get("endpoint", "efficacy")
        model_type = compound.get("model_type", "hill")

        result = fit_dose_response(
            drug_name=compound["name"],
            doses_mg=compound["doses_mg"],
            responses=compound["responses"],
            endpoint=endpoint,
            model_type=model_type,
        )
        results.append(result)

    # Sort by EC50 ascending (lowest EC50 = most potent = first)
    results.sort(key=lambda r: r.ec50_mg)
    return results
