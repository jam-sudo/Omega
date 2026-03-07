"""Dose-response surface analysis with Hill equation fitting and combination modeling.

Supports single-drug Hill curve fitting and combination dose-response surface
analysis using Bliss independence model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "HillFitResult",
    "DoseResponseSurfaceResult",
    "fit_hill_curve",
    "dose_response_surface",
    "optimal_combination",
]

_MIN_DATA_POINTS = 4


@dataclass(frozen=True)
class HillFitResult:
    """Result of fitting a Hill (sigmoidal) dose-response curve."""

    emax: float
    ed50: float
    hill_n: float
    r2: float
    rmse: float
    fitted_responses: list[float]
    notes: str


@dataclass(frozen=True)
class DoseResponseSurfaceResult:
    """Result of dose-response surface analysis for a two-drug combination."""

    drug1_fit: HillFitResult
    drug2_fit: HillFitResult
    interaction_matrix: list[list[float]]  # observed / bliss (II); >1 antagonism, <1 synergy
    synergy_count: int
    antagonism_count: int
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dose_response(doses: list[float], responses: list[float]) -> None:
    """Validate doses and responses arrays."""
    if len(doses) != len(responses):
        raise ValueError(
            f"doses and responses must have equal length, got {len(doses)} vs {len(responses)}"
        )
    if len(doses) < _MIN_DATA_POINTS:
        raise ValueError(f"At least {_MIN_DATA_POINTS} data points required, got {len(doses)}")
    if any(d <= 0 for d in doses):
        raise ValueError("All doses must be > 0")
    if any(r < 0 or r > 100 for r in responses):
        raise ValueError("All responses must be in [0, 100]")


# ---------------------------------------------------------------------------
# Hill model
# ---------------------------------------------------------------------------


def _hill_model(dose: np.ndarray, emax: float, ed50: float, n: float) -> np.ndarray:
    """Hill equation: E = Emax * D^n / (ED50^n + D^n)."""
    d = np.asarray(dose, dtype=float)
    d_n = np.power(np.maximum(d, 0.0), n)
    ed50_n = ed50**n
    return emax * d_n / (ed50_n + d_n)


def _compute_r2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res < 1e-12 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def fit_hill_curve(doses: list[float], responses: list[float]) -> HillFitResult:
    """Fit a Hill (sigmoidal) dose-response curve.

    Model: E = Emax * D^n / (ED50^n + D^n)

    Parameters
    ----------
    doses:
        Dose values (all > 0), length >= 4.
    responses:
        Observed responses in [0, 100], same length as doses.

    Returns
    -------
    HillFitResult
        Fitted parameters with goodness-of-fit metrics.
    """
    _validate_dose_response(doses, responses)

    doses_arr = np.array(doses, dtype=float)
    responses_arr = np.array(responses, dtype=float)

    emax_init = float(np.max(responses_arr))
    if emax_init <= 0:
        emax_init = 100.0
    ed50_init = float(np.median(doses_arr))
    n_init = 1.0

    # Bounds: Emax in (0, 120], ED50 > 0, n in (0.1, 10)
    lower = [0.0, 1e-9, 0.1]
    upper = [120.0, np.max(doses_arr) * 100.0, 10.0]

    notes = "Hill curve fit successful"
    try:
        popt, _ = curve_fit(
            _hill_model,
            doses_arr,
            responses_arr,
            p0=[emax_init, ed50_init, n_init],
            bounds=(lower, upper),
            maxfev=10000,
        )
        emax, ed50, hill_n = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError) as exc:
        # Fallback: use empirical estimates
        emax = float(np.max(responses_arr))
        ed50 = float(np.median(doses_arr))
        hill_n = 1.0
        notes = f"Curve fit did not converge ({exc}); using empirical estimates"

    fitted = _hill_model(doses_arr, emax, ed50, hill_n)
    fitted_list = [float(v) for v in fitted]
    r2 = _compute_r2(responses_arr, fitted)
    rmse = float(np.sqrt(np.mean((responses_arr - fitted) ** 2)))

    return HillFitResult(
        emax=emax,
        ed50=ed50,
        hill_n=hill_n,
        r2=r2,
        rmse=rmse,
        fitted_responses=fitted_list,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dose-response surface
# ---------------------------------------------------------------------------


def dose_response_surface(
    drug1_doses: list[float],
    drug2_doses: list[float],
    responses_matrix: list[list[float]],
) -> DoseResponseSurfaceResult:
    """Fit Hill curves per drug and compute Bliss independence surface.

    Parameters
    ----------
    drug1_doses:
        Doses of drug 1 (rows of responses_matrix). All > 0, length >= 4.
    drug2_doses:
        Doses of drug 2 (columns of responses_matrix). All > 0, length >= 4.
    responses_matrix:
        Observed responses[i][j] for (drug1_doses[i], drug2_doses[j]),
        values in [0, 100]. Shape: (len(drug1_doses), len(drug2_doses)).

    Returns
    -------
    DoseResponseSurfaceResult
    """
    if len(drug1_doses) < _MIN_DATA_POINTS:
        raise ValueError(f"drug1_doses must have >= {_MIN_DATA_POINTS} entries")
    if len(drug2_doses) < _MIN_DATA_POINTS:
        raise ValueError(f"drug2_doses must have >= {_MIN_DATA_POINTS} entries")
    if any(d <= 0 for d in drug1_doses):
        raise ValueError("All drug1_doses must be > 0")
    if any(d <= 0 for d in drug2_doses):
        raise ValueError("All drug2_doses must be > 0")

    n1 = len(drug1_doses)
    n2 = len(drug2_doses)
    if len(responses_matrix) != n1:
        raise ValueError(f"responses_matrix must have {n1} rows, got {len(responses_matrix)}")
    for i, row in enumerate(responses_matrix):
        if len(row) != n2:
            raise ValueError(f"responses_matrix row {i} must have {n2} columns, got {len(row)}")
        if any(r < 0 or r > 100 for r in row):
            raise ValueError(f"responses_matrix row {i} values must be in [0, 100]")

    # Fit drug 1 using its dose-response (collapse over drug2 — use max response per dose)
    drug1_responses = [max(row) for row in responses_matrix]
    drug1_fit = fit_hill_curve(drug1_doses, drug1_responses)

    # Fit drug 2 using its dose-response (collapse over drug1 — use max response per dose)
    drug2_responses = [max(responses_matrix[i][j] for i in range(n1)) for j in range(n2)]
    drug2_fit = fit_hill_curve(drug2_doses, drug2_responses)

    # Compute Bliss independence for each combination and interaction index
    interaction_matrix: list[list[float]] = []
    synergy_count = 0
    antagonism_count = 0

    for i, d1 in enumerate(drug1_doses):
        row: list[float] = []
        for j, d2 in enumerate(drug2_doses):
            e1 = (
                float(
                    _hill_model(np.array([d1]), drug1_fit.emax, drug1_fit.ed50, drug1_fit.hill_n)[0]
                )
                / 100.0
            )
            e2 = (
                float(
                    _hill_model(np.array([d2]), drug2_fit.emax, drug2_fit.ed50, drug2_fit.hill_n)[0]
                )
                / 100.0
            )

            # Bliss independence (fractional scale 0-1, then back to %)
            bliss = (e1 + e2 - e1 * e2) * 100.0
            observed = float(responses_matrix[i][j])

            if bliss > 1e-9:
                ii = observed / bliss
            else:
                ii = 1.0  # undefined; assume additive

            row.append(round(ii, 4))

            if ii < 0.9:
                synergy_count += 1
            elif ii > 1.1:
                antagonism_count += 1

        interaction_matrix.append(row)

    notes = (
        f"Surface: {n1}x{n2} grid; synergy={synergy_count}, antagonism={antagonism_count}; "
        f"Drug1 R2={drug1_fit.r2:.3f}, Drug2 R2={drug2_fit.r2:.3f}"
    )

    return DoseResponseSurfaceResult(
        drug1_fit=drug1_fit,
        drug2_fit=drug2_fit,
        interaction_matrix=interaction_matrix,
        synergy_count=synergy_count,
        antagonism_count=antagonism_count,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Optimal combination
# ---------------------------------------------------------------------------


def optimal_combination(
    drug1_doses: list[float],
    drug2_doses: list[float],
    responses_matrix: list[list[float]],
    target_response: float = 50.0,
) -> dict:
    """Find the dose combination achieving target response with minimum total dose.

    Parameters
    ----------
    drug1_doses:
        Doses of drug 1 (rows).
    drug2_doses:
        Doses of drug 2 (columns).
    responses_matrix:
        Observed responses matrix.
    target_response:
        Target efficacy level (%). Default 50 (EC50 equivalent).

    Returns
    -------
    dict
        Keys: drug1_dose, drug2_dose, response, total_dose, distance_to_target, notes.
    """
    if target_response < 0 or target_response > 100:
        raise ValueError(f"target_response must be in [0, 100], got {target_response}")

    n1 = len(drug1_doses)
    n2 = len(drug2_doses)

    best: dict = {}
    best_score = float("inf")

    for i in range(n1):
        for j in range(n2):
            resp = float(responses_matrix[i][j])
            dist = abs(resp - target_response)
            total_dose = drug1_doses[i] + drug2_doses[j]
            # Score: primarily minimize distance, then total dose
            score = dist * 1000.0 + total_dose

            if score < best_score:
                best_score = score
                best = {
                    "drug1_dose": drug1_doses[i],
                    "drug2_dose": drug2_doses[j],
                    "response": resp,
                    "total_dose": total_dose,
                    "distance_to_target": dist,
                    "notes": (
                        f"Best combination at grid point ({i},{j}): "
                        f"D1={drug1_doses[i]}, D2={drug2_doses[j]}, "
                        f"response={resp:.1f}%, target={target_response}%"
                    ),
                }

    if not best:
        best = {
            "drug1_dose": drug1_doses[0],
            "drug2_dose": drug2_doses[0],
            "response": float(responses_matrix[0][0]),
            "total_dose": drug1_doses[0] + drug2_doses[0],
            "distance_to_target": abs(float(responses_matrix[0][0]) - target_response),
            "notes": "No valid combination found; returning first entry",
        }

    return best
