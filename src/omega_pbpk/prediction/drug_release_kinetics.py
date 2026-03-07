"""Phase 221 — Drug Release Kinetics Model Fitting.

Fits and compares zero-order, first-order, Higuchi, and Korsmeyer-Peppas
dissolution models to in-vitro drug release data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseModelResult:
    """Result of fitting a release kinetic model to dissolution data."""

    drug_name: str
    model: str
    k_param: float
    n_param: float | None
    r_squared: float
    fitted_values: tuple
    release_mechanism: str
    notes: str


def _linspace(start: float, stop: float, num: int) -> list:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _ss_res(observed: list, predicted: list) -> float:
    return sum((o - p) ** 2 for o, p in zip(observed, predicted))


def _ss_tot(observed: list) -> float:
    mean_y = sum(observed) / len(observed)
    return sum((o - mean_y) ** 2 for o in observed)


def _r_squared(observed: list, predicted: list) -> float:
    ss_tot = _ss_tot(observed)
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - _ss_res(observed, predicted) / ss_tot


def _predict_zero_order(times: list, k0: float) -> list:
    return [k0 * t for t in times]


def _predict_first_order(times: list, k1: float) -> list:
    return [100.0 * (1.0 - math.exp(-k1 * t)) for t in times]


def _predict_higuchi(times: list, kH: float) -> list:
    return [kH * math.sqrt(t) for t in times]


def _predict_korsmeyer_peppas(times: list, kKP: float, n: float) -> list:
    result = []
    for t in times:
        if t <= 0:
            result.append(0.0)
        else:
            result.append(kKP * (t**n))
    return result


def _release_mechanism(model: str, n: float | None) -> str:
    if model == "zero_order":
        return "Zero-order (constant rate, erosion-controlled)"
    if model == "first_order":
        return "First-order (concentration-dependent, porous matrix)"
    if model == "higuchi":
        return "Higuchi (Fickian diffusion from matrix)"
    if model == "korsmeyer_peppas":
        if n is None:
            return "Korsmeyer-Peppas (power law)"
        if n < 0.45:
            return "Korsmeyer-Peppas (Fickian diffusion, n<0.45)"
        if n <= 0.89:
            return "Korsmeyer-Peppas (anomalous/non-Fickian transport, 0.45<=n<=0.89)"
        return "Korsmeyer-Peppas (Case II transport/erosion, n>0.89)"
    return "Unknown"


def fit_release_model(
    drug_name: str,
    times_min: list,
    pct_released: list,
    model: str,
) -> ReleaseModelResult:
    """Fit a single release kinetic model to dissolution data.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    times_min:
        List of time points in minutes.
    pct_released:
        List of percent released at each time point (0–100).
    model:
        One of "zero_order", "first_order", "higuchi", "korsmeyer_peppas".

    Returns
    -------
    ReleaseModelResult with best-fit parameters and R².
    """
    if len(times_min) < 3:
        raise ValueError("At least 3 data points are required for model fitting.")
    if len(times_min) != len(pct_released):
        raise ValueError(
            f"times_min and pct_released must have the same length "
            f"(got {len(times_min)} and {len(pct_released)})."
        )

    valid_models = {"zero_order", "first_order", "higuchi", "korsmeyer_peppas"}
    if model not in valid_models:
        raise ValueError(f"model must be one of {valid_models}, got '{model}'.")

    max_pct = max(pct_released)
    max_t = max(times_min)

    best_k = 0.0
    best_n: float | None = None
    best_sse = float("inf")
    best_predicted: list = []

    if model == "zero_order":
        upper_k0 = max_pct / max_t if max_t > 0 else 1.0
        for k0 in _linspace(0.01, upper_k0, 30):
            predicted = _predict_zero_order(times_min, k0)
            sse = _ss_res(pct_released, predicted)
            if sse < best_sse:
                best_sse = sse
                best_k = k0
                best_predicted = predicted

    elif model == "first_order":
        for k1 in _linspace(0.001, 0.2, 30):
            predicted = _predict_first_order(times_min, k1)
            sse = _ss_res(pct_released, predicted)
            if sse < best_sse:
                best_sse = sse
                best_k = k1
                best_predicted = predicted

    elif model == "higuchi":
        sqrt_max_t = math.sqrt(max_t) if max_t > 0 else 1.0
        upper_kH = max_pct / sqrt_max_t * 1.5
        for kH in _linspace(0.1, upper_kH, 30):
            predicted = _predict_higuchi(times_min, kH)
            sse = _ss_res(pct_released, predicted)
            if sse < best_sse:
                best_sse = sse
                best_k = kH
                best_predicted = predicted

    elif model == "korsmeyer_peppas":
        for kKP in _linspace(0.1, 20.0, 15):
            for n in _linspace(0.1, 1.5, 15):
                predicted = _predict_korsmeyer_peppas(times_min, kKP, n)
                sse = _ss_res(pct_released, predicted)
                if sse < best_sse:
                    best_sse = sse
                    best_k = kKP
                    best_n = n
                    best_predicted = predicted

    r2 = _r_squared(pct_released, best_predicted)
    # Clamp R² to reasonable range
    r2 = max(-1.0, min(1.0, r2))

    mechanism = _release_mechanism(model, best_n)

    if model == "korsmeyer_peppas":
        notes = (
            f"Best fit: kKP={best_k:.4f}, n={best_n:.4f}, R²={r2:.4f}. "
            f"Grid search over 15×15 parameter combinations."
        )
    else:
        param_name = {"zero_order": "k0", "first_order": "k1", "higuchi": "kH"}[model]
        notes = (
            f"Best fit: {param_name}={best_k:.4f}, R²={r2:.4f}. "
            f"Grid search over 30 parameter values."
        )

    return ReleaseModelResult(
        drug_name=drug_name,
        model=model,
        k_param=best_k,
        n_param=best_n,
        r_squared=r2,
        fitted_values=tuple(best_predicted),
        release_mechanism=mechanism,
        notes=notes,
    )


def compare_models(
    drug_name: str,
    times_min: list,
    pct_released: list,
) -> list:
    """Fit all four release models and return results sorted by R² descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    times_min:
        List of time points in minutes.
    pct_released:
        List of percent released at each time point (0–100).

    Returns
    -------
    List of ReleaseModelResult for all 4 models, sorted by R² descending.
    """
    models = ["zero_order", "first_order", "higuchi", "korsmeyer_peppas"]
    results = [fit_release_model(drug_name, times_min, pct_released, m) for m in models]
    return sorted(results, key=lambda r: r.r_squared, reverse=True)
