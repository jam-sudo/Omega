"""Compartmental model selection — 1- vs 2-compartment AIC/BIC analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz  # noqa: F401


@dataclass(frozen=True)
class CompartmentResult:
    drug_name: str
    n_compartments: int
    aic: float
    bic: float
    r_squared: float
    params: dict
    predicted_conc: list[float]
    residuals: list[float]
    half_lives: list[float]


def _fit_one_compartment(times: np.ndarray, conc: np.ndarray) -> tuple[dict, list[float]]:
    """Fit C = A*exp(-ke*t) via log-linear regression."""
    log_c = np.log(np.maximum(conc, 1e-12))
    coeffs = np.polyfit(times, log_c, 1)
    ke = float(-coeffs[0])
    A = float(math.exp(coeffs[1]))
    ke = max(ke, 1e-6)
    predicted = (A * np.exp(-ke * times)).tolist()
    params = {"A": A, "ke": ke, "thalf": math.log(2) / ke}
    return params, predicted


def _fit_two_compartment(times: np.ndarray, conc: np.ndarray) -> tuple[dict, list[float]]:
    """Fit C = A*exp(-alpha*t) + B*exp(-beta*t) via method of residuals."""
    n = len(times)
    n_terminal = max(int(0.4 * n), 2)
    t_term = times[-n_terminal:]
    c_term = conc[-n_terminal:]
    log_c_term = np.log(np.maximum(c_term, 1e-12))
    coeffs_beta = np.polyfit(t_term, log_c_term, 1)
    beta = float(max(-coeffs_beta[0], 1e-6))
    B = float(math.exp(coeffs_beta[1]))

    c_residual = conc - B * np.exp(-beta * times)
    c_residual = np.maximum(c_residual, 1e-12)
    n_early = max(int(0.5 * n), 2)
    log_resid = np.log(c_residual[:n_early])
    coeffs_alpha = np.polyfit(times[:n_early], log_resid, 1)
    alpha = float(max(-coeffs_alpha[0], beta + 1e-6))
    A = float(math.exp(coeffs_alpha[1]))

    predicted = (A * np.exp(-alpha * times) + B * np.exp(-beta * times)).tolist()
    params = {
        "A": A,
        "alpha": alpha,
        "B": B,
        "beta": beta,
        "thalf_alpha": math.log(2) / alpha,
        "thalf_beta": math.log(2) / beta,
    }
    return params, predicted


def calculate_aic(n_points: int, n_params: int, sse: float) -> float:
    """AIC = n*ln(SSE/n) + 2*k."""
    if sse <= 0:
        sse = 1e-12
    return n_points * math.log(sse / n_points) + 2 * n_params


def calculate_bic(n_points: int, n_params: int, sse: float) -> float:
    """BIC = n*ln(SSE/n) + k*ln(n)."""
    if sse <= 0:
        sse = 1e-12
    return n_points * math.log(sse / n_points) + n_params * math.log(n_points)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def select_compartment_model(
    drug_name: str,
    times: list[float],
    conc: list[float],
) -> CompartmentResult:
    """Fit 1- and 2-cpt models, select best by AIC."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(conc, dtype=float)

    params1, pred1 = _fit_one_compartment(t, c)
    pred1_arr = np.asarray(pred1)
    sse1 = float(np.sum((c - pred1_arr) ** 2))
    aic1 = calculate_aic(len(t), 2, sse1)  # 2 params: A, ke
    bic1 = calculate_bic(len(t), 2, sse1)
    r2_1 = _r_squared(c, pred1_arr)

    params2, pred2 = _fit_two_compartment(t, c)
    pred2_arr = np.asarray(pred2)
    sse2 = float(np.sum((c - pred2_arr) ** 2))
    aic2 = calculate_aic(len(t), 4, sse2)  # 4 params: A, alpha, B, beta
    bic2 = calculate_bic(len(t), 4, sse2)
    r2_2 = _r_squared(c, pred2_arr)

    if aic1 <= aic2:
        best_n = 1
        params = params1
        predicted = pred1
        aic = aic1
        bic = bic1
        r2 = r2_1
        residuals = (c - pred1_arr).tolist()
        halves = [params1["thalf"]]
    else:
        best_n = 2
        params = params2
        predicted = pred2
        aic = aic2
        bic = bic2
        r2 = r2_2
        residuals = (c - pred2_arr).tolist()
        halves = [params2["thalf_alpha"], params2["thalf_beta"]]

    return CompartmentResult(
        drug_name=drug_name,
        n_compartments=best_n,
        aic=aic,
        bic=bic,
        r_squared=r2,
        params=params,
        predicted_conc=predicted,
        residuals=residuals,
        half_lives=halves,
    )


def compare_subjects(subjects: list[dict]) -> dict:
    """Fit each subject; return summary of best model counts and mean AIC.

    Each subject dict must have keys: subject_id, times, conc.
    """
    results = [
        select_compartment_model(s.get("subject_id", "unknown"), s["times"], s["conc"])
        for s in subjects
    ]
    counts: dict[int, int] = {}
    for r in results:
        counts[r.n_compartments] = counts.get(r.n_compartments, 0) + 1
    mean_aic = float(np.mean([r.aic for r in results])) if results else float("nan")
    return {
        "best_model_counts": counts,
        "mean_aic": mean_aic,
        "results": results,
    }


__all__ = [
    "CompartmentResult",
    "calculate_aic",
    "calculate_bic",
    "select_compartment_model",
    "compare_subjects",
]
