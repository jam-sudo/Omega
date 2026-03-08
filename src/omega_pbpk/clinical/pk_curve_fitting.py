"""Phase 527 — Concentration-Time Curve Fitting.

Fit observed PK data to 1-compartment models using grid search (pure Python).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKCurveFitResult:
    """Result of PK curve fitting."""

    drug_name: str
    model_type: str  # "1cpt_iv" or "1cpt_oral"
    dose_mg: float
    n_obs: int
    vd_fit_L: float
    cl_fit_L_per_h: float
    ka_fit_per_h: float  # -1.0 for IV
    t_half_fit_h: float
    rss: float
    r_squared: float
    fit_quality: str  # "excellent", "good", "poor"
    notes: str


def _predict_1cpt_iv(dose_mg: float, vd: float, cl: float, t: float) -> float:
    """Predict concentration for 1-compartment IV model."""
    if vd <= 0 or cl <= 0:
        return 0.0
    k = cl / vd
    return (dose_mg / vd) * math.exp(-k * t)


def _predict_1cpt_oral(
    dose_mg: float, vd: float, cl: float, ka: float, f: float, t: float
) -> float:
    """Predict concentration for 1-compartment oral model (F=1 unless specified)."""
    if vd <= 0 or cl <= 0 or ka <= 0:
        return 0.0
    k = cl / vd
    if abs(ka - k) < 1e-9:
        # Avoid division by zero — slightly offset ka
        ka = k * 1.001
    c = dose_mg * f * ka / (vd * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
    return max(0.0, c)


def _compute_rss(observed: list, predicted: list) -> float:
    """Compute residual sum of squares."""
    return sum((o - p) ** 2 for o, p in zip(observed, predicted))


def _compute_r_squared(observed: list, predicted: list) -> float:
    """Compute R² coefficient of determination."""
    n = len(observed)
    if n == 0:
        return 0.0
    mean_obs = sum(observed) / n
    tss = sum((o - mean_obs) ** 2 for o in observed)
    if tss < 1e-15:
        # All observed values equal; if predictions match, R²=1
        rss = _compute_rss(observed, predicted)
        return 1.0 if rss < 1e-12 else 0.0
    rss = _compute_rss(observed, predicted)
    r2 = 1.0 - rss / tss
    # Clamp to [0, 1]
    return max(0.0, min(1.0, r2))


def _fit_quality(r_squared: float) -> str:
    """Classify fit quality."""
    if r_squared > 0.95:
        return "excellent"
    elif r_squared > 0.85:
        return "good"
    else:
        return "poor"


def fit_pk_curve(
    drug_name: str,
    dose_mg: float,
    times_h: list,
    concentrations_mg_L: list,
    model_type: str = "1cpt_iv",
) -> PKCurveFitResult:
    """Fit observed PK concentration-time data to a compartmental model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    times_h : list of float
        Observation time points in hours.
    concentrations_mg_L : list of float
        Observed plasma concentrations in mg/L.
    model_type : str
        One of "1cpt_iv" or "1cpt_oral".

    Returns
    -------
    PKCurveFitResult
    """
    # --- Validation ---
    if model_type not in {"1cpt_iv", "1cpt_oral"}:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be '1cpt_iv' or '1cpt_oral'.")
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError(
            "times_h and concentrations_mg_L must have the same length. "
            f"Got {len(times_h)} and {len(concentrations_mg_L)}."
        )
    if len(times_h) < 3:
        raise ValueError(f"At least 3 observations required for fitting. Got {len(times_h)}.")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive. Got {dose_mg}.")

    obs = list(concentrations_mg_L)
    ts = list(times_h)
    n = len(obs)

    best_rss = float("inf")
    best_vd = 1.0
    best_cl = 1.0
    best_ka = -1.0

    if model_type == "1cpt_iv":
        # Coarse log-spaced grid: Vd in [1, 500], CL in [0.1, 100] — 30×30
        vd_min, vd_max, n_vd = 1.0, 500.0, 30
        cl_min, cl_max, n_cl = 0.1, 100.0, 30

        vd_values = [vd_min * (vd_max / vd_min) ** (i / (n_vd - 1)) for i in range(n_vd)]
        cl_values = [cl_min * (cl_max / cl_min) ** (i / (n_cl - 1)) for i in range(n_cl)]

        for vd in vd_values:
            for cl in cl_values:
                predicted = [_predict_1cpt_iv(dose_mg, vd, cl, t) for t in ts]
                rss = _compute_rss(obs, predicted)
                if rss < best_rss:
                    best_rss = rss
                    best_vd = vd
                    best_cl = cl

        # Refinement: zoom-in grid around best point (3 rounds)
        for _ in range(3):
            vd_lo = best_vd * 0.7
            vd_hi = best_vd * 1.3
            cl_lo = best_cl * 0.7
            cl_hi = best_cl * 1.3
            vd_ref = [vd_lo + (vd_hi - vd_lo) * i / 19 for i in range(20)]
            cl_ref = [cl_lo + (cl_hi - cl_lo) * i / 19 for i in range(20)]
            for vd in vd_ref:
                for cl in cl_ref:
                    if vd <= 0 or cl <= 0:
                        continue
                    predicted = [_predict_1cpt_iv(dose_mg, vd, cl, t) for t in ts]
                    rss = _compute_rss(obs, predicted)
                    if rss < best_rss:
                        best_rss = rss
                        best_vd = vd
                        best_cl = cl

        best_ka = -1.0

    else:  # 1cpt_oral
        # Coarse grid: Vd in [1, 500], CL in [0.1, 100], ka in [0.1, 10] — 15×15×12
        vd_min, vd_max, n_vd = 1.0, 500.0, 15
        cl_min, cl_max, n_cl = 0.1, 100.0, 15
        ka_min, ka_max, n_ka = 0.1, 10.0, 12

        vd_values = [vd_min * (vd_max / vd_min) ** (i / (n_vd - 1)) for i in range(n_vd)]
        cl_values = [cl_min * (cl_max / cl_min) ** (i / (n_cl - 1)) for i in range(n_cl)]
        ka_values = [ka_min * (ka_max / ka_min) ** (i / (n_ka - 1)) for i in range(n_ka)]

        f = 1.0  # assume F=1
        for vd in vd_values:
            for cl in cl_values:
                for ka in ka_values:
                    k = cl / vd
                    if abs(ka - k) < 1e-6:
                        continue
                    predicted = [_predict_1cpt_oral(dose_mg, vd, cl, ka, f, t) for t in ts]
                    rss = _compute_rss(obs, predicted)
                    if rss < best_rss:
                        best_rss = rss
                        best_vd = vd
                        best_cl = cl
                        best_ka = ka

        # Refinement: zoom-in grid around best point (3 rounds)
        for _ in range(3):
            vd_lo = best_vd * 0.7
            vd_hi = best_vd * 1.3
            cl_lo = best_cl * 0.7
            cl_hi = best_cl * 1.3
            ka_lo = best_ka * 0.7
            ka_hi = best_ka * 1.3
            vd_ref = [vd_lo + (vd_hi - vd_lo) * i / 14 for i in range(15)]
            cl_ref = [cl_lo + (cl_hi - cl_lo) * i / 14 for i in range(15)]
            ka_ref = [ka_lo + (ka_hi - ka_lo) * i / 11 for i in range(12)]
            for vd in vd_ref:
                for cl in cl_ref:
                    for ka in ka_ref:
                        if vd <= 0 or cl <= 0 or ka <= 0:
                            continue
                        k = cl / vd
                        if abs(ka - k) < 1e-6:
                            continue
                        predicted = [_predict_1cpt_oral(dose_mg, vd, cl, ka, f, t) for t in ts]
                        rss = _compute_rss(obs, predicted)
                        if rss < best_rss:
                            best_rss = rss
                            best_vd = vd
                            best_cl = cl
                            best_ka = ka

    # Compute R²
    if model_type == "1cpt_iv":
        predicted_best = [_predict_1cpt_iv(dose_mg, best_vd, best_cl, t) for t in ts]
    else:
        predicted_best = [
            _predict_1cpt_oral(dose_mg, best_vd, best_cl, best_ka, 1.0, t) for t in ts
        ]

    r2 = _compute_r_squared(obs, predicted_best)
    t_half = 0.693 * best_vd / best_cl
    quality = _fit_quality(r2)

    notes = (
        f"Grid search fit for {model_type}. "
        f"Vd={best_vd:.2f} L, CL={best_cl:.2f} L/h, t½={t_half:.2f} h."
    )

    return PKCurveFitResult(
        drug_name=drug_name,
        model_type=model_type,
        dose_mg=dose_mg,
        n_obs=n,
        vd_fit_L=best_vd,
        cl_fit_L_per_h=best_cl,
        ka_fit_per_h=best_ka,
        t_half_fit_h=t_half,
        rss=best_rss,
        r_squared=r2,
        fit_quality=quality,
        notes=notes,
    )


def predict_from_fit(result: PKCurveFitResult, times_h: list) -> list:
    """Generate predicted concentrations from a fitted PK model.

    Parameters
    ----------
    result : PKCurveFitResult
        Fitted PK model result.
    times_h : list of float
        Time points at which to predict concentrations.

    Returns
    -------
    list of float
        Predicted concentrations in mg/L.
    """
    dose_mg = result.dose_mg
    vd = result.vd_fit_L
    cl = result.cl_fit_L_per_h

    if result.model_type == "1cpt_iv":
        return [_predict_1cpt_iv(dose_mg, vd, cl, t) for t in times_h]
    else:
        ka = result.ka_fit_per_h
        return [_predict_1cpt_oral(dose_mg, vd, cl, ka, 1.0, t) for t in times_h]
