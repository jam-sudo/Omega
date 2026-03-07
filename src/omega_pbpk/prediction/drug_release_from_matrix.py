"""Drug release from matrix tablets — Phase 267.

Multiple kinetic models for matrix drug release:
  zero_order, first_order, higuchi, korsmeyer_peppas, weibull.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_MODELS = ("zero_order", "first_order", "higuchi", "korsmeyer_peppas", "weibull")


@dataclass
class MatrixReleaseResult:
    """Result of matrix drug release simulation."""

    drug_name: str
    dose_mg: float
    release_model: str
    times_h: list
    released_mg: list
    released_pct: list
    t50_h: float
    t80_h: float
    release_rate_avg_mg_per_h: float
    fit_r2: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _find_time_for_pct(times: list, released_pct: list, target_pct: float) -> float:
    """Return interpolated time when released_pct first reaches target_pct."""
    for i in range(1, len(released_pct)):
        if released_pct[i] >= target_pct:
            # Linear interpolation
            t0 = times[i - 1]
            t1 = times[i]
            p0 = released_pct[i - 1]
            p1 = released_pct[i]
            if p1 > p0:
                frac = (target_pct - p0) / (p1 - p0)
                return t0 + frac * (t1 - t0)
            return t1
    # Never reached — return end time
    return times[-1]


def simulate_matrix_release(
    drug_name: str,
    dose_mg: float,
    release_model: str,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    k_release: float = 0.1,
    n_exponent: float = 0.5,
) -> MatrixReleaseResult:
    """Simulate drug release from a matrix tablet using a kinetic model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total drug dose (mg). Must be > 0.
    release_model : str
        One of: zero_order, first_order, higuchi, korsmeyer_peppas, weibull.
    t_end_h : float
        Simulation end time (h). Must be > 0.
    dt_h : float
        Time step (h).
    k_release : float
        Rate constant (units depend on model).
    n_exponent : float
        Diffusion exponent (for korsmeyer_peppas and weibull).

    Returns
    -------
    MatrixReleaseResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if release_model not in _VALID_MODELS:
        raise ValueError(f"release_model must be one of {_VALID_MODELS}, got '{release_model}'")

    n_steps = max(1, int(round(t_end_h / dt_h)))
    times_h: list = []
    released_mg: list = []
    released_pct: list = []

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)

        if release_model == "zero_order":
            t100 = 1.0 / k_release if k_release > 0 else 1e9
            q = dose_mg * t / t100

        elif release_model == "first_order":
            k1 = k_release
            q = dose_mg * (1.0 - math.exp(-k1 * t))

        elif release_model == "higuchi":
            kH = k_release * math.sqrt(dose_mg)
            q = kH * math.sqrt(t)

        elif release_model == "korsmeyer_peppas":
            if t == 0.0:
                q = 0.0
            else:
                q = dose_mg * min(1.0, (k_release * t) ** n_exponent)

        elif release_model == "weibull":
            b = 1.0 / k_release if k_release > 0 else 1e9
            a = n_exponent
            q = dose_mg * (1.0 - math.exp(-((t / b) ** a)))

        else:
            q = 0.0

        # Cap at dose_mg
        q = max(0.0, min(q, dose_mg))
        released_mg.append(q)
        released_pct.append(q / dose_mg * 100.0)

    t50_h = _find_time_for_pct(times_h, released_pct, 50.0)
    t80_h = _find_time_for_pct(times_h, released_pct, 80.0)

    # Average release rate over t80
    if t80_h > 0:
        release_rate_avg_mg_per_h = dose_mg * 0.80 / t80_h
    else:
        release_rate_avg_mg_per_h = 0.0

    # Notes
    final_pct = released_pct[-1]
    if final_pct >= 95.0:
        notes = f"{release_model}: complete release achieved by {t_end_h:.1f} h."
    elif final_pct >= 80.0:
        notes = f"{release_model}: extended release, {final_pct:.1f}% released by {t_end_h:.1f} h."
    else:
        notes = f"{release_model}: slow/incomplete release, {final_pct:.1f}% by {t_end_h:.1f} h — consider longer simulation."

    return MatrixReleaseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        release_model=release_model,
        times_h=times_h,
        released_mg=released_mg,
        released_pct=released_pct,
        t50_h=t50_h,
        t80_h=t80_h,
        release_rate_avg_mg_per_h=release_rate_avg_mg_per_h,
        fit_r2=1.0,
        notes=notes,
    )


def compare_release_models(
    drug_name: str,
    dose_mg: float,
    k_release: float = 0.1,
) -> list:
    """Compare all 5 release models, sorted by t50_h ascending (fastest first).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total dose (mg).
    k_release : float
        Shared rate constant for all models.

    Returns
    -------
    list[MatrixReleaseResult] sorted by t50_h ascending.
    """
    results = []
    for model in _VALID_MODELS:
        r = simulate_matrix_release(
            drug_name=drug_name,
            dose_mg=dose_mg,
            release_model=model,
            t_end_h=72.0,
            dt_h=0.5,
            k_release=k_release,
        )
        results.append(r)
    results.sort(key=lambda x: x.t50_h)
    return results


__all__ = [
    "MatrixReleaseResult",
    "compare_release_models",
    "simulate_matrix_release",
]
