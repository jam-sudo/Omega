"""Controlled release drug release kinetics — Phase 462.

Mathematical models for in vitro drug release from controlled release
formulations, including zero-order, first-order, Higuchi, and
Korsmeyer-Peppas models.  Also provides model fitting and profile
comparison via the f1/f2 similarity factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseModelFitResult:
    """Result of fitting multiple release models to observed data."""

    best_model: str  # 'zero_order' | 'first_order' | 'higuchi' | 'korsmeyer_peppas'
    r_squared_values: dict  # {model_name: R²}
    best_r_squared: float
    model_parameters: dict  # {param_name: value}
    release_mechanism: str  # 'fickian_diffusion' | 'case_ii_transport' | 'anomalous'
    notes: str


# ---------------------------------------------------------------------------
# Release models
# ---------------------------------------------------------------------------

def zero_order_release(
    t_h: float,
    k0_per_h: float,
    t_lag_h: float = 0.0,
) -> float:
    """Zero-order release: fraction = k0 * (t - t_lag), capped at 1.

    Parameters
    ----------
    t_h : float
        Time (h). Returns 0 if t_h < t_lag_h.
    k0_per_h : float
        Zero-order release rate constant (1/h).
    t_lag_h : float
        Lag time before release begins (h).

    Returns
    -------
    float
        Cumulative fraction released in [0, 1].
    """
    if k0_per_h < 0:
        raise ValueError("k0_per_h must be >= 0")
    if t_lag_h < 0:
        raise ValueError("t_lag_h must be >= 0")
    effective_t = t_h - t_lag_h
    if effective_t <= 0:
        return 0.0
    return min(k0_per_h * effective_t, 1.0)


def first_order_release(
    t_h: float,
    k1_per_h: float,
    t_lag_h: float = 0.0,
) -> float:
    """First-order release: fraction = 1 - exp(-k1 * (t - t_lag)).

    Parameters
    ----------
    t_h : float
        Time (h).
    k1_per_h : float
        First-order release rate constant (1/h).
    t_lag_h : float
        Lag time before release begins (h).

    Returns
    -------
    float
        Cumulative fraction released in [0, 1].
    """
    if k1_per_h < 0:
        raise ValueError("k1_per_h must be >= 0")
    if t_lag_h < 0:
        raise ValueError("t_lag_h must be >= 0")
    effective_t = t_h - t_lag_h
    if effective_t <= 0:
        return 0.0
    return min(1.0 - math.exp(-k1_per_h * effective_t), 1.0)


def higuchi_release(
    t_h: float,
    kH: float,
) -> float:
    """Higuchi matrix release: fraction = kH * sqrt(t).

    Parameters
    ----------
    t_h : float
        Time (h). Must be >= 0.
    kH : float
        Higuchi release constant (1/sqrt(h)).

    Returns
    -------
    float
        Cumulative fraction released (may exceed 1 for large t; caller should cap).
    """
    if kH < 0:
        raise ValueError("kH must be >= 0")
    if t_h < 0:
        raise ValueError("t_h must be >= 0")
    return kH * math.sqrt(t_h)


def korsmeyer_peppas(
    t_h: float,
    k_kp: float,
    n_exponent: float,
) -> float:
    """Korsmeyer-Peppas power-law release: fraction = k * t^n.

    n < 0.45  → Fickian diffusion (slab)
    n = 0.45  → Fickian diffusion boundary
    0.45 < n < 0.89 → anomalous transport
    n = 0.89  → Case-II transport (zero-order)
    n > 0.89  → super Case-II

    Parameters
    ----------
    t_h : float
        Time (h). Must be > 0 for power-law; returns 0 at t=0.
    k_kp : float
        Release rate constant.
    n_exponent : float
        Diffusion exponent.

    Returns
    -------
    float
        Cumulative fraction released.
    """
    if k_kp < 0:
        raise ValueError("k_kp must be >= 0")
    if t_h <= 0:
        return 0.0
    return k_kp * (t_h ** n_exponent)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ss_res(observed: list[float], predicted: list[float]) -> float:
    return sum((o - p) ** 2 for o, p in zip(observed, predicted, strict=False))


def _ss_tot(observed: list[float]) -> float:
    m = _mean(observed)
    return sum((o - m) ** 2 for o in observed)


def _r_squared(observed: list[float], predicted: list[float]) -> float:
    ss_t = _ss_tot(observed)
    if ss_t == 0:
        # Perfect constant signal — check if predictions also constant
        return 1.0 if all(abs(p - observed[0]) < 1e-12 for p in predicted) else 0.0
    return 1.0 - _ss_res(observed, predicted) / ss_t


def _ols_slope_intercept(x: list[float], y: list[float]) -> tuple[float, float]:
    """OLS linear regression: y = slope * x + intercept."""
    n = len(x)
    if n < 2:
        return 0.0, (y[0] if y else 0.0)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi ** 2 for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-15:
        return 0.0, _mean(y)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_release_model(
    times_h: list[float],
    fractions_released: list[float],
) -> ReleaseModelFitResult:
    """Fit multiple release models and select the best by R².

    Linear transformations used:
    - Zero-order  : F vs t  → slope = k0, intercept ~0
    - First-order : ln(1-F) vs t → slope = -k1
    - Higuchi     : F vs sqrt(t) → slope = kH
    - KP          : ln(F) vs ln(t) → slope = n, intercept = ln(k)

    Parameters
    ----------
    times_h : list[float]
        Observation times (h). Must have >= 2 points.
    fractions_released : list[float]
        Observed cumulative fractions released (0–1).

    Returns
    -------
    ReleaseModelFitResult
    """
    if len(times_h) < 2:
        raise ValueError("At least 2 data points required for fitting")
    if len(times_h) != len(fractions_released):
        raise ValueError("times_h and fractions_released must have the same length")
    if any(f < 0 or f > 1.0 + 1e-9 for f in fractions_released):
        raise ValueError("fractions_released must be in [0, 1]")

    # Cap fractions at 1 for safety
    frac = [min(f, 1.0) for f in fractions_released]
    t = times_h

    r2_values: dict[str, float] = {}
    params: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Zero-order: F = k0 * t  → linear regression F vs t (force through ~0)
    # ------------------------------------------------------------------
    slope_zo, intercept_zo = _ols_slope_intercept(t, frac)
    k0 = max(slope_zo, 1e-9)
    pred_zo = [k0 * ti + intercept_zo for ti in t]
    r2_values["zero_order"] = _r_squared(frac, pred_zo)
    params["zero_order"] = {"k0_per_h": k0, "intercept": intercept_zo}

    # ------------------------------------------------------------------
    # First-order: ln(1 - F) = -k1 * t  → slope = -k1
    # Filter out F >= 1 (undefined ln)
    # ------------------------------------------------------------------
    fo_pairs = [(ti, fi) for ti, fi in zip(t, frac, strict=False) if fi < 1.0]
    if len(fo_pairs) >= 2:
        t_fo = [p[0] for p in fo_pairs]
        y_fo = [math.log(1.0 - p[1]) for p in fo_pairs]
        slope_fo, intercept_fo = _ols_slope_intercept(t_fo, y_fo)
        k1 = max(-slope_fo, 1e-9)
        # Reconstruct predicted fractions for full dataset
        pred_fo_full = [1.0 - math.exp(slope_fo * ti + intercept_fo) for ti in t]
        r2_values["first_order"] = _r_squared(frac, pred_fo_full)
        params["first_order"] = {"k1_per_h": k1}
    else:
        r2_values["first_order"] = 0.0
        params["first_order"] = {"k1_per_h": 0.0}
        k1 = 0.0

    # ------------------------------------------------------------------
    # Higuchi: F = kH * sqrt(t) → linear regression F vs sqrt(t)
    # ------------------------------------------------------------------
    sqrt_t = [math.sqrt(max(ti, 0.0)) for ti in t]
    slope_h, intercept_h = _ols_slope_intercept(sqrt_t, frac)
    kH = max(slope_h, 1e-9)
    pred_h = [kH * st + intercept_h for st in sqrt_t]
    r2_values["higuchi"] = _r_squared(frac, pred_h)
    params["higuchi"] = {"kH": kH}

    # ------------------------------------------------------------------
    # Korsmeyer-Peppas: ln(F) = n*ln(t) + ln(k) → linear in log-log
    # Filter F>0 and t>0
    # ------------------------------------------------------------------
    kp_pairs = [(ti, fi) for ti, fi in zip(t, frac, strict=False) if ti > 0 and fi > 0]
    if len(kp_pairs) >= 2:
        ln_t_kp = [math.log(p[0]) for p in kp_pairs]
        ln_f_kp = [math.log(p[1]) for p in kp_pairs]
        slope_kp, intercept_kp = _ols_slope_intercept(ln_t_kp, ln_f_kp)
        n_exp = slope_kp
        k_kp = math.exp(intercept_kp)
        pred_kp_full = [k_kp * (ti ** n_exp) if ti > 0 else 0.0 for ti in t]
        r2_values["korsmeyer_peppas"] = _r_squared(frac, pred_kp_full)
        params["korsmeyer_peppas"] = {"k_kp": k_kp, "n_exponent": n_exp}
    else:
        r2_values["korsmeyer_peppas"] = 0.0
        params["korsmeyer_peppas"] = {"k_kp": 0.0, "n_exponent": 0.0}
        n_exp = 0.0

    # ------------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------------
    best_model = max(r2_values, key=lambda m: r2_values[m])
    best_r2 = r2_values[best_model]
    best_params = params[best_model]

    # ------------------------------------------------------------------
    # Release mechanism from Korsmeyer-Peppas n exponent
    # ------------------------------------------------------------------
    n_kp = params["korsmeyer_peppas"].get("n_exponent", 0.45)
    if n_kp <= 0.45:
        mechanism = "fickian_diffusion"
    elif n_kp >= 0.89:
        mechanism = "case_ii_transport"
    else:
        mechanism = "anomalous"

    notes = (
        f"Best model: {best_model} (R²={best_r2:.4f}); "
        f"KP n={n_kp:.3f} → {mechanism}"
    )

    return ReleaseModelFitResult(
        best_model=best_model,
        r_squared_values=r2_values,
        best_r_squared=best_r2,
        model_parameters=best_params,
        release_mechanism=mechanism,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def compare_release_profiles(
    times_h: list[float],
    profile1: list[float],
    profile2: list[float],
) -> dict:
    """Compute f1 and f2 similarity factors between two dissolution profiles.

    FDA guidance (1997):
        f1 = (sum|R_t - T_t|) / (sum R_t) * 100
        f2 = 50 * log10(sqrt(1 / (1 + mean((R_t - T_t)^2))) * 100)

    Profiles are considered similar when f2 >= 50.

    Parameters
    ----------
    times_h : list[float]
        Time points (h) common to both profiles.
    profile1 : list[float]
        Reference cumulative fractions released (0–1).
    profile2 : list[float]
        Test cumulative fractions released (0–1).

    Returns
    -------
    dict with keys:
        'f1_similarity' : float
        'f2_similarity' : float
        'similar'       : bool (f2 >= 50)
        'notes'         : str
    """
    if len(times_h) != len(profile1) or len(times_h) != len(profile2):
        raise ValueError("times_h, profile1, and profile2 must have the same length")
    if len(times_h) < 1:
        raise ValueError("At least 1 time point required")

    n = len(times_h)

    # f1: difference factor
    sum_ref = sum(profile1)
    sum_diff = sum(abs(r - t) for r, t in zip(profile1, profile2, strict=False))
    f1 = (sum_diff / sum_ref * 100) if sum_ref > 0 else 0.0

    # f2: similarity factor (FDA 1997)
    # Differences are computed on the percentage (0-100) scale.
    # f2 = 50 * log10(sqrt(1 / (1 + mean((Rt - Tt)^2))) * 100)
    # where Rt, Tt are expressed as percent dissolved (fraction * 100).
    mean_sq_diff_pct = (
        sum(((r - t) * 100.0) ** 2 for r, t in zip(profile1, profile2, strict=False)) / n
    )
    f2 = 50.0 * math.log10(math.sqrt(1.0 / (1.0 + mean_sq_diff_pct)) * 100.0)

    similar = f2 >= 50.0

    notes = (
        f"n={n} time points; f1={f1:.2f}, f2={f2:.2f}; "
        f"{'similar' if similar else 'not similar'} (f2 threshold=50)"
    )

    return {
        "f1_similarity": f1,
        "f2_similarity": f2,
        "similar": similar,
        "notes": notes,
    }
