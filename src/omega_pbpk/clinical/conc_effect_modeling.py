"""Concentration-effect (PK/PD) modeling.

Fits sigmoid Emax and related models to observed concentration-effect data
using grid search (no external optimizers).  Supports:

  * emax        — simple hyperbolic Emax (Hill = 1)
  * sigmoid_emax — Hill equation with variable Hill coefficient
  * linear      — linear E = E0 + slope * C
  * power       — mono-exponential saturation E = E0 + Emax * (1 - exp(-k*C))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcEffectResult:
    """Fitted concentration-effect model result."""

    drug_name: str
    model_type: str  # "emax", "sigmoid_emax", "linear", "power"
    emax: float
    ec50: float
    hill_coef: float  # 1.0 for simple Emax / linear / power
    e0: float  # baseline effect
    r_squared: float  # goodness of fit
    rmse: float
    ec10: float  # concentration for 10% of max additional effect
    ec90: float  # concentration for 90% of max additional effect
    therapeutic_index: float  # ec90 / ec10 (steepness; < 10 = steep)
    predicted_effects: list  # predicted E at each input concentration
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _r_squared(observed: list[float], predicted: list[float]) -> float:
    """Coefficient of determination R²."""
    n = len(observed)
    if n == 0:
        return 0.0
    mean_obs = sum(observed) / n
    ss_tot = sum((y - mean_obs) ** 2 for y in observed)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(observed, predicted))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(observed: list[float], predicted: list[float]) -> float:
    """Root mean squared error."""
    n = len(observed)
    if n == 0:
        return 0.0
    mse = sum((y - yh) ** 2 for y, yh in zip(observed, predicted)) / n
    return math.sqrt(mse)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """Return n evenly spaced values in [lo, hi]."""
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _logspace(lo: float, hi: float, n: int) -> list[float]:
    """Return n log-spaced values in [lo, hi] (lo, hi > 0)."""
    if n == 1:
        return [lo]
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    step = (log_hi - log_lo) / (n - 1)
    return [10.0 ** (log_lo + i * step) for i in range(n)]


def _predict_emax(c: float, e0: float, emax: float, ec50: float, hill: float) -> float:
    """Hill / sigmoid-Emax prediction."""
    if c <= 0.0:
        return e0
    cn = c**hill
    ec50n = ec50**hill
    denom = ec50n + cn
    if denom == 0.0:
        return e0
    return e0 + (emax - e0) * cn / denom


def _predict_linear(c: float, e0: float, slope: float) -> float:
    return e0 + slope * c


def _predict_power(c: float, e0: float, emax: float, k: float) -> float:
    return e0 + emax * (1.0 - math.exp(-k * c))


def _ec_from_hill(ec50: float, hill: float, frac: float) -> float:
    """Compute EC_frac (frac of span above E0) using the Hill inverse.

    frac in (0, 1).
    EC_frac = EC50 * (frac / (1 - frac))^(1/hill)
    """
    if hill <= 0.0 or ec50 <= 0.0:
        return ec50
    ratio = frac / (1.0 - frac)
    return ec50 * (ratio ** (1.0 / hill))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_emax_model(
    drug_name: str,
    concentrations: list[float],
    effects: list[float],
    model_type: str = "sigmoid_emax",
) -> ConcEffectResult:
    """Fit a concentration-effect model to observed data.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    concentrations:
        Measured concentrations (mg/L).  Must be same length as *effects*.
    effects:
        Observed effects (% or response units).
    model_type:
        One of ``"emax"``, ``"sigmoid_emax"``, ``"linear"``, ``"power"``.

    Returns
    -------
    ConcEffectResult
    """
    # --- validation ---------------------------------------------------------
    if len(concentrations) != len(effects):
        raise ValueError(
            f"concentrations and effects must have the same length "
            f"({len(concentrations)} vs {len(effects)})"
        )
    if len(concentrations) < 3:
        raise ValueError(
            f"At least 3 data points are required for fitting (got {len(concentrations)})"
        )

    # --- baseline -----------------------------------------------------------
    e0 = effects[0] if concentrations[0] == 0.0 else 0.0

    c_min = (
        min(c for c in concentrations if c > 0.0) if any(c > 0 for c in concentrations) else 1e-6
    )
    c_max = max(concentrations)
    e_max_obs = max(effects)
    e_min_obs = min(effects)
    e_range = e_max_obs - e_min_obs

    best_ss = math.inf
    best_params: tuple = ()

    if model_type in ("emax", "sigmoid_emax"):
        # Use log-spaced grid for EC50 for better resolution across orders of magnitude
        emax_grid = _linspace(e0, e_max_obs * 2.0, 30)
        ec50_grid = _logspace(c_min * 0.1, c_max * 10.0, 30)

        if model_type == "emax":
            for em in emax_grid:
                for ec50 in ec50_grid:
                    ss = sum(
                        (_predict_emax(c, e0, em, ec50, 1.0) - y) ** 2
                        for c, y in zip(concentrations, effects)
                    )
                    if ss < best_ss:
                        best_ss = ss
                        best_params = (em, ec50, 1.0)
        else:  # sigmoid_emax
            hill_grid = _linspace(0.5, 4.0, 12)
            for em in emax_grid:
                for ec50 in ec50_grid:
                    for hill in hill_grid:
                        ss = sum(
                            (_predict_emax(c, e0, em, ec50, hill) - y) ** 2
                            for c, y in zip(concentrations, effects)
                        )
                        if ss < best_ss:
                            best_ss = ss
                            best_params = (em, ec50, hill)

        emax_fit, ec50_fit, hill_fit = best_params
        predicted = [_predict_emax(c, e0, emax_fit, ec50_fit, hill_fit) for c in concentrations]
        ec10 = _ec_from_hill(ec50_fit, hill_fit, 0.1)
        ec90 = _ec_from_hill(ec50_fit, hill_fit, 0.9)
        notes = f"Grid search ({model_type}): Emax={emax_fit:.3f}, EC50={ec50_fit:.4f}, n={hill_fit:.2f}"

    elif model_type == "linear":
        # E = E0 + slope * C; grid search over slope
        slope_max = e_range / c_min if c_min > 0 else 1.0
        slope_grid = _linspace(0.0, slope_max * 2.0, 20)
        best_slope = 0.0
        for slope in slope_grid:
            ss = sum(
                (_predict_linear(c, e0, slope) - y) ** 2 for c, y in zip(concentrations, effects)
            )
            if ss < best_ss:
                best_ss = ss
                best_slope = slope

        predicted = [_predict_linear(c, e0, best_slope) for c in concentrations]
        # Effective Emax at max concentration
        emax_fit = best_slope * c_max
        ec50_fit = c_max / 2.0  # half-max concentration in linear regime
        hill_fit = 1.0
        # For linear: EC10 and EC90 are fractions along the linear range
        # Use span = emax_fit; EC_f = f * Emax / slope
        ec10 = 0.1 * emax_fit / best_slope if best_slope > 0 else 0.0
        ec90 = 0.9 * emax_fit / best_slope if best_slope > 0 else c_max
        notes = f"Linear model: slope={best_slope:.4f}, E0={e0:.3f}"

    elif model_type == "power":
        emax_grid = _linspace(e0, e_max_obs * 2.0, 20)
        # k (rate constant): span [1e-4 * c_max, 10 / c_min]
        k_lo = 1e-4 / c_max if c_max > 0 else 1e-6
        k_hi = 10.0 / c_min if c_min > 0 else 10.0
        k_grid = _linspace(k_lo, k_hi, 20)

        for em in emax_grid:
            for k in k_grid:
                ss = sum(
                    (_predict_power(c, e0, em, k) - y) ** 2 for c, y in zip(concentrations, effects)
                )
                if ss < best_ss:
                    best_ss = ss
                    best_params = (em, k)

        emax_fit, k_fit = best_params
        predicted = [_predict_power(c, e0, emax_fit, k_fit) for c in concentrations]
        # EC50 from power model: Emax * (1 - exp(-k*C)) = 0.5*Emax => C = -ln(0.5)/k
        ec50_fit = math.log(2.0) / k_fit if k_fit > 0 else c_max / 2.0
        hill_fit = 1.0
        ec10 = -math.log(0.9) / k_fit if k_fit > 0 else 0.0  # 10% saturation
        ec90 = -math.log(0.1) / k_fit if k_fit > 0 else c_max  # 90% saturation
        notes = f"Power (mono-exp) model: Emax={emax_fit:.3f}, k={k_fit:.4f}, E0={e0:.3f}"

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Must be one of: 'emax', 'sigmoid_emax', 'linear', 'power'."
        )

    r2 = _r_squared(effects, predicted)
    rmse_val = _rmse(effects, predicted)

    # Ensure ec10 < ec50 < ec90 and all positive
    ec10 = max(ec10, 1e-12)
    ec90 = max(ec90, ec10 * 1.01)
    ti = ec90 / ec10 if ec10 > 0 else 1.0

    return ConcEffectResult(
        drug_name=drug_name,
        model_type=model_type,
        emax=emax_fit,
        ec50=ec50_fit,
        hill_coef=hill_fit,
        e0=e0,
        r_squared=r2,
        rmse=rmse_val,
        ec10=ec10,
        ec90=ec90,
        therapeutic_index=ti,
        predicted_effects=predicted,
        notes=notes,
    )


def predict_effect(drug_name: str, concentration_mg_L: float, result: ConcEffectResult) -> float:
    """Apply a fitted model to a new concentration.

    Parameters
    ----------
    drug_name:
        Name used for annotation (not used in computation).
    concentration_mg_L:
        New concentration (mg/L).
    result:
        Previously fitted :class:`ConcEffectResult`.

    Returns
    -------
    float
        Predicted effect.
    """
    c = concentration_mg_L
    mt = result.model_type

    if mt in ("emax", "sigmoid_emax"):
        return _predict_emax(c, result.e0, result.emax, result.ec50, result.hill_coef)
    elif mt == "linear":
        slope = result.emax / result.ec50 / 2.0 if result.ec50 > 0 else 0.0
        # Reconstruct slope from stored emax / c_max relationship
        # More precisely: emax_fit = slope * c_max => slope = emax_fit / c_max
        # We stored ec50_fit = c_max/2 => c_max = 2 * ec50_fit
        slope = result.emax / (2.0 * result.ec50) if result.ec50 > 0 else 0.0
        return _predict_linear(c, result.e0, slope)
    elif mt == "power":
        # k = ln(2) / ec50  (from ec50 = ln(2)/k)
        k = math.log(2.0) / result.ec50 if result.ec50 > 0 else 1.0
        return _predict_power(c, result.e0, result.emax, k)
    else:
        raise ValueError(f"Unknown model_type '{mt}' in result.")
