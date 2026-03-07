"""Concentration-Effect Hysteresis Quantification.

Quantifies clockwise vs counter-clockwise hysteresis in observed
concentration-effect data using:
- Shoelace formula for enclosed loop area
- Ascending/descending limb regression
- ke0 estimation from time-lag and optimisation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class HysteresisResult:
    """Result of hysteresis classification from observed C-E data."""

    hysteresis_area: float  # Shoelace formula result (signed)
    hysteresis_type: str  # 'clockwise' / 'counter_clockwise' / 'none'
    ascending_slope: float  # slope of log-conc vs effect on ascending limb
    descending_slope: float  # slope of log-conc vs effect on descending limb
    estimated_ke0_per_h: float  # estimated equilibration rate constant (1/h)
    t_lag_h: float  # time between Cmax and Emax
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    concentrations: list[float],
    effects: list[float],
) -> None:
    """Validate concentration and effect arrays."""
    if len(concentrations) != len(effects):
        raise ValueError("concentrations and effects must have the same length")
    if len(concentrations) < 4:
        raise ValueError("At least 4 data points are required")
    if any(c <= 0 for c in concentrations):
        raise ValueError("All concentrations must be > 0")
    if any(e < 0 or e > 100 for e in effects):
        raise ValueError("All effects must be in [0, 100]")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_hysteresis_area(
    concentrations: list[float],
    effects: list[float],
) -> float:
    """Compute area enclosed by C-E hysteresis loop using Shoelace formula.

    Parameters
    ----------
    concentrations:
        Observed plasma concentrations (any positive unit).
    effects:
        Corresponding pharmacodynamic effects in [0, 100].

    Returns
    -------
    float
        Signed area of the hysteresis loop.
        Positive  -> counter-clockwise (sensitisation / tolerance build-up lag).
        Negative  -> clockwise (acute tolerance / effect-compartment delay).
    """
    _validate_inputs(concentrations, effects)
    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(effects, dtype=float)
    # Shoelace formula (signed).
    # PK/PD sign convention:
    #   Negative = clockwise (effect lags concentration, effect-compartment delay)
    #   Positive = counter-clockwise (sensitisation / indirect-response)
    # The standard shoelace for a C-E loop where ascending limb < descending limb
    # (clockwise PK/PD) gives a positive value geometrically, so we negate.
    area = -0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def _split_ascending_descending(
    concentrations: list[float],
) -> tuple[int, list[int], list[int]]:
    """Return (cmax_idx, ascending_indices, descending_indices)."""
    c = np.asarray(concentrations, dtype=float)
    cmax_idx = int(np.argmax(c))
    ascending = list(range(0, cmax_idx + 1))
    descending = list(range(cmax_idx, len(c)))
    return cmax_idx, ascending, descending


def _linear_slope(log_conc: np.ndarray, effect: np.ndarray) -> float:
    """Least-squares slope of effect ~ log_conc."""
    if len(log_conc) < 2:
        return 0.0
    # Use numpy polyfit (degree 1)
    coeffs = np.polyfit(log_conc, effect, 1)
    return float(coeffs[0])


def classify_hysteresis(
    concentrations: list[float],
    effects: list[float],
    times_h: list[float],
) -> HysteresisResult:
    """Classify hysteresis type from observed concentration-effect-time data.

    Parameters
    ----------
    concentrations:
        Observed plasma concentrations (any positive unit).
    effects:
        Pharmacodynamic effects in [0, 100].
    times_h:
        Observation times in hours (same length as concentrations).

    Returns
    -------
    HysteresisResult
    """
    _validate_inputs(concentrations, effects)
    if len(times_h) != len(concentrations):
        raise ValueError("times_h must have the same length as concentrations")

    area = compute_hysteresis_area(concentrations, effects)

    c = np.asarray(concentrations, dtype=float)
    e = np.asarray(effects, dtype=float)
    t = np.asarray(times_h, dtype=float)

    cmax_idx, asc_idx, desc_idx = _split_ascending_descending(concentrations)

    # Ascending limb slope
    if len(asc_idx) >= 2:
        asc_slope = _linear_slope(np.log(c[asc_idx]), e[asc_idx])
    else:
        asc_slope = 0.0

    # Descending limb slope
    if len(desc_idx) >= 2:
        desc_slope = _linear_slope(np.log(c[desc_idx]), e[desc_idx])
    else:
        desc_slope = 0.0

    # Determine type
    threshold = 1e-6
    if abs(area) < threshold:
        htype = "none"
    elif area < 0:
        htype = "clockwise"
    else:
        htype = "counter_clockwise"

    # Time between Cmax and Emax
    emax_idx = int(np.argmax(e))
    t_cmax = float(t[cmax_idx])
    t_emax = float(t[emax_idx])
    t_lag = abs(t_emax - t_cmax)

    # Estimate ke0 from time-lag: ke0 ≈ ln(2) / t_lag
    if t_lag > 1e-9:
        ke0_est = math.log(2.0) / t_lag
    else:
        ke0_est = float("inf")

    # Notes
    if htype == "clockwise":
        notes = (
            "Clockwise hysteresis: effect lags behind concentration, "
            "consistent with effect-compartment delay or acute tolerance."
        )
    elif htype == "counter_clockwise":
        notes = (
            "Counter-clockwise hysteresis: effect precedes concentration decline, "
            "consistent with sensitisation or indirect-response mechanism."
        )
    else:
        notes = "No significant hysteresis detected."

    # Additional note on slope comparison
    if asc_slope > desc_slope + 1e-3:
        notes += " Ascending slope > descending slope: sensitisation pattern."
    elif desc_slope > asc_slope + 1e-3:
        notes += " Descending slope > ascending slope: tolerance pattern."

    return HysteresisResult(
        hysteresis_area=area,
        hysteresis_type=htype,
        ascending_slope=asc_slope,
        descending_slope=desc_slope,
        estimated_ke0_per_h=ke0_est,
        t_lag_h=t_lag,
        notes=notes,
    )


def fit_effect_compartment_ke0(
    times_h: list[float],
    concentrations: list[float],
    effects: list[float],
    emax: float = 100.0,
    ec50_init: float | None = None,
) -> float:
    """Fit ke0 by minimising least-squares error between simulated and observed effects.

    Simulates effect compartment via Forward Euler:
        dCe/dt = ke0 * (C(t) - Ce)
    Effect is computed with Hill equation (hill=1):
        E = Emax * Ce / (EC50 + Ce)

    Parameters
    ----------
    times_h:
        Observation times in hours.
    concentrations:
        Plasma concentrations at each time point (any positive unit).
    effects:
        Observed pharmacodynamic effects in [0, 100].
    emax:
        Maximum effect (default 100).
    ec50_init:
        Initial estimate of EC50. If None, uses median concentration.

    Returns
    -------
    float
        Best-fit ke0 in 1/h.
    """
    _validate_inputs(concentrations, effects)
    if len(times_h) != len(concentrations):
        raise ValueError("times_h must have the same length as concentrations")

    t = np.asarray(times_h, dtype=float)
    cp = np.asarray(concentrations, dtype=float)
    e_obs = np.asarray(effects, dtype=float)

    ec50 = ec50_init if ec50_init is not None else float(np.median(cp))
    if ec50 <= 0:
        ec50 = float(np.max(cp)) / 2.0

    def _simulate_effect(ke0: float) -> np.ndarray:
        n = len(t)
        ce = np.zeros(n)
        eff = np.zeros(n)
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            dce = ke0 * (cp[i] - ce[i])
            ce[i + 1] = max(ce[i] + dce * dt, 0.0)
        for i in range(n):
            eff[i] = emax * ce[i] / (ec50 + ce[i]) if (ec50 + ce[i]) > 0 else 0.0
        return eff

    def _objective(ke0: float) -> float:
        e_sim = _simulate_effect(ke0)
        return float(np.sum((e_sim - e_obs) ** 2))

    result = minimize_scalar(_objective, bounds=(0.01, 10.0), method="bounded")
    return float(result.x)


def plot_hysteresis_loop_data(
    concentrations: list[float],
    effects: list[float],
    ascending_idx: list[int],
    descending_idx: list[int],
) -> dict:
    """Return ascending and descending phase data for plotting the hysteresis loop.

    Parameters
    ----------
    concentrations:
        Full concentration array.
    effects:
        Full effect array.
    ascending_idx:
        Indices corresponding to the ascending (pre-Cmax) phase.
    descending_idx:
        Indices corresponding to the descending (post-Cmax) phase.

    Returns
    -------
    dict with keys:
        ascending_conc, ascending_effect, descending_conc, descending_effect
    """
    c = np.asarray(concentrations, dtype=float)
    e = np.asarray(effects, dtype=float)

    return {
        "ascending_conc": c[ascending_idx].tolist(),
        "ascending_effect": e[ascending_idx].tolist(),
        "descending_conc": c[descending_idx].tolist(),
        "descending_effect": e[descending_idx].tolist(),
    }


__all__ = [
    "HysteresisResult",
    "compute_hysteresis_area",
    "classify_hysteresis",
    "fit_effect_compartment_ke0",
    "plot_hysteresis_loop_data",
]
