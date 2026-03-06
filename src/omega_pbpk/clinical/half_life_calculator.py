"""Comprehensive half-life calculations for PK dosing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HalfLifeResult",
    "calculate_half_life",
    "optimal_dosing_interval",
    "half_life_from_data",
]

_N_HALF_LIVES_TO_SS = math.log(1.0 / (1.0 - 0.95)) / math.log(2.0)  # ~4.3219


@dataclass
class HalfLifeResult:
    drug_name: str
    t_half_h: float
    t_half_alpha_h: float | None
    t_half_beta_h: float | None
    n_half_lives_to_steady_state: float
    n_half_lives_to_washout: float
    tss_h: float
    twash_h: float
    accumulation_ratio: float
    dosing_interval_h: float
    recommended_doses_per_day: float


def calculate_half_life(
    drug_name: str,
    t_half_h: float,
    dosing_interval_h: float | None = None,
    target_fluctuation_pct: float = 100.0,
) -> HalfLifeResult:
    """Calculate half-life-derived PK dosing parameters.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    t_half_h:
        Effective half-life in hours. Must be > 0.
    dosing_interval_h:
        Dosing interval tau in hours. Defaults to t_half_h when None.
    target_fluctuation_pct:
        Target peak-to-trough fluctuation (unused in computation, kept for API
        compatibility).

    Returns
    -------
    HalfLifeResult
    """
    if t_half_h <= 0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")

    tau = t_half_h if dosing_interval_h is None else dosing_interval_h

    accumulation_ratio = 1.0 / (1.0 - math.exp(-0.693 * tau / t_half_h))
    recommended_doses_per_day = 24.0 / tau
    tss_h = _N_HALF_LIVES_TO_SS * t_half_h

    return HalfLifeResult(
        drug_name=drug_name,
        t_half_h=t_half_h,
        t_half_alpha_h=None,
        t_half_beta_h=None,
        n_half_lives_to_steady_state=_N_HALF_LIVES_TO_SS,
        n_half_lives_to_washout=_N_HALF_LIVES_TO_SS,
        tss_h=tss_h,
        twash_h=tss_h,
        accumulation_ratio=accumulation_ratio,
        dosing_interval_h=tau,
        recommended_doses_per_day=recommended_doses_per_day,
    )


def optimal_dosing_interval(
    drug_name: str,
    t_half_h: float,
    target_accumulation_ratio: float = 2.0,
) -> HalfLifeResult:
    """Determine the dosing interval that achieves a desired accumulation ratio.

    Solves R = 1 / (1 - exp(-0.693*tau/t_half)) for tau:
        tau = t_half * log2(R / (R - 1))

    tau is clamped to [0.5, 48] hours.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    t_half_h:
        Effective half-life in hours. Must be > 0.
    target_accumulation_ratio:
        Desired steady-state accumulation ratio R (must be > 1).

    Returns
    -------
    HalfLifeResult
    """
    if t_half_h <= 0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")
    if target_accumulation_ratio <= 1.0:
        raise ValueError(f"target_accumulation_ratio must be > 1, got {target_accumulation_ratio}")

    R = target_accumulation_ratio
    tau_unclamped = t_half_h * math.log2(R / (R - 1.0))
    tau = max(0.5, min(48.0, tau_unclamped))

    return calculate_half_life(drug_name, t_half_h, dosing_interval_h=tau)


def half_life_from_data(times_h: list | np.ndarray, conc_mg_L: list | np.ndarray) -> float:
    """Estimate elimination half-life from concentration-time data.

    Performs log-linear regression on the terminal 40% of data points and
    returns -0.693 / slope.

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    conc_mg_L:
        Corresponding drug concentrations in mg/L. Values must be > 0.

    Returns
    -------
    float
        Estimated half-life in hours.

    Raises
    ------
    ValueError
        If fewer than 3 data points are provided.
    """
    t = np.asarray(times_h, dtype=float)
    c = np.asarray(conc_mg_L, dtype=float)

    if len(t) < 3:
        raise ValueError(f"At least 3 data points are required, got {len(t)}")

    n_terminal = max(2, int(math.ceil(0.4 * len(t))))
    t_term = t[-n_terminal:]
    c_term = c[-n_terminal:]

    # Filter non-positive concentrations before taking log
    mask = c_term > 0
    t_term = t_term[mask]
    log_c = np.log(c_term[mask])

    if len(t_term) < 2:
        raise ValueError("Insufficient positive concentration values in terminal phase.")

    # Ordinary least-squares slope
    t_mean = np.mean(t_term)
    log_c_mean = np.mean(log_c)
    slope = float(np.sum((t_term - t_mean) * (log_c - log_c_mean)) / np.sum((t_term - t_mean) ** 2))

    if slope >= 0:
        raise ValueError(
            f"Non-negative slope ({slope:.4f}) in terminal phase; cannot estimate half-life."
        )

    return -0.693 / slope
