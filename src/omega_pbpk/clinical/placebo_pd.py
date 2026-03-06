"""Placebo effect pharmacodynamic model.

Models the placebo effect in clinical trials using time-varying placebo response
models (Sheiner 1995). Combines a sigmoid Emax drug effect with a decaying or
constant placebo response to give a total observed response.

Models supported:
  - "none"     : no placebo effect (e_placebo = 0)
  - "constant" : flat placebo response P(t) = P0
  - "decay"    : decaying placebo P(t) = P0 * exp(-kp * t)  [Sheiner 1995]

Drug effect uses the sigmoid Emax model:
  E_drug(t) = Emax * C(t)^n / (EC50^n + C(t)^n)

Total observed response:
  E_total(t) = baseline + E_drug(t) + E_placebo(t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

_VALID_PLACEBO_MODELS = ("none", "constant", "decay")


@dataclass(frozen=True)
class PlaceboResult:
    """Result of a placebo PD simulation."""

    drug_name: str
    dose_mg: float
    placebo_model: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    e_drug: list[float]
    e_placebo: list[float]
    e_total: list[float]
    peak_drug_effect: float
    peak_placebo_effect: float
    peak_total_effect: float
    auc_drug_effect: float
    auc_placebo_effect: float
    auc_total_effect: float
    treatment_difference: float
    notes: list[str] = field(default_factory=list)


def simulate_placebo_pd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    emax: float,
    ec50_mg_L: float,
    hill_n: float = 1.0,
    placebo_model: str = "decay",
    p0: float = 20.0,
    kp_per_h: float = 0.02,
    baseline: float = 0.0,
    route: str = "oral",
    ka_per_h: float = 1.0,
    t_end_h: float = 168.0,
    dt_h: float = 1.0,
) -> PlaceboResult:
    """Simulate drug PK/PD combined with a placebo response model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    emax:
        Maximum drug effect (same units as the response endpoint).
    ec50_mg_L:
        Drug concentration at half-maximal effect (mg/L).
    hill_n:
        Hill coefficient (sigmoidicity).
    placebo_model:
        One of "none", "constant", or "decay".
    p0:
        Initial placebo response amplitude (same units as emax).
    kp_per_h:
        Rate constant for placebo decay (h^-1); only used for "decay" model.
    baseline:
        Baseline (pre-treatment) response added to all time points.
    route:
        "oral" (1-compartment first-order) or "iv" (bolus).
    ka_per_h:
        First-order oral absorption rate constant (h^-1); used when route="oral".
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    PlaceboResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if emax <= 0:
        raise ValueError("emax must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if hill_n <= 0:
        raise ValueError("hill_n must be > 0")
    if p0 < 0:
        raise ValueError("p0 must be >= 0")
    if placebo_model not in _VALID_PLACEBO_MODELS:
        raise ValueError(f"placebo_model must be one of {_VALID_PLACEBO_MODELS}")
    if route not in ("oral", "iv"):
        raise ValueError("route must be 'oral' or 'iv'")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    k_el = cl_L_per_h / vd_L

    # --- Plasma concentration profile (analytical) ---
    if route == "iv":
        c_plasma = (dose_mg / vd_L) * np.exp(-k_el * times)
    else:
        # 1-compartment oral
        if abs(ka_per_h - k_el) < 1e-9:
            ka_per_h = ka_per_h + 1e-6
        c_plasma = (
            dose_mg
            * ka_per_h
            / (vd_L * (ka_per_h - k_el))
            * (np.exp(-k_el * times) - np.exp(-ka_per_h * times))
        )
        c_plasma = np.maximum(c_plasma, 0.0)

    # --- Drug effect: sigmoid Emax ---
    cn = np.power(np.maximum(c_plasma, 0.0), hill_n)
    ec50n = ec50_mg_L**hill_n
    e_drug = emax * cn / (ec50n + cn)

    # --- Placebo effect ---
    if placebo_model == "none":
        e_placebo = np.zeros_like(times)
    elif placebo_model == "constant":
        e_placebo = np.full_like(times, p0)
        # At t=0 (pre-dose) placebo is already active
    elif placebo_model == "decay":
        e_placebo = p0 * np.exp(-kp_per_h * times)
    else:
        e_placebo = np.zeros_like(times)

    # --- Total response ---
    e_total = baseline + e_drug + e_placebo

    # --- Derived metrics ---
    peak_drug_effect = float(np.max(e_drug))
    peak_placebo_effect = float(np.max(e_placebo))
    peak_total_effect = float(np.max(e_total))

    auc_drug_effect = float(np_trapz(e_drug, times))
    auc_placebo_effect = float(np_trapz(e_placebo, times))
    auc_total_effect = float(np_trapz(e_total, times))

    denom = peak_drug_effect + peak_placebo_effect
    treatment_difference = peak_drug_effect / denom if denom > 0.0 else 0.0

    notes: list[str] = []
    if placebo_model == "none":
        notes.append("No placebo model applied; e_placebo is zero throughout")
    if peak_placebo_effect > peak_drug_effect:
        notes.append("Peak placebo effect exceeds peak drug effect")

    return PlaceboResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        placebo_model=placebo_model,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        e_drug=e_drug.tolist(),
        e_placebo=e_placebo.tolist(),
        e_total=e_total.tolist(),
        peak_drug_effect=peak_drug_effect,
        peak_placebo_effect=peak_placebo_effect,
        peak_total_effect=peak_total_effect,
        auc_drug_effect=auc_drug_effect,
        auc_placebo_effect=auc_placebo_effect,
        auc_total_effect=auc_total_effect,
        treatment_difference=treatment_difference,
        notes=notes,
    )


def estimate_drug_effect(placebo_result: PlaceboResult) -> dict:
    """Estimate the drug-specific effect from a PlaceboResult.

    Returns a summary dict separating the drug contribution from the
    placebo contribution at the population level.

    Parameters
    ----------
    placebo_result:
        The result of a ``simulate_placebo_pd`` call.

    Returns
    -------
    dict with keys:
        - max_drug_only_effect: float
        - max_placebo_effect: float
        - drug_vs_placebo_ratio: float
        - treatment_difference: float (fraction of total peak from drug)
    """
    max_drug = placebo_result.peak_drug_effect
    max_placebo = placebo_result.peak_placebo_effect

    drug_vs_placebo_ratio = max_drug / max_placebo if max_placebo > 0.0 else float("inf")

    denom = max_drug + max_placebo
    treatment_difference = max_drug / denom if denom > 0.0 else 0.0

    return {
        "max_drug_only_effect": max_drug,
        "max_placebo_effect": max_placebo,
        "drug_vs_placebo_ratio": drug_vs_placebo_ratio,
        "treatment_difference": treatment_difference,
    }


__all__ = [
    "PlaceboResult",
    "simulate_placebo_pd",
    "estimate_drug_effect",
]
