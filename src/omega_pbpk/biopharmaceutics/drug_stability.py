"""Drug stability in formulation: hydrolysis, oxidation, photodegradation.

Models chemical degradation using Arrhenius kinetics for shelf-life prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DegradationResult",
    "AcceleratedStabilityResult",
    "arrhenius_rate",
    "simulate_degradation",
    "shelf_life",
    "accelerated_stability",
]

# Universal gas constant in kJ/mol/K
_R_KJ = 8.314e-3


@dataclass(frozen=True)
class DegradationResult:
    drug_name: str
    k_per_day: float
    model: str
    times_days: list[float]
    pct_remaining: list[float]
    t90_days: float  # time to 90% remaining
    shelf_life_days: float


@dataclass(frozen=True)
class AcceleratedStabilityResult:
    drug_name: str
    k_40C_per_day: float
    ea_kJ_per_mol: float
    k_25C_per_day: float
    shelf_life_25C_days: float
    shelf_life_40C_days: float
    acceleration_factor: float
    notes: str


def arrhenius_rate(
    k_ref: float,
    ea_kJ_per_mol: float,
    t_ref_C: float,
    t_storage_C: float,
) -> float:
    """Return temperature-adjusted rate constant using Arrhenius equation.

    k(T) = k_ref * exp(Ea/R * (1/T_ref - 1/T))

    Parameters
    ----------
    k_ref:
        Rate constant at reference temperature (same units as returned value).
    ea_kJ_per_mol:
        Activation energy in kJ/mol.
    t_ref_C:
        Reference temperature in Celsius.
    t_storage_C:
        Storage temperature in Celsius.

    Returns
    -------
    float
        Rate constant at storage temperature.
    """
    t_ref_k = t_ref_C + 273.15
    t_storage_k = t_storage_C + 273.15
    exponent = (ea_kJ_per_mol / _R_KJ) * (1.0 / t_ref_k - 1.0 / t_storage_k)
    return k_ref * math.exp(exponent)


def shelf_life(
    k_per_day: float,
    target_pct: float = 90.0,
    model: str = "first_order",
) -> float:
    """Return time (days) to reach target_pct remaining.

    Parameters
    ----------
    k_per_day:
        Degradation rate constant (per day).
    target_pct:
        Target percentage remaining (default 90%).
    model:
        Degradation model: "first_order" or "zero_order".

    Returns
    -------
    float
        Shelf life in days.

    Raises
    ------
    ValueError
        If inputs are invalid or model is unsupported.
    """
    if k_per_day <= 0:
        raise ValueError(f"k_per_day must be > 0, got {k_per_day}")
    if not (0 < target_pct < 100):
        raise ValueError(f"target_pct must be in (0, 100), got {target_pct}")

    if model == "first_order":
        # C(t) = 100 * exp(-k*t) => t = -ln(target_pct/100) / k
        return -math.log(target_pct / 100.0) / k_per_day
    elif model == "zero_order":
        # C(t) = 100 - k*100*t => t = (100 - target_pct) / (k * 100)
        return (100.0 - target_pct) / (k_per_day * 100.0)
    elif model == "autocatalytic":
        # Numerical approach for autocatalytic model
        # Approximate with first-order as conservative estimate
        return -math.log(target_pct / 100.0) / k_per_day
    else:
        raise ValueError(
            f"Unknown model: {model}. Use 'first_order', 'zero_order', or 'autocatalytic'."
        )


def simulate_degradation(
    drug_name: str,
    initial_pct: float = 100.0,
    k_per_day: float = 0.01,
    degradation_model: str = "first_order",
    t_end_days: float = 365.0,
    dt_days: float = 1.0,
) -> DegradationResult:
    """Simulate drug degradation over time in formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    initial_pct:
        Initial drug content as percentage (default 100%).
    k_per_day:
        Degradation rate constant per day.
    degradation_model:
        Kinetic model: "first_order", "zero_order", or "autocatalytic".
    t_end_days:
        Simulation end time in days.
    dt_days:
        Time step in days.

    Returns
    -------
    DegradationResult
        Time-course of drug percentage remaining.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not (0 < initial_pct <= 100):
        raise ValueError(f"initial_pct must be in (0, 100], got {initial_pct}")
    if k_per_day <= 0:
        raise ValueError(f"k_per_day must be > 0, got {k_per_day}")
    if t_end_days <= 0:
        raise ValueError(f"t_end_days must be > 0, got {t_end_days}")
    if dt_days <= 0:
        raise ValueError(f"dt_days must be > 0, got {dt_days}")
    valid_models = ("first_order", "zero_order", "autocatalytic")
    if degradation_model not in valid_models:
        raise ValueError(
            f"degradation_model must be one of {valid_models}, got {degradation_model!r}"
        )

    n_steps = int(t_end_days / dt_days) + 1
    times = [i * dt_days for i in range(n_steps)]

    pct = [0.0] * n_steps
    pct[0] = initial_pct

    for i in range(1, n_steps):
        c = pct[i - 1]
        if degradation_model == "first_order":
            pct[i] = max(0.0, c * math.exp(-k_per_day * dt_days))
        elif degradation_model == "zero_order":
            pct[i] = max(0.0, c - k_per_day * 100.0 * dt_days)
        elif degradation_model == "autocatalytic":
            # dC/dt = -k * C * (1 - C/Cmax), Cmax = initial_pct
            c_max = initial_pct
            dc_dt = -k_per_day * c * (1.0 - c / c_max)
            pct[i] = max(0.0, c + dc_dt * dt_days)

    # Compute t90 (time to 90% remaining)
    target = 0.9 * initial_pct
    t90 = t_end_days  # default to end if not reached
    for i, p in enumerate(pct):
        if p <= target:
            if i > 0:
                # Linear interpolation
                frac = (pct[i - 1] - target) / (pct[i - 1] - p)
                t90 = times[i - 1] + frac * dt_days
            else:
                t90 = times[i]
            break

    sl = shelf_life(k_per_day, target_pct=90.0, model=degradation_model)

    return DegradationResult(
        drug_name=drug_name,
        k_per_day=k_per_day,
        model=degradation_model,
        times_days=times,
        pct_remaining=pct,
        t90_days=t90,
        shelf_life_days=sl,
    )


def accelerated_stability(
    drug_name: str,
    k_40C_per_day: float,
    ea_kJ_per_mol: float,
    target_pct: float = 90.0,
) -> AcceleratedStabilityResult:
    """Predict shelf-life at 25 deg C from 40 deg C accelerated stability data.

    Uses Arrhenius equation to extrapolate from accelerated (40 deg C) to
    real-world (25 deg C) storage conditions (ICH Q1A guidelines).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    k_40C_per_day:
        Degradation rate constant at 40 deg C (per day).
    ea_kJ_per_mol:
        Activation energy in kJ/mol.
    target_pct:
        Target percentage remaining for shelf-life calculation (default 90%).

    Returns
    -------
    AcceleratedStabilityResult
        Shelf-life prediction at 25 deg C and 40 deg C.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if k_40C_per_day <= 0:
        raise ValueError(f"k_40C_per_day must be > 0, got {k_40C_per_day}")
    if ea_kJ_per_mol <= 0:
        raise ValueError(f"ea_kJ_per_mol must be > 0, got {ea_kJ_per_mol}")
    if not (0 < target_pct < 100):
        raise ValueError(f"target_pct must be in (0, 100), got {target_pct}")

    # Convert 40 deg C rate to 25 deg C using Arrhenius
    # k_25C = k_40C * exp(Ea/R * (1/T_40 - 1/T_25))
    k_25c = arrhenius_rate(k_40C_per_day, ea_kJ_per_mol, t_ref_C=40.0, t_storage_C=25.0)

    sl_25c = shelf_life(k_25c, target_pct=target_pct, model="first_order")
    sl_40c = shelf_life(k_40C_per_day, target_pct=target_pct, model="first_order")

    acceleration_factor = k_40C_per_day / k_25c if k_25c > 0 else float("inf")

    notes = (
        f"ICH Q1A accelerated stability: 40 deg C/75% RH → 25 deg C/60% RH extrapolation. "
        f"Ea = {ea_kJ_per_mol:.1f} kJ/mol. "
        f"Acceleration factor = {acceleration_factor:.2f}x."
    )

    return AcceleratedStabilityResult(
        drug_name=drug_name,
        k_40C_per_day=k_40C_per_day,
        ea_kJ_per_mol=ea_kJ_per_mol,
        k_25C_per_day=k_25c,
        shelf_life_25C_days=sl_25c,
        shelf_life_40C_days=sl_40c,
        acceleration_factor=acceleration_factor,
        notes=notes,
    )
