"""Drug stability prediction — Phase 453.

Predicts drug degradation rates under various storage conditions using
Arrhenius kinetics. Supports first-order and zero-order degradation,
accelerated stability studies, and Q10 extrapolation.

Reference: ICH Q1A(R2) Stability Testing of New Drug Substances and Products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "StabilityStudyResult",
    "arrhenius_rate",
    "predict_shelf_life",
    "accelerated_stability_study",
    "q10_extrapolation",
]

# Universal gas constant in kJ/(mol·K)
_R_KJ = 8.314e-3

# Default ICH storage temperatures (°C)
_DEFAULT_TEMPERATURES_C = [25.0, 30.0, 40.0, 50.0, 60.0]


@dataclass
class StabilityStudyResult:
    """Result of an accelerated stability study."""

    k_ref: float
    ea_kJ_mol: float
    temperatures_C: list[float]
    rate_constants: list[float]
    shelf_life_months: list[float]
    q10_factor: float
    notes: str


def arrhenius_rate(
    k_ref: float,
    ea_kJ_mol: float,
    t_ref_C: float = 25.0,
    t_C: float = 40.0,
) -> float:
    """Return rate constant at temperature t_C using the Arrhenius equation.

    k(T) = k_ref * exp(Ea/R * (1/T_ref - 1/T))

    Parameters
    ----------
    k_ref:
        Rate constant at reference temperature (per month).
    ea_kJ_mol:
        Activation energy in kJ/mol. Must be > 0.
    t_ref_C:
        Reference temperature in Celsius (default 25°C).
    t_C:
        Target temperature in Celsius (default 40°C).

    Returns
    -------
    float
        Rate constant at t_C in same units as k_ref.

    Raises
    ------
    ValueError
        If k_ref <= 0 or ea_kJ_mol <= 0.
    """
    if k_ref <= 0:
        raise ValueError(f"k_ref must be > 0, got {k_ref}")
    if ea_kJ_mol <= 0:
        raise ValueError(f"ea_kJ_mol must be > 0, got {ea_kJ_mol}")

    t_ref_K = t_ref_C + 273.15
    t_K = t_C + 273.15
    exponent = (ea_kJ_mol / _R_KJ) * (1.0 / t_ref_K - 1.0 / t_K)
    return k_ref * math.exp(exponent)


def predict_shelf_life(
    k_ref: float,
    ea_kJ_mol: float,
    degradation_limit_pct: float = 5.0,
    t_C: float = 25.0,
    t_ref_C: float = 25.0,
    order: int = 1,
) -> float:
    """Predict shelf life (months) at storage temperature t_C.

    For first-order (order=1):
        t_shelf = ln(100 / (100 - degradation_limit_pct)) / k(T)

    For zero-order (order=0):
        t_shelf = degradation_limit_pct / (k(T) * 100)

    Parameters
    ----------
    k_ref:
        Rate constant at reference temperature t_ref_C (per month).
    ea_kJ_mol:
        Activation energy in kJ/mol.
    degradation_limit_pct:
        Acceptable degradation percentage (default 5% → 95% assay).
    t_C:
        Storage temperature in Celsius (default 25°C).
    t_ref_C:
        Reference temperature for k_ref in Celsius (default 25°C).
    order:
        Reaction order: 1 (first-order) or 0 (zero-order).

    Returns
    -------
    float
        Predicted shelf life in months.

    Raises
    ------
    ValueError
        If inputs are invalid or order is unsupported.
    """
    if not (0 < degradation_limit_pct < 100):
        raise ValueError(
            f"degradation_limit_pct must be in (0, 100), got {degradation_limit_pct}"
        )
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")

    k_at_T = arrhenius_rate(k_ref, ea_kJ_mol, t_ref_C=t_ref_C, t_C=t_C)

    if order == 1:
        # C(t) = C0 * exp(-k*t), shelf life when C drops by degradation_limit_pct
        return math.log(100.0 / (100.0 - degradation_limit_pct)) / k_at_T
    else:
        # Zero-order: C(t) = C0 - k*t (k in %/month)
        # t = degradation_limit_pct / k_at_T
        return degradation_limit_pct / k_at_T


def accelerated_stability_study(
    k_ref: float,
    ea_kJ_mol: float,
    temperatures_C: list[float] | None = None,
    humidity_correction: float = 1.0,
) -> StabilityStudyResult:
    """Simulate an accelerated stability study across multiple temperatures.

    Computes rate constants and predicted shelf lives at each temperature.
    An optional humidity correction factor scales all rate constants (e.g.,
    increased relative humidity accelerates hydrolysis).

    Parameters
    ----------
    k_ref:
        Rate constant at 25°C reference (per month).
    ea_kJ_mol:
        Activation energy in kJ/mol.
    temperatures_C:
        List of storage temperatures to evaluate (°C). Defaults to ICH
        conditions: [25, 30, 40, 50, 60].
    humidity_correction:
        Multiplicative correction for humidity effects (default 1.0 = no
        humidity effect). Must be >= 1.0.

    Returns
    -------
    StabilityStudyResult
        Results for each temperature including rate constants and shelf lives.

    Raises
    ------
    ValueError
        If humidity_correction < 1.0 or other inputs are invalid.
    """
    if humidity_correction < 1.0:
        raise ValueError(
            f"humidity_correction must be >= 1.0, got {humidity_correction}"
        )
    if temperatures_C is None:
        temperatures_C = list(_DEFAULT_TEMPERATURES_C)
    if not temperatures_C:
        raise ValueError("temperatures_C must not be empty")

    rate_constants: list[float] = []
    shelf_life_months: list[float] = []

    for t_C in temperatures_C:
        k_t = arrhenius_rate(k_ref, ea_kJ_mol, t_ref_C=25.0, t_C=t_C)
        k_corrected = k_t * humidity_correction
        rate_constants.append(k_corrected)
        # First-order shelf life for 5% degradation limit
        sl = math.log(100.0 / 95.0) / k_corrected
        shelf_life_months.append(sl)

    # Q10 factor: ratio of rates at (T+10) vs T, estimated at 25°C → 35°C
    k_25 = arrhenius_rate(k_ref, ea_kJ_mol, t_ref_C=25.0, t_C=25.0)
    k_35 = arrhenius_rate(k_ref, ea_kJ_mol, t_ref_C=25.0, t_C=35.0)
    q10_factor = k_35 / k_25

    notes_parts = [
        f"Ea = {ea_kJ_mol:.1f} kJ/mol",
        f"Q10 = {q10_factor:.2f}",
    ]
    if humidity_correction > 1.0:
        notes_parts.append(f"humidity correction = {humidity_correction:.2f}x")

    return StabilityStudyResult(
        k_ref=k_ref,
        ea_kJ_mol=ea_kJ_mol,
        temperatures_C=list(temperatures_C),
        rate_constants=rate_constants,
        shelf_life_months=shelf_life_months,
        q10_factor=q10_factor,
        notes="; ".join(notes_parts),
    )


def q10_extrapolation(
    t_half_25C_months: float,
    q10: float = 2.0,
    target_temp_C: float = 40.0,
) -> float:
    """Extrapolate shelf life to a different temperature using Q10 rule.

    The Q10 rule states that for every 10°C increase in temperature, the
    reaction rate increases by a factor of Q10 (commonly 2).

    t_shelf(T) = t_shelf(25°C) / Q10^((T - 25) / 10)

    Parameters
    ----------
    t_half_25C_months:
        Shelf life (half-life) at 25°C in months.
    q10:
        Q10 factor (rate multiplier per 10°C increase). Default 2.0.
    target_temp_C:
        Target storage temperature in Celsius.

    Returns
    -------
    float
        Predicted shelf life in months at target_temp_C.

    Raises
    ------
    ValueError
        If t_half_25C_months <= 0 or q10 <= 0.
    """
    if t_half_25C_months <= 0:
        raise ValueError(
            f"t_half_25C_months must be > 0, got {t_half_25C_months}"
        )
    if q10 <= 0:
        raise ValueError(f"q10 must be > 0, got {q10}")

    delta_T = target_temp_C - 25.0
    acceleration = q10 ** (delta_T / 10.0)
    return t_half_25C_months / acceleration
