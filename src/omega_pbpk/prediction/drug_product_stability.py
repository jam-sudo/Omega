"""Drug product stability prediction from degradation kinetics."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Gas constant
# ---------------------------------------------------------------------------

_R = 8.314  # J / (mol * K)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICHStabilityResult:
    """ICH stability prediction results for multiple storage zones."""

    drug_name: str
    k_25C_per_h: float
    ea_kJ_per_mol: float
    zone1_shelf_months: float
    zone2_shelf_months: float
    zone3_shelf_months: float
    accelerated_shelf_months: float
    predicted_long_term_months: float
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def arrhenius_rate(
    k_ref: float,
    ea_kJ_per_mol: float,
    t_ref_C: float,
    t_storage_C: float,
) -> float:
    """Compute degradation rate at storage temperature using Arrhenius equation.

    k_storage = k_ref * exp(-Ea * 1000 / R * (1/T_storage - 1/T_ref))

    where R = 8.314 J/(mol*K) and temperatures are in Kelvin.

    Parameters
    ----------
    k_ref : float
        Reference degradation rate constant (per hour). Must be > 0.
    ea_kJ_per_mol : float
        Activation energy in kJ/mol. Must be > 0.
    t_ref_C : float
        Reference temperature in degrees Celsius.
    t_storage_C : float
        Storage temperature in degrees Celsius.

    Returns
    -------
    float
        Degradation rate constant at storage temperature (per hour).
    """
    if k_ref <= 0:
        raise ValueError("k_ref must be > 0")
    if ea_kJ_per_mol <= 0:
        raise ValueError("ea_kJ_per_mol must be > 0")

    t_ref_k = t_ref_C + 273.15
    t_storage_k = t_storage_C + 273.15

    exponent = -(ea_kJ_per_mol * 1000.0 / _R) * (1.0 / t_storage_k - 1.0 / t_ref_k)
    return k_ref * math.exp(exponent)


def shelf_life_first_order(k_per_h: float, potency_limit_pct: float = 90.0) -> float:
    """Compute shelf life for first-order degradation.

    t_shelf = -ln(potency_limit_pct / 100) / k_per_h

    Parameters
    ----------
    k_per_h : float
        First-order degradation rate constant (per hour). Must be > 0.
    potency_limit_pct : float
        Acceptable potency limit (%). Must be in (0, 100). Default is 90.

    Returns
    -------
    float
        Shelf life in hours.
    """
    if k_per_h <= 0:
        raise ValueError("k_per_h must be > 0")
    if not (0.0 < potency_limit_pct < 100.0):
        raise ValueError("potency_limit_pct must be in (0, 100)")

    return -math.log(potency_limit_pct / 100.0) / k_per_h


def q10_extrapolation(
    t_shelf_ref_months: float,
    t_ref_C: float,
    t_storage_C: float,
    q10: float = 2.0,
) -> float:
    """Extrapolate shelf life to a different temperature using the Q10 rule.

    Rate doubles every 10°C rise (Q10 = 2 by default).

    t_shelf_storage = t_shelf_ref / Q10^(delta_T / 10)

    Parameters
    ----------
    t_shelf_ref_months : float
        Known shelf life at reference temperature (months). Must be > 0.
    t_ref_C : float
        Reference temperature (°C).
    t_storage_C : float
        Target storage temperature (°C).
    q10 : float
        Temperature coefficient (rate doubles every 10°C). Default 2.0.

    Returns
    -------
    float
        Shelf life at storage temperature in months.
    """
    if t_shelf_ref_months <= 0:
        raise ValueError("t_shelf_ref_months must be > 0")
    if q10 <= 0:
        raise ValueError("q10 must be > 0")

    delta_t = t_storage_C - t_ref_C
    factor = q10 ** (delta_t / 10.0)
    return t_shelf_ref_months / factor


def ich_stability_prediction(
    drug_name: str,
    k_25C_per_h: float,
    ea_kJ_per_mol: float,
    humidity_acceleration_factor: float = 1.0,
) -> ICHStabilityResult:
    """Predict ICH stability for multiple climate zones.

    ICH conditions:
      - Zone I:       21°C / 45% RH
      - Zone II:      25°C / 60% RH  (long-term)
      - Zone III:     30°C / 65% RH
      - Zone IV:      30°C / 65% RH
      - Accelerated:  40°C / 75% RH

    Shelf life at each condition = shelf_life_first_order(k_zone, 90%) / 720
    (converted from hours to months, where 1 month = 720 hours).

    The humidity_acceleration_factor multiplicatively increases k for conditions
    with higher RH (Zone III/IV and accelerated).

    Parameters
    ----------
    drug_name : str
        Drug product name.
    k_25C_per_h : float
        Degradation rate constant at 25°C (per hour). Must be > 0.
    ea_kJ_per_mol : float
        Activation energy in kJ/mol. Must be > 0.
    humidity_acceleration_factor : float
        Additional factor applied to k for high-humidity conditions
        (Zone III, Zone IV, Accelerated). Default 1.0 (no extra effect).

    Returns
    -------
    ICHStabilityResult
        Frozen dataclass with shelf life predictions for each ICH zone.
    """
    if k_25C_per_h <= 0:
        raise ValueError("k_25C_per_h must be > 0")
    if ea_kJ_per_mol <= 0:
        raise ValueError("ea_kJ_per_mol must be > 0")
    if humidity_acceleration_factor <= 0:
        raise ValueError("humidity_acceleration_factor must be > 0")

    t_ref_c = 25.0  # reference is 25°C

    def _shelf_months(t_c: float, apply_humidity: bool = False) -> float:
        k = arrhenius_rate(k_25C_per_h, ea_kJ_per_mol, t_ref_c, t_c)
        if apply_humidity:
            k *= humidity_acceleration_factor
        t_h = shelf_life_first_order(k, 90.0)
        return t_h / 720.0

    zone1 = _shelf_months(21.0, apply_humidity=False)
    zone2 = _shelf_months(25.0, apply_humidity=False)
    # Zone III and Zone IV share the same ICH condition: 30°C/65% RH
    zone3 = _shelf_months(30.0, apply_humidity=True)
    accelerated = _shelf_months(40.0, apply_humidity=True)

    # Long-term prediction from Zone II (25°C/60% RH)
    predicted_long_term = zone2

    notes = (
        f"{drug_name}: Ea={ea_kJ_per_mol:.1f} kJ/mol, k_25C={k_25C_per_h:.3e}/h. "
        f"Zone I (21C): {zone1:.1f} months; Zone II (25C): {zone2:.1f} months; "
        f"Zone III/IV (30C): {zone3:.1f} months; "
        f"Accelerated (40C): {accelerated:.1f} months."
    )

    return ICHStabilityResult(
        drug_name=drug_name,
        k_25C_per_h=k_25C_per_h,
        ea_kJ_per_mol=ea_kJ_per_mol,
        zone1_shelf_months=zone1,
        zone2_shelf_months=zone2,
        zone3_shelf_months=zone3,
        accelerated_shelf_months=accelerated,
        predicted_long_term_months=predicted_long_term,
        notes=notes,
    )


def degradation_profile(
    drug_name: str,
    initial_potency_pct: float,
    k_per_h: float,
    t_end_months: float = 36,
    n_points: int = 13,
) -> dict:
    """Compute potency over time for first-order degradation.

    P(t) = initial_potency_pct * exp(-k * t_h)
    where t_h = t_months * 720.

    Parameters
    ----------
    drug_name : str
        Drug product name.
    initial_potency_pct : float
        Initial potency at t=0 (%). Must be in (0, 100].
    k_per_h : float
        First-order degradation rate constant (per hour). Must be > 0.
    t_end_months : float
        End time for profile (months). Default 36.
    n_points : int
        Number of time points (including t=0). Default 13.

    Returns
    -------
    dict with keys:
        drug_name : str
        times_months : list[float]
        potency_pct : list[float]
        shelf_life_months : float  (time when potency first drops below 90%)
    """
    if k_per_h <= 0:
        raise ValueError("k_per_h must be > 0")
    if not (0.0 < initial_potency_pct <= 100.0):
        raise ValueError("initial_potency_pct must be in (0, 100]")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if t_end_months <= 0:
        raise ValueError("t_end_months must be > 0")

    step = t_end_months / (n_points - 1)
    times_months = [i * step for i in range(n_points)]

    potency_pct: list[float] = []
    for t_m in times_months:
        t_h = t_m * 720.0
        p = initial_potency_pct * math.exp(-k_per_h * t_h)
        potency_pct.append(p)

    # Find shelf life: first time potency < 90%
    limit = 90.0
    shelf_life_months = float("inf")
    for i, p in enumerate(potency_pct):
        if p < limit:
            # Linear interpolate between i-1 and i
            if i == 0:
                shelf_life_months = 0.0
            else:
                p_prev = potency_pct[i - 1]
                t_prev = times_months[i - 1]
                t_curr = times_months[i]
                # fraction = (limit - p_prev) / (p - p_prev)
                frac = (limit - p_prev) / (p - p_prev)
                shelf_life_months = t_prev + frac * (t_curr - t_prev)
            break

    # If never drops below limit within range, use analytical value
    if shelf_life_months == float("inf"):
        t_h_shelf = shelf_life_first_order(k_per_h, limit * initial_potency_pct / 100.0)
        shelf_life_months = t_h_shelf / 720.0

    return {
        "drug_name": drug_name,
        "times_months": times_months,
        "potency_pct": potency_pct,
        "shelf_life_months": shelf_life_months,
    }


__all__ = [
    "ICHStabilityResult",
    "arrhenius_rate",
    "shelf_life_first_order",
    "q10_extrapolation",
    "ich_stability_prediction",
    "degradation_profile",
]
