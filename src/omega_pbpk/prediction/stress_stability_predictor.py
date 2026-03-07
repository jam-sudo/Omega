"""Phase 279 — Drug Stability Under Stress Conditions Predictor.

Predict drug degradation rate and shelf life under ICH Q1 stress conditions:
thermal, humidity, photolytic, acid/base hydrolysis, oxidative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Gas constant in kJ/(mol·K)
_R_KJ = 8.314e-3

_VALID_CONDITIONS = frozenset(
    [
        "thermal",
        "humidity",
        "photolytic",
        "acid_hydrolysis",
        "base_hydrolysis",
        "oxidative",
    ]
)

_CONDITION_MULTIPLIERS = {
    "thermal": 1.0,
    "humidity": 1.0,  # handled separately with humidity_pct
    "photolytic": 2.0,
    "acid_hydrolysis": 1.5,
    "base_hydrolysis": 1.5,
    "oxidative": 1.2,
}

_STABILITY_NOTES = {
    "thermal": "Arrhenius temperature-dependent degradation",
    "humidity": "Humidity-accelerated degradation; ICH Q1A condition",
    "photolytic": "ICH Q1B photostability; light-accelerated degradation",
    "acid_hydrolysis": "Acid hydrolysis under pH-stress conditions",
    "base_hydrolysis": "Base hydrolysis under alkaline stress conditions",
    "oxidative": "Oxidative degradation; peroxide stress conditions",
}


@dataclass(frozen=True)
class StressStabilityResult:
    """Result of stress stability prediction."""

    drug_name: str
    condition: str
    temperature_C: float
    humidity_pct: float
    initial_purity_pct: float
    k_deg_eff_per_day: float
    purity_at_30d: float
    purity_at_90d: float
    purity_at_180d: float
    purity_at_365d: float
    shelf_life_days: float
    stability_class: str
    notes: str


def _clamp_purity(value: float) -> float:
    """Clamp purity to [0, 100]."""
    return max(0.0, min(100.0, value))


def _classify_stability(shelf_life_days: float) -> str:
    """Return stability_class based on shelf life."""
    if shelf_life_days > 365.0:
        return "stable"
    elif shelf_life_days >= 90.0:
        return "moderately_stable"
    else:
        return "unstable"


def predict_stress_stability(
    drug_name: str,
    condition: str,
    temperature_C: float,
    humidity_pct: float,
    initial_purity_pct: float,
    k_deg_ref_per_day: float,
    ea_kJ_per_mol: float = 80.0,
) -> StressStabilityResult:
    """Predict drug stability under ICH Q1 stress conditions.

    Args:
        drug_name: Name of the drug.
        condition: One of "thermal", "humidity", "photolytic",
            "acid_hydrolysis", "base_hydrolysis", "oxidative".
        temperature_C: Storage temperature in degrees Celsius.
        humidity_pct: Relative humidity in percent [0, 100].
        initial_purity_pct: Initial drug purity in percent (0, 100].
        k_deg_ref_per_day: Reference first-order degradation rate at 25 C (per day).
        ea_kJ_per_mol: Activation energy in kJ/mol (default 80.0).

    Returns:
        StressStabilityResult dataclass with purity predictions and shelf life.

    Raises:
        ValueError: On invalid parameters.
    """
    # Validation
    if condition not in _VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(_VALID_CONDITIONS)}, got {condition!r}")
    if not (-80.0 < temperature_C < 200.0):
        raise ValueError(f"temperature_C must be in (-80, 200), got {temperature_C}")
    if not (0.0 <= humidity_pct <= 100.0):
        raise ValueError(f"humidity_pct must be in [0, 100], got {humidity_pct}")
    if not (0.0 < initial_purity_pct <= 100.0):
        raise ValueError(f"initial_purity_pct must be in (0, 100], got {initial_purity_pct}")
    if k_deg_ref_per_day <= 0.0:
        raise ValueError(f"k_deg_ref_per_day must be > 0, got {k_deg_ref_per_day}")
    if ea_kJ_per_mol <= 0.0:
        raise ValueError(f"ea_kJ_per_mol must be > 0, got {ea_kJ_per_mol}")

    # Arrhenius correction: k_deg at the given temperature
    t_ref_K = 298.15  # 25 C in Kelvin
    t_K = temperature_C + 273.15
    k_deg = k_deg_ref_per_day * math.exp(ea_kJ_per_mol / _R_KJ * (1.0 / t_ref_K - 1.0 / t_K))

    # Condition-specific effective rate
    multiplier = _CONDITION_MULTIPLIERS[condition]
    if condition == "humidity":
        k_eff = k_deg * (1.0 + 0.01 * humidity_pct)
    else:
        k_eff = k_deg * multiplier

    # Purity at various time points: purity(t) = initial * exp(-k_eff * t)
    def purity_at(t_days: float) -> float:
        return _clamp_purity(initial_purity_pct * math.exp(-k_eff * t_days))

    purity_30 = purity_at(30.0)
    purity_90 = purity_at(90.0)
    purity_180 = purity_at(180.0)
    purity_365 = purity_at(365.0)

    # Shelf life: time to 90% of initial purity
    # initial * exp(-k_eff * t) = 0.9 * initial  => t = -ln(0.9) / k_eff
    if k_eff > 0.0:
        shelf_life_days = -math.log(0.9) / k_eff
    else:
        shelf_life_days = float("inf")

    stability_class = _classify_stability(shelf_life_days)

    notes = _STABILITY_NOTES[condition]

    return StressStabilityResult(
        drug_name=drug_name,
        condition=condition,
        temperature_C=temperature_C,
        humidity_pct=humidity_pct,
        initial_purity_pct=initial_purity_pct,
        k_deg_eff_per_day=k_eff,
        purity_at_30d=purity_30,
        purity_at_90d=purity_90,
        purity_at_180d=purity_180,
        purity_at_365d=purity_365,
        shelf_life_days=shelf_life_days,
        stability_class=stability_class,
        notes=notes,
    )


def compare_conditions(
    drug_name: str,
    conditions_list: list[str],
    temperature_C: float,
    humidity_pct: float,
    initial_purity_pct: float,
    k_deg_ref_per_day: float,
    ea_kJ_per_mol: float = 80.0,
) -> list[StressStabilityResult]:
    """Compare drug stability across multiple stress conditions.

    Args:
        drug_name: Name of the drug.
        conditions_list: List of condition names to compare.
        temperature_C: Storage temperature in degrees Celsius.
        humidity_pct: Relative humidity in percent [0, 100].
        initial_purity_pct: Initial drug purity in percent (0, 100].
        k_deg_ref_per_day: Reference first-order degradation rate at 25 C (per day).
        ea_kJ_per_mol: Activation energy in kJ/mol (default 80.0).

    Returns:
        List of StressStabilityResult sorted by purity_at_30d ascending (worst first).
    """
    results = [
        predict_stress_stability(
            drug_name=drug_name,
            condition=cond,
            temperature_C=temperature_C,
            humidity_pct=humidity_pct,
            initial_purity_pct=initial_purity_pct,
            k_deg_ref_per_day=k_deg_ref_per_day,
            ea_kJ_per_mol=ea_kJ_per_mol,
        )
        for cond in conditions_list
    ]
    return sorted(results, key=lambda r: r.purity_at_30d)
