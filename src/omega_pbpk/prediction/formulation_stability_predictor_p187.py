"""Formulation stability prediction using Arrhenius kinetics — Phase 187."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "StabilityResult",
    "predict_stability",
    "compare_storage_conditions",
]

# Gas constant in kJ/(mol·K)
_R_KJ_MOL_K = 8.314e-3

# ICH storage conditions: name → temperature_C
_STORAGE_CONDITIONS: dict[str, float] = {
    "25C_60RH": 25.0,
    "30C_65RH": 30.0,
    "40C_75RH": 40.0,
    "5C": 5.0,
}

_VALID_CONDITIONS = set(_STORAGE_CONDITIONS.keys())


@dataclass(frozen=True)
class StabilityResult:
    """Result of Arrhenius-based formulation stability prediction."""

    drug_name: str
    storage_condition: str
    temperature_C: float
    k_per_day: float
    pct_remaining_6m: float
    pct_remaining_12m: float
    pct_remaining_24m: float
    pct_remaining_36m: float
    shelf_life_months: float
    stability_class: str
    notes: str


def _arrhenius_k(
    k_ref_per_day: float,
    activation_energy_kJ_mol: float,
    t_ref_C: float,
    t_storage_C: float,
) -> float:
    """Compute degradation rate at storage temperature using Arrhenius equation.

    k(T) = k_ref * exp(Ea/R * (1/T_ref - 1/T))
    """
    t_ref_k = t_ref_C + 273.15
    t_k = t_storage_C + 273.15
    exponent = (activation_energy_kJ_mol / _R_KJ_MOL_K) * (1.0 / t_ref_k - 1.0 / t_k)
    return k_ref_per_day * math.exp(exponent)


def _pct_remaining(k_per_day: float, months: float) -> float:
    """Compute % drug remaining after `months` months (first-order decay).

    C(t) = 100 * exp(-k * t), where t is in days.
    """
    days = months * 30.4375  # average days per month
    value = 100.0 * math.exp(-k_per_day * days)
    return max(0.0, min(100.0, value))


def _shelf_life_months(k_per_day: float) -> float:
    """Compute shelf life in months (time to 90% remaining).

    90% remaining: 100*exp(-k*t) = 90 → t = -ln(0.9)/k
    """
    if k_per_day <= 0.0:
        return float("inf")
    days = -math.log(0.9) / k_per_day
    months = days / 30.4375
    return months


def _stability_class(pct_remaining_24m: float) -> str:
    """Classify stability based on % remaining at 24 months."""
    if pct_remaining_24m > 95.0:
        return "stable"
    elif pct_remaining_24m >= 90.0:
        return "acceptable"
    else:
        return "unstable"


def predict_stability(
    drug_name: str,
    activation_energy_kJ_mol: float,
    k_ref_per_day: float,
    t_ref_C: float,
    storage_condition: str,
) -> StabilityResult:
    """Predict long-term stability of a solid oral dosage form.

    Uses Arrhenius kinetics to extrapolate from reference degradation rate
    to the target ICH storage condition.

    Args:
        drug_name: Name of the drug or formulation.
        activation_energy_kJ_mol: Activation energy (Ea) in kJ/mol. Must be > 0.
        k_ref_per_day: First-order degradation rate constant at t_ref_C (1/day). Must be > 0.
        t_ref_C: Reference temperature in °C (e.g. 25.0).
        storage_condition: ICH storage condition string. One of:
            "25C_60RH" (ICH Zone II), "30C_65RH" (Zone IVa),
            "40C_75RH" (accelerated), "5C" (refrigerated).

    Returns:
        StabilityResult with Arrhenius-derived predictions.

    Raises:
        ValueError: If activation_energy_kJ_mol <= 0, k_ref_per_day <= 0, or
                    storage_condition is not recognized.
    """
    if activation_energy_kJ_mol <= 0.0:
        raise ValueError(f"activation_energy_kJ_mol must be > 0, got {activation_energy_kJ_mol}")
    if k_ref_per_day <= 0.0:
        raise ValueError(f"k_ref_per_day must be > 0, got {k_ref_per_day}")
    if storage_condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"Invalid storage_condition '{storage_condition}'. "
            f"Must be one of: {sorted(_VALID_CONDITIONS)}"
        )

    temperature_c = _STORAGE_CONDITIONS[storage_condition]
    k = _arrhenius_k(k_ref_per_day, activation_energy_kJ_mol, t_ref_C, temperature_c)

    pct_6m = _pct_remaining(k, 6.0)
    pct_12m = _pct_remaining(k, 12.0)
    pct_24m = _pct_remaining(k, 24.0)
    pct_36m = _pct_remaining(k, 36.0)

    sl = _shelf_life_months(k)
    stab_class = _stability_class(pct_24m)

    notes = (
        f"Ea={activation_energy_kJ_mol:.1f} kJ/mol; "
        f"k_ref={k_ref_per_day:.4e}/day at {t_ref_C}°C; "
        f"k_storage={k:.4e}/day at {temperature_c}°C; "
        f"shelf_life={sl:.1f} months (to 90%)"
    )

    return StabilityResult(
        drug_name=drug_name,
        storage_condition=storage_condition,
        temperature_C=temperature_c,
        k_per_day=k,
        pct_remaining_6m=pct_6m,
        pct_remaining_12m=pct_12m,
        pct_remaining_24m=pct_24m,
        pct_remaining_36m=pct_36m,
        shelf_life_months=sl,
        stability_class=stab_class,
        notes=notes,
    )


def compare_storage_conditions(
    drug_name: str,
    activation_energy_kJ_mol: float,
    k_ref_per_day: float,
    t_ref_C: float,
) -> list[StabilityResult]:
    """Compare stability across all four ICH storage conditions.

    Args:
        drug_name: Name of the drug or formulation.
        activation_energy_kJ_mol: Activation energy (Ea) in kJ/mol.
        k_ref_per_day: Degradation rate constant at t_ref_C (1/day).
        t_ref_C: Reference temperature in °C.

    Returns:
        List of StabilityResult for all 4 conditions, sorted by
        shelf_life_months descending (longest shelf life first).
    """
    results = [
        predict_stability(drug_name, activation_energy_kJ_mol, k_ref_per_day, t_ref_C, cond)
        for cond in _VALID_CONDITIONS
    ]
    results.sort(key=lambda r: r.shelf_life_months, reverse=True)
    return results
