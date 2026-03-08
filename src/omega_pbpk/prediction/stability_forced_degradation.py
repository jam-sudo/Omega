"""Phase 1072 — Stability Forced Degradation Kinetics.

ICH Q1A-aligned forced degradation kinetics for drug stability prediction.
Stress conditions: acid hydrolysis, base hydrolysis, oxidation, photolysis, thermal.

First-order degradation kinetics:
    A(t) = A0 * exp(-k * t)
    purity_pct(t) = 100 * A(t) / A0 = 100 * exp(-k * t)

Rate constants depend on molecular lability/sensitivity and temperature (thermal only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Valid stress conditions
# ---------------------------------------------------------------------------
VALID_CONDITIONS = frozenset({"acid", "base", "oxidation", "photolysis", "thermal"})

# ---------------------------------------------------------------------------
# Base rate constants (1/h) — ICH Q1A inspired defaults
# ---------------------------------------------------------------------------
K_BASE_ACID: float = 0.001
K_BASE_BASE: float = 0.0005
K_BASE_OX: float = 0.002
K_BASE_PHOTO: float = 0.003
K_BASE_THERMAL: float = 0.0002

# Gas constant in kJ/(mol·K)
R_KJ_MOL_K: float = 8.314e-3


@dataclass(frozen=True)
class StabilityDegradationResult:
    """Results from a single stress degradation prediction."""

    drug_name: str
    stress_condition: str
    k_degradation_per_h: float
    purity_at_1h: float
    purity_at_24h: float
    purity_at_48h: float
    t50_h: float
    t90_h: float
    stability_rating: str  # "stable", "moderate", or "labile"
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purity(k: float, t: float) -> float:
    """Return purity (%) at time t given rate constant k."""
    return 100.0 * math.exp(-k * t)


def _t_to_purity(k: float, target_pct: float) -> float:
    """Return time (h) to reach *target_pct* purity from 100%."""
    # purity = 100*exp(-k*t)  =>  t = -ln(target_pct/100) / k
    if k <= 0.0:
        return float("inf")
    return -math.log(target_pct / 100.0) / k


def _stability_rating(t90_h: float) -> str:
    """Classify stability from t90 (time to 90% purity).

    "stable"   : t90 > 168 h (1 week)
    "moderate" : 24 h < t90 <= 168 h
    "labile"   : t90 <= 24 h
    """
    if t90_h > 168.0:
        return "stable"
    if t90_h > 24.0:
        return "moderate"
    return "labile"


def _validate_lability(name: str, value: float) -> None:
    if not (0.0 <= value <= 10.0):
        raise ValueError(f"{name} must be in [0, 10], got {value}")


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_stress_degradation(
    drug_name: str,
    stress_condition: str,
    acid_lability: float = 0.0,
    base_lability: float = 0.0,
    oxidation_sensitivity: float = 0.0,
    photo_sensitivity: float = 0.0,
    temperature_C: float = 25.0,
    activation_energy_kJ_mol: float = 80.0,
) -> StabilityDegradationResult:
    """Predict first-order degradation kinetics under a single stress condition.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    stress_condition:
        One of "acid", "base", "oxidation", "photolysis", "thermal".
    acid_lability:
        Acid susceptibility score (0–10).
    base_lability:
        Base susceptibility score (0–10).
    oxidation_sensitivity:
        Oxidation susceptibility score (0–10).
    photo_sensitivity:
        Photolysis susceptibility score (0–10).
    temperature_C:
        Temperature in °C (used only for "thermal" condition).
    activation_energy_kJ_mol:
        Arrhenius activation energy in kJ/mol (used for "thermal").
    """
    # --- Validation ---
    if stress_condition not in VALID_CONDITIONS:
        raise ValueError(
            f"stress_condition must be one of {sorted(VALID_CONDITIONS)}, got '{stress_condition}'"
        )
    _validate_lability("acid_lability", acid_lability)
    _validate_lability("base_lability", base_lability)
    _validate_lability("oxidation_sensitivity", oxidation_sensitivity)
    _validate_lability("photo_sensitivity", photo_sensitivity)
    if not (0.0 <= temperature_C <= 100.0):
        raise ValueError(f"temperature_C must be in [0, 100], got {temperature_C}")

    # --- Compute rate constant ---
    if stress_condition == "acid":
        k = K_BASE_ACID * (1.0 + 0.5 * acid_lability)
    elif stress_condition == "base":
        k = K_BASE_BASE * (1.0 + 0.5 * base_lability)
    elif stress_condition == "oxidation":
        k = K_BASE_OX * (1.0 + 0.3 * oxidation_sensitivity)
    elif stress_condition == "photolysis":
        k = K_BASE_PHOTO * (1.0 + 0.2 * photo_sensitivity)
    else:  # thermal
        T_K = temperature_C + 273.15
        T_ref_K = 298.15  # 25 °C
        k = K_BASE_THERMAL * math.exp(
            (activation_energy_kJ_mol / R_KJ_MOL_K) * (1.0 / T_ref_K - 1.0 / T_K)
        )

    # --- Purity at key time points ---
    p1h = _purity(k, 1.0)
    p24h = _purity(k, 24.0)
    p48h = _purity(k, 48.0)

    # --- Half-lives ---
    # t50_h: time to 50% purity (drug half-life under stress)
    t50_h = _t_to_purity(k, 50.0)
    # t90_h: time to 90% purity (shelf-life indicator) — always < t50_h
    t90_h = _t_to_purity(k, 90.0)

    rating = _stability_rating(t90_h)

    notes = (
        f"stress={stress_condition}; "
        f"k={k:.6f} /h; "
        f"t90={t90_h:.1f} h, t50={t50_h:.1f} h; "
        f"stability={rating}"
    )

    return StabilityDegradationResult(
        drug_name=drug_name,
        stress_condition=stress_condition,
        k_degradation_per_h=k,
        purity_at_1h=p1h,
        purity_at_24h=p24h,
        purity_at_48h=p48h,
        t50_h=t50_h,
        t90_h=t90_h,
        stability_rating=rating,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screen all 5 stress conditions
# ---------------------------------------------------------------------------


def screen_stress_conditions(
    drug_name: str,
    acid_lability: float = 0.0,
    base_lability: float = 0.0,
    oxidation_sensitivity: float = 0.0,
    photo_sensitivity: float = 0.0,
    temperature_C: float = 25.0,
    activation_energy_kJ_mol: float = 80.0,
) -> list:
    """Run all 5 stress conditions and return sorted by k_degradation_per_h descending.

    Returns
    -------
    list[StabilityDegradationResult] with 5 entries, sorted by k descending.
    """
    results = []
    for condition in VALID_CONDITIONS:
        r = predict_stress_degradation(
            drug_name=drug_name,
            stress_condition=condition,
            acid_lability=acid_lability,
            base_lability=base_lability,
            oxidation_sensitivity=oxidation_sensitivity,
            photo_sensitivity=photo_sensitivity,
            temperature_C=temperature_C,
            activation_energy_kJ_mol=activation_energy_kJ_mol,
        )
        results.append(r)
    results.sort(key=lambda x: x.k_degradation_per_h, reverse=True)
    return results
