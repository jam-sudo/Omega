"""
Phase 440: Predict drug degradation kinetics as a function of formulation pH
(acid/base hydrolysis).

First-order pH-rate profile with Arrhenius temperature correction.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PHStabilityResult:
    """Result of formulation pH stability prediction."""

    drug_name: str
    formulation_ph: float
    storage_temp_C: float
    k_deg_per_day: float
    t90_days: float
    t95_days: float
    shelf_life_days: float
    stability_class: str
    degradation_pct_6months: float
    optimal_ph: float
    activation_energy_kJ_per_mol: float
    q10_factor: float
    temperature_sensitivity: str
    notes: str


def _assign_stability_class(shelf_life_days: float) -> str:
    """Assign stability class based on shelf life."""
    if shelf_life_days > 730:
        return "excellent"
    elif shelf_life_days >= 365:
        return "good"
    elif shelf_life_days >= 90:
        return "moderate"
    else:
        return "poor"


def _assign_temperature_sensitivity(ea_kJ_per_mol: float) -> str:
    """Assign temperature sensitivity based on activation energy."""
    if ea_kJ_per_mol < 50:
        return "low"
    elif ea_kJ_per_mol <= 100:
        return "moderate"
    else:
        return "high"


def predict_ph_stability(
    drug_name: str,
    formulation_ph: float,
    storage_temp_C: float = 25.0,
    optimal_ph: float = 7.0,
    k0_per_day: float = 1e-4,
    ea_kJ_per_mol: float = 80.0,
) -> PHStabilityResult:
    """
    Predict chemical stability of a drug formulation at a given pH and temperature.

    pH-rate profile: V-shaped minimum at optimal_ph.
    - Acid contribution: 10^(0.5 * max(0, optimal_ph - ph))
    - Base contribution: 10^(0.3 * max(0, ph - optimal_ph))
    Temperature correction: Arrhenius with Ea.

    Parameters
    ----------
    drug_name : str
    formulation_ph : float
        Target formulation pH (0-14).
    storage_temp_C : float
        Storage temperature in Celsius (-20 to 80).
    optimal_ph : float
        pH of minimum degradation rate.
    k0_per_day : float
        Base degradation rate constant at optimal_ph and 25 degrees C (per day).
    ea_kJ_per_mol : float
        Activation energy in kJ/mol (must be > 0).

    Returns
    -------
    PHStabilityResult
    """
    # --- Validation ---
    if not (0 <= formulation_ph <= 14):
        raise ValueError(f"formulation_ph must be in [0, 14], got {formulation_ph}")
    if not (-20 <= storage_temp_C <= 80):
        raise ValueError(f"storage_temp_C must be in [-20, 80], got {storage_temp_C}")
    if ea_kJ_per_mol <= 0:
        raise ValueError(f"ea_kJ_per_mol must be > 0, got {ea_kJ_per_mol}")

    # --- pH-rate profile ---
    acid_factor = 10 ** (0.5 * max(0.0, optimal_ph - formulation_ph))
    base_factor = 10 ** (0.3 * max(0.0, formulation_ph - optimal_ph))
    ph_factor = max(acid_factor, base_factor)
    k_deg_at_25C = k0_per_day * ph_factor

    # --- Arrhenius temperature correction ---
    R = 8.314  # J/(mol*K)
    Ea_J = ea_kJ_per_mol * 1000.0
    T_storage = storage_temp_C + 273.15
    T_ref = 298.15  # 25 degrees C in K
    arrhenius_factor = math.exp(Ea_J / R * (1.0 / T_ref - 1.0 / T_storage))
    k_deg_per_day = k_deg_at_25C * arrhenius_factor

    # Guard against zero or negative rate (numerical edge case)
    if k_deg_per_day <= 0:
        k_deg_per_day = 1e-12

    # --- Derived metrics ---
    t90_days = -math.log(0.9) / k_deg_per_day
    t95_days = -math.log(0.95) / k_deg_per_day
    shelf_life_days = t90_days

    stability_class = _assign_stability_class(shelf_life_days)
    degradation_pct_6months = (1.0 - math.exp(-k_deg_per_day * 180.0)) * 100.0

    # Q10 factor
    q10_factor = 2.0 ** ((storage_temp_C - 25.0) / 10.0)

    temperature_sensitivity = _assign_temperature_sensitivity(ea_kJ_per_mol)

    notes = (
        f"At pH {formulation_ph:.1f} and {storage_temp_C:.0f}C, "
        f"shelf life (t90) is {shelf_life_days:.0f} days ({stability_class}). "
        f"Optimal pH for stability: {optimal_ph:.1f}. "
        f"Degradation in 6 months: {degradation_pct_6months:.1f}%. "
        f"Temperature sensitivity: {temperature_sensitivity} (Ea={ea_kJ_per_mol:.0f} kJ/mol)."
    )

    return PHStabilityResult(
        drug_name=drug_name,
        formulation_ph=formulation_ph,
        storage_temp_C=storage_temp_C,
        k_deg_per_day=k_deg_per_day,
        t90_days=t90_days,
        t95_days=t95_days,
        shelf_life_days=shelf_life_days,
        stability_class=stability_class,
        degradation_pct_6months=degradation_pct_6months,
        optimal_ph=optimal_ph,
        activation_energy_kJ_per_mol=ea_kJ_per_mol,
        q10_factor=q10_factor,
        temperature_sensitivity=temperature_sensitivity,
        notes=notes,
    )


def screen_ph_range(
    drug_name: str,
    ph_values: list,
    storage_temp_C: float = 25.0,
    optimal_ph: float = 7.0,
    k0_per_day: float = 1e-4,
    ea_kJ_per_mol: float = 80.0,
) -> list:
    """
    Screen stability across a range of pH values.

    Returns list of PHStabilityResult sorted by shelf_life_days descending (most stable first).
    """
    results = []
    for ph in ph_values:
        result = predict_ph_stability(
            drug_name=drug_name,
            formulation_ph=ph,
            storage_temp_C=storage_temp_C,
            optimal_ph=optimal_ph,
            k0_per_day=k0_per_day,
            ea_kJ_per_mol=ea_kJ_per_mol,
        )
        results.append(result)
    results.sort(key=lambda r: r.shelf_life_days, reverse=True)
    return results
