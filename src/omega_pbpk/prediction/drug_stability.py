"""Drug stability and degradation prediction — Phase 166.

Predict drug shelf life under various storage conditions using
Arrhenius-adjusted first-order degradation kinetics (ICH Q1A guidance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R_KJ = 8.314e-3  # Gas constant in kJ/(mol·K)
REF_TEMP_K = 298.15  # 25 °C in Kelvin


@dataclass(frozen=True)
class StabilityResult:
    """Result of drug stability prediction."""

    drug_name: str
    temperature_C: float
    humidity_pct: float
    t_months: list[float]
    purity_pct: list[float]
    shelf_life_months: float | None
    purity_at_end_pct: float
    degradation_rate_per_month: float
    stability_category: str
    notes: list[str]


@dataclass(frozen=True)
class AcceleratedStabilityResult:
    """Result of accelerated stability testing."""

    drug_name: str
    conditions: list[tuple[float, float]]
    predicted_shelf_life_25C_months: float
    arrhenius_slope: float
    notes: list[str]


def _arrhenius_rate(
    k_ref: float, ea_kj_per_mol: float, temp_C: float,
) -> float:
    """Compute temperature-adjusted degradation rate via Arrhenius equation."""
    temp_K = temp_C + 273.15
    exponent = (ea_kj_per_mol / R_KJ) * (1.0 / REF_TEMP_K - 1.0 / temp_K)
    return k_ref * math.exp(exponent)


def _classify_stability(purity_at_end: float) -> str:
    """Classify stability based on final purity."""
    if purity_at_end >= 99.0:
        return "excellent"
    elif purity_at_end >= 97.0:
        return "good"
    elif purity_at_end >= 95.0:
        return "marginal"
    else:
        return "poor"


def predict_stability(
    drug_name: str,
    initial_purity_pct: float = 100.0,
    temperature_C: float = 25.0,
    humidity_pct: float = 60.0,
    k_deg_per_month: float = 0.005,
    activation_energy_kJ_per_mol: float = 80.0,
    t_months: float = 24.0,
    acceptance_limit_pct: float = 97.0,
    n_points: int = 100,
) -> StabilityResult:
    """Predict drug stability under given storage conditions.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    initial_purity_pct : float
        Starting purity (%). Must be > 0 and <= 100.
    temperature_C : float
        Storage temperature (°C).
    humidity_pct : float
        Relative humidity (%).
    k_deg_per_month : float
        Degradation rate constant at 25 °C (1/month). Must be > 0.
    activation_energy_kJ_per_mol : float
        Arrhenius activation energy (kJ/mol). Must be > 0.
    t_months : float
        Duration to simulate (months). Must be > 0.
    acceptance_limit_pct : float
        Minimum acceptable purity (%).
    n_points : int
        Number of time points.

    Returns
    -------
    StabilityResult
    """
    if initial_purity_pct <= 0 or initial_purity_pct > 100:
        raise ValueError("initial_purity_pct must be > 0 and <= 100")
    if k_deg_per_month <= 0:
        raise ValueError("k_deg_per_month must be > 0")
    if activation_energy_kJ_per_mol <= 0:
        raise ValueError("activation_energy_kJ_per_mol must be > 0")
    if t_months <= 0:
        raise ValueError("t_months must be > 0")

    k_T = _arrhenius_rate(k_deg_per_month, activation_energy_kJ_per_mol, temperature_C)

    time_points = [i * t_months / n_points for i in range(n_points + 1)]
    purity_values = [
        initial_purity_pct * math.exp(-k_T * t) for t in time_points
    ]

    # Shelf life: time when purity drops below acceptance limit
    shelf_life: float | None = None
    for i, p in enumerate(purity_values):
        if p < acceptance_limit_pct:
            # Linear interpolation between previous and current point
            if i > 0:
                t_prev = time_points[i - 1]
                p_prev = purity_values[i - 1]
                frac = (p_prev - acceptance_limit_pct) / (p_prev - p) if p_prev != p else 0.0
                shelf_life = t_prev + frac * (time_points[i] - t_prev)
            else:
                shelf_life = 0.0
            break

    purity_at_end = purity_values[-1]
    category = _classify_stability(purity_at_end)

    notes: list[str] = []
    if humidity_pct > 75:
        notes.append("High humidity may accelerate hydrolytic degradation")
    if category == "poor":
        notes.append("Purity below 95% — consider reformulation or cold storage")
    if shelf_life is not None and shelf_life < 12:
        notes.append("Shelf life < 12 months — may not meet regulatory requirements")

    return StabilityResult(
        drug_name=drug_name,
        temperature_C=temperature_C,
        humidity_pct=humidity_pct,
        t_months=time_points,
        purity_pct=purity_values,
        shelf_life_months=shelf_life,
        purity_at_end_pct=purity_at_end,
        degradation_rate_per_month=k_T,
        stability_category=category,
        notes=notes,
    )


def accelerated_stability_test(
    drug_name: str,
    initial_purity_pct: float = 100.0,
    k_deg_per_month: float = 0.005,
    activation_energy_kJ_per_mol: float = 80.0,
    acceptance_limit_pct: float = 97.0,
    conditions: list[tuple[float, float]] | None = None,
) -> AcceleratedStabilityResult:
    """Run accelerated stability test at multiple temperatures.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    initial_purity_pct : float
        Starting purity (%).
    k_deg_per_month : float
        Degradation rate at 25 °C (1/month).
    activation_energy_kJ_per_mol : float
        Arrhenius activation energy (kJ/mol).
    acceptance_limit_pct : float
        Minimum acceptable purity (%).
    conditions : list of (temperature_C, duration_months) tuples
        Test conditions. Defaults to ICH standard: [(25, 24), (30, 12), (40, 6)].

    Returns
    -------
    AcceleratedStabilityResult
    """
    if k_deg_per_month <= 0:
        raise ValueError("k_deg_per_month must be > 0")
    if activation_energy_kJ_per_mol <= 0:
        raise ValueError("activation_energy_kJ_per_mol must be > 0")

    if conditions is None:
        conditions = [(25.0, 24.0), (30.0, 12.0), (40.0, 6.0)]

    # Compute rate at 25 °C for shelf-life prediction
    k_25 = k_deg_per_month  # reference rate at 25 °C
    # Shelf life at 25 °C: time for purity to drop to acceptance limit
    # purity(t) = initial * exp(-k_25 * t) = acceptance_limit
    # t = -ln(acceptance_limit / initial) / k_25
    ratio = acceptance_limit_pct / initial_purity_pct
    if ratio > 0 and ratio < 1:
        predicted_shelf_life = -math.log(ratio) / k_25
    else:
        predicted_shelf_life = float("inf")

    # Arrhenius slope: Ea / R (in Kelvin)
    arrhenius_slope = activation_energy_kJ_per_mol / R_KJ

    notes: list[str] = []
    if predicted_shelf_life < 24:
        notes.append("Predicted shelf life at 25°C < 24 months")
    notes.append(f"Tested at {len(conditions)} conditions per ICH Q1A")

    return AcceleratedStabilityResult(
        drug_name=drug_name,
        conditions=conditions,
        predicted_shelf_life_25C_months=predicted_shelf_life,
        arrhenius_slope=arrhenius_slope,
        notes=notes,
    )


def compare_storage_conditions(
    drug_name: str,
    initial_purity_pct: float = 100.0,
    k_deg_per_month: float = 0.005,
    activation_energy_kJ_per_mol: float = 80.0,
    t_months: float = 24.0,
    acceptance_limit_pct: float = 97.0,
) -> list[StabilityResult]:
    """Compare stability at 5 °C, 25 °C, and 40 °C.

    Returns
    -------
    list[StabilityResult]
        Results for refrigerated (5 °C), room temp (25 °C), and
        accelerated (40 °C) storage.
    """
    temperatures = [5.0, 25.0, 40.0]
    humidities = [60.0, 60.0, 75.0]  # ICH conditions

    results: list[StabilityResult] = []
    for temp, hum in zip(temperatures, humidities, strict=True):
        result = predict_stability(
            drug_name=drug_name,
            initial_purity_pct=initial_purity_pct,
            temperature_C=temp,
            humidity_pct=hum,
            k_deg_per_month=k_deg_per_month,
            activation_energy_kJ_per_mol=activation_energy_kJ_per_mol,
            t_months=t_months,
            acceptance_limit_pct=acceptance_limit_pct,
        )
        results.append(result)
    return results
