"""Phase 376 — Biologic Drug Aggregation Kinetics.

Models aggregation kinetics of biologic drugs (proteins/mAbs) under stress
conditions using nucleation-elongation (first-order decay) kinetics.
"""

import math
from dataclasses import dataclass

_STRESS_MULTIPLIERS: dict[str, float] = {
    "normal": 1.0,
    "freeze_thaw": 5.0,
    "heat_40C": 20.0,
    "heat_60C": 200.0,
    "agitation": 8.0,
    "pH_stress": 15.0,
    "light_UV": 10.0,
}


@dataclass
class BiologicAggregationResult:
    """Result of biologic aggregation kinetics simulation."""

    drug_name: str
    stress_condition: str
    initial_monomer_mg_mL: float
    times_h: list
    monomer_fraction: list
    aggregate_fraction: list
    t50_h: float
    t10_h: float
    aggregation_rate_per_h: float
    shelf_life_h: float
    stress_multiplier: float
    notes: str


def simulate_aggregation(
    drug_name: str,
    stress_condition: str,
    initial_monomer_mg_mL: float,
    base_aggregation_rate_per_h: float = 0.001,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> BiologicAggregationResult:
    """Simulate biologic drug aggregation under a given stress condition.

    Uses simple first-order monomer loss:  dM/dt = -k_eff * M
    giving M(t) = exp(-k_eff * t).

    Parameters
    ----------
    drug_name:
        Name of the biologic drug.
    stress_condition:
        One of the keys in _STRESS_MULTIPLIERS.
    initial_monomer_mg_mL:
        Initial monomer concentration (mg/mL).
    base_aggregation_rate_per_h:
        Baseline first-order aggregation rate (1/h) under normal conditions.
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Time step for trajectory (hours).

    Returns
    -------
    BiologicAggregationResult
    """
    # --- Validation ---
    if initial_monomer_mg_mL <= 0:
        raise ValueError("initial_monomer_mg_mL must be positive")
    if base_aggregation_rate_per_h <= 0:
        raise ValueError("base_aggregation_rate_per_h must be positive")
    if stress_condition not in _STRESS_MULTIPLIERS:
        valid = ", ".join(sorted(_STRESS_MULTIPLIERS.keys()))
        raise ValueError(
            f"stress_condition '{stress_condition}' not recognised. Valid options: {valid}"
        )

    stress_multiplier = _STRESS_MULTIPLIERS[stress_condition]
    effective_rate = base_aggregation_rate_per_h * stress_multiplier

    # Build time trajectory
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times_h: list[float] = [i * dt_h for i in range(n_steps)]
    # Ensure last point is exactly t_end_h
    times_h[-1] = t_end_h

    monomer_fraction: list[float] = []
    aggregate_fraction: list[float] = []
    for t in times_h:
        mf = math.exp(-effective_rate * t)
        monomer_fraction.append(mf)
        aggregate_fraction.append(1.0 - mf)

    # t50: time for 50% monomer loss  → exp(-k*t50) = 0.5
    t50_h = math.log(2.0) / effective_rate

    # t10: time for 10% monomer loss  → exp(-k*t10) = 0.9
    t10_h = -math.log(0.9) / effective_rate

    shelf_life_h = t10_h

    # Notes
    notes_parts = [
        f"Stress condition: {stress_condition} (multiplier={stress_multiplier:.1f}).",
        f"Effective rate: {effective_rate:.4g} /h.",
        f"Shelf life (10% loss): {shelf_life_h:.1f} h.",
    ]
    if stress_condition in ("heat_60C", "heat_40C"):
        notes_parts.append("Thermal stress significantly accelerates aggregation.")
    if stress_multiplier >= 100.0:
        notes_parts.append(
            "Very rapid aggregation predicted; storage under these conditions not recommended."
        )

    return BiologicAggregationResult(
        drug_name=drug_name,
        stress_condition=stress_condition,
        initial_monomer_mg_mL=initial_monomer_mg_mL,
        times_h=times_h,
        monomer_fraction=monomer_fraction,
        aggregate_fraction=aggregate_fraction,
        t50_h=t50_h,
        t10_h=t10_h,
        aggregation_rate_per_h=effective_rate,
        shelf_life_h=shelf_life_h,
        stress_multiplier=stress_multiplier,
        notes=" ".join(notes_parts),
    )


def screen_stress_conditions(
    drug_name: str,
    initial_monomer_mg_mL: float,
    base_aggregation_rate_per_h: float = 0.001,
) -> list[BiologicAggregationResult]:
    """Screen all stress conditions and rank by stability.

    Parameters
    ----------
    drug_name:
        Name of the biologic drug.
    initial_monomer_mg_mL:
        Initial monomer concentration (mg/mL).
    base_aggregation_rate_per_h:
        Baseline aggregation rate under normal conditions (1/h).

    Returns
    -------
    List of BiologicAggregationResult sorted by shelf_life_h descending
    (most stable condition first).
    """
    results = [
        simulate_aggregation(
            drug_name=drug_name,
            stress_condition=cond,
            initial_monomer_mg_mL=initial_monomer_mg_mL,
            base_aggregation_rate_per_h=base_aggregation_rate_per_h,
        )
        for cond in _STRESS_MULTIPLIERS
    ]
    return sorted(results, key=lambda r: r.shelf_life_h, reverse=True)
