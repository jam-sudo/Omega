"""Phase 294 — Formulation Stability Under Temperature Cycling.

Predict drug degradation during temperature cycling using Arrhenius kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TemperatureCyclingResult:
    """Result of a temperature cycling stability simulation."""

    drug_name: str
    n_cycles: int
    temp_cold_C: float
    temp_warm_C: float
    total_cold_h: float
    total_warm_h: float
    k_cold_per_h: float
    k_warm_per_h: float
    initial_purity_pct: float
    purity_final_pct: float
    degradation_pct: float
    accelerated_degradation_factor: float
    stability_class: str
    failure_risk: str
    notes: str


_R_kJ_PER_MOL_K = 8.314e-3  # kJ / (mol * K)
_T_REF_K = 298.15  # 25 degC reference temperature


def _arrhenius_k(k_ref: float, ea_kJ_per_mol: float, temp_C: float) -> float:
    """Compute rate constant at temp_C using Arrhenius equation.

    k = k_ref * exp(Ea/R * (1/T_ref - 1/T))
    """
    t_k = temp_C + 273.15
    exponent = (ea_kJ_per_mol / _R_kJ_PER_MOL_K) * (1.0 / _T_REF_K - 1.0 / t_k)
    return k_ref * math.exp(exponent)


def _validate_cycling_inputs(
    n_cycles: int,
    t_cold_h_per_cycle: float,
    t_warm_h_per_cycle: float,
    temp_cold_C: float,
    temp_warm_C: float,
    initial_purity_pct: float,
    k_deg_ref_per_h: float,
    ea_kJ_per_mol: float,
) -> None:
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if t_cold_h_per_cycle <= 0:
        raise ValueError("t_cold_h_per_cycle must be > 0")
    if t_warm_h_per_cycle <= 0:
        raise ValueError("t_warm_h_per_cycle must be > 0")
    if temp_warm_C <= temp_cold_C:
        raise ValueError("temp_warm_C must be > temp_cold_C")
    if not (0 < initial_purity_pct <= 100):
        raise ValueError("initial_purity_pct must be in (0, 100]")
    if k_deg_ref_per_h <= 0:
        raise ValueError("k_deg_ref_per_h must be > 0")
    if ea_kJ_per_mol <= 0:
        raise ValueError("ea_kJ_per_mol must be > 0")


def simulate_temperature_cycling(
    drug_name: str,
    n_cycles: int,
    t_cold_h_per_cycle: float,
    t_warm_h_per_cycle: float,
    temp_cold_C: float,
    temp_warm_C: float,
    initial_purity_pct: float,
    k_deg_ref_per_h: float,
    ea_kJ_per_mol: float,
) -> TemperatureCyclingResult:
    """Predict drug degradation during repeated temperature cycling.

    Parameters
    ----------
    drug_name:
        Name of the drug / formulation.
    n_cycles:
        Number of temperature cycles.
    t_cold_h_per_cycle:
        Time at cold temperature per cycle (h).
    t_warm_h_per_cycle:
        Time at warm temperature per cycle (h).
    temp_cold_C:
        Cold storage temperature (degC).
    temp_warm_C:
        Warm/excursion temperature (degC).
    initial_purity_pct:
        Initial purity percentage (0 < value <= 100).
    k_deg_ref_per_h:
        First-order degradation rate constant at 25 degC reference (h^-1).
    ea_kJ_per_mol:
        Activation energy (kJ/mol).

    Returns
    -------
    TemperatureCyclingResult
    """
    _validate_cycling_inputs(
        n_cycles,
        t_cold_h_per_cycle,
        t_warm_h_per_cycle,
        temp_cold_C,
        temp_warm_C,
        initial_purity_pct,
        k_deg_ref_per_h,
        ea_kJ_per_mol,
    )

    k_cold = _arrhenius_k(k_deg_ref_per_h, ea_kJ_per_mol, temp_cold_C)
    k_warm = _arrhenius_k(k_deg_ref_per_h, ea_kJ_per_mol, temp_warm_C)

    total_cold_h = n_cycles * t_cold_h_per_cycle
    total_warm_h = n_cycles * t_warm_h_per_cycle

    # Total degradation across all cycles (independent degradation per phase)
    purity_final_pct = max(
        0.0,
        initial_purity_pct * math.exp(-k_cold * total_cold_h) * math.exp(-k_warm * total_warm_h),
    )

    degradation_pct = initial_purity_pct - purity_final_pct

    # Purity if entire time had been at cold temperature
    total_time_h = total_cold_h + total_warm_h
    purity_at_cold_only = initial_purity_pct * math.exp(-k_cold * total_time_h)
    cold_only_degradation = initial_purity_pct - purity_at_cold_only

    if cold_only_degradation > 0:
        accelerated_degradation_factor = degradation_pct / cold_only_degradation
    else:
        accelerated_degradation_factor = 1.0

    # Classify stability
    if purity_final_pct >= 0.97 * initial_purity_pct:
        stability_class = "stable"
    elif purity_final_pct >= 0.90 * initial_purity_pct:
        stability_class = "marginally_stable"
    else:
        stability_class = "unstable"

    # Failure risk
    if purity_final_pct < 0.90 * initial_purity_pct:
        failure_risk = "high"
    elif purity_final_pct < 0.95 * initial_purity_pct:
        failure_risk = "moderate"
    else:
        failure_risk = "low"

    notes = (
        f"Arrhenius kinetics; Ea={ea_kJ_per_mol} kJ/mol; "
        f"k_cold={k_cold:.6f}/h at {temp_cold_C}C; "
        f"k_warm={k_warm:.6f}/h at {temp_warm_C}C; "
        f"{n_cycles} cycles"
    )

    return TemperatureCyclingResult(
        drug_name=drug_name,
        n_cycles=n_cycles,
        temp_cold_C=temp_cold_C,
        temp_warm_C=temp_warm_C,
        total_cold_h=total_cold_h,
        total_warm_h=total_warm_h,
        k_cold_per_h=k_cold,
        k_warm_per_h=k_warm,
        initial_purity_pct=initial_purity_pct,
        purity_final_pct=purity_final_pct,
        degradation_pct=degradation_pct,
        accelerated_degradation_factor=accelerated_degradation_factor,
        stability_class=stability_class,
        failure_risk=failure_risk,
        notes=notes,
    )


def assess_cold_chain_risk(
    drug_name: str,
    n_excursions: int,
    excursion_duration_h: float,
    excursion_temp_C: float,
    initial_purity_pct: float,
    k_deg_ref_per_h: float,
    ea_kJ_per_mol: float,
) -> TemperatureCyclingResult:
    """Assess cold chain risk from temperature excursions.

    Models repeated excursions from default 5 degC cold storage to an elevated
    excursion temperature, with 1 h background cold time per cycle.

    Parameters
    ----------
    drug_name:
        Name of the drug / formulation.
    n_excursions:
        Number of excursion events.
    excursion_duration_h:
        Duration of each excursion (h).
    excursion_temp_C:
        Excursion temperature (degC). Must be > 5 degC.
    initial_purity_pct:
        Initial purity percentage (0 < value <= 100).
    k_deg_ref_per_h:
        First-order degradation rate constant at 25 degC reference (h^-1).
    ea_kJ_per_mol:
        Activation energy (kJ/mol).

    Returns
    -------
    TemperatureCyclingResult
    """
    return simulate_temperature_cycling(
        drug_name=drug_name,
        n_cycles=n_excursions,
        t_cold_h_per_cycle=1.0,
        t_warm_h_per_cycle=excursion_duration_h,
        temp_cold_C=5.0,
        temp_warm_C=excursion_temp_C,
        initial_purity_pct=initial_purity_pct,
        k_deg_ref_per_h=k_deg_ref_per_h,
        ea_kJ_per_mol=ea_kJ_per_mol,
    )
