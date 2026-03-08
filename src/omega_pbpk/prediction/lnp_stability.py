"""Phase 534 — Drug Lipid Nanoparticle Stability.

Predict lipid nanoparticle stability and drug retention kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LNPStabilityResult:
    """Result of LNP stability prediction."""

    formulation_name: str
    drug_name: str
    pka_lipid: float
    temperature_celsius: float
    ph_storage: float
    k_leak_per_h: float
    t50_days: float
    times_days: list
    fraction_retained: list
    pdi_values: list
    diameter_nm: list
    shelf_life_90pct_days: float  # -1 if stable throughout
    stability_class: str  # "stable" / "moderate" / "unstable"
    aggregation_risk: bool
    storage_recommendation: str  # "room_temp" / "refrigerated" / "frozen"
    notes: str


def predict_lnp_stability(
    formulation_name: str,
    drug_name: str,
    pka_lipid: float,
    initial_diameter_nm: float = 100.0,
    initial_pdi: float = 0.1,
    temperature_celsius: float = 4.0,
    ph_storage: float = 7.4,
    t_max_days: float = 90.0,
) -> LNPStabilityResult:
    """Predict LNP stability and drug retention over time.

    Parameters
    ----------
    formulation_name:
        Name of the LNP formulation.
    drug_name:
        Name of the encapsulated drug.
    pka_lipid:
        Apparent pKa of the ionizable lipid.
    initial_diameter_nm:
        Initial particle diameter in nm.
    initial_pdi:
        Initial polydispersity index (0 < PDI < 1).
    temperature_celsius:
        Storage temperature in °C.
    ph_storage:
        Storage pH.
    t_max_days:
        Maximum storage duration to simulate (days).

    Returns
    -------
    LNPStabilityResult
    """
    if pka_lipid <= 0:
        raise ValueError("pka_lipid must be positive")
    if initial_diameter_nm <= 0:
        raise ValueError("initial_diameter_nm must be positive")
    if not (0 < initial_pdi < 1):
        raise ValueError("initial_pdi must be between 0 and 1 (exclusive)")

    # Leakage rate constants
    k_base = 0.01  # 1/h base leakage rate
    t_factor = 2.0 ** ((temperature_celsius - 4.0) / 10.0)  # Q10 rule, ref 4°C
    # pH factor: protonated (pH < pKa) → stable; deprotonated (pH > pKa) → more leakage
    ph_factor = 1.0 if ph_storage < pka_lipid else 2.0
    k_leak = k_base * t_factor * ph_factor  # 1/h

    # t50 in days: fraction_retained(t50) = 0.5
    # exp(-k_leak * t50 * 24) = 0.5 → t50 = ln(2) / (k_leak * 24)
    t50_days = math.log(2.0) / (k_leak * 24.0)

    # Build stability profile at 0, 10, 20, ..., t_max_days
    step = 10.0
    n_steps = int(t_max_days / step)
    times_days = [step * i for i in range(n_steps + 1)]

    fraction_retained = []
    pdi_values = []
    diameter_nm = []

    shelf_life_90pct_days = -1.0  # assume stable unless found

    for t in times_days:
        # Drug retention
        fr = math.exp(-k_leak * t * 24.0)
        fraction_retained.append(fr)

        # PDI growth
        pdi_t = initial_pdi + 0.001 * t
        pdi_values.append(pdi_t)

        # Size growth (only when PDI > 0.2)
        if pdi_t > 0.2:
            d_t = initial_diameter_nm * (1.0 + 0.005 * t)
        else:
            d_t = initial_diameter_nm
        diameter_nm.append(d_t)

        # Shelf life: first time point where retention drops below 90%
        if shelf_life_90pct_days < 0.0 and fr < 0.90 and t > 0:
            shelf_life_90pct_days = t

    # Aggregation risk: PDI at day 90 (or last simulated time)
    pdi_at_90 = initial_pdi + 0.001 * t_max_days
    aggregation_risk = pdi_at_90 > 0.25

    # Stability class based on t50
    if t50_days > 30.0:
        stability_class = "stable"
    elif t50_days >= 10.0:
        stability_class = "moderate"
    else:
        stability_class = "unstable"

    # Storage recommendation
    if temperature_celsius <= 2.0:
        storage_recommendation = "frozen"
    elif temperature_celsius <= 8.0:
        storage_recommendation = "refrigerated"
    else:
        storage_recommendation = "room_temp"

    # Notes
    notes_parts = [
        f"k_leak={k_leak:.5f}/h (T_factor={t_factor:.3f}, pH_factor={ph_factor:.1f}).",
        f"t50={t50_days:.1f} days; stability class: {stability_class}.",
    ]
    if aggregation_risk:
        notes_parts.append("PDI at 90 days exceeds 0.25; aggregation risk present.")
    if shelf_life_90pct_days > 0:
        notes_parts.append(f"90% retention shelf life: {shelf_life_90pct_days:.0f} days.")
    else:
        notes_parts.append("Retains >90% drug throughout simulation period.")
    notes = " ".join(notes_parts)

    return LNPStabilityResult(
        formulation_name=formulation_name,
        drug_name=drug_name,
        pka_lipid=pka_lipid,
        temperature_celsius=temperature_celsius,
        ph_storage=ph_storage,
        k_leak_per_h=k_leak,
        t50_days=t50_days,
        times_days=times_days,
        fraction_retained=fraction_retained,
        pdi_values=pdi_values,
        diameter_nm=diameter_nm,
        shelf_life_90pct_days=shelf_life_90pct_days,
        stability_class=stability_class,
        aggregation_risk=aggregation_risk,
        storage_recommendation=storage_recommendation,
        notes=notes,
    )


def compare_storage_temperatures(
    formulation_name: str,
    drug_name: str,
    pka_lipid: float,
    temperatures: list,
) -> list:
    """Compare LNP stability at different storage temperatures.

    Parameters
    ----------
    formulation_name:
        Name of the LNP formulation.
    drug_name:
        Name of the drug.
    pka_lipid:
        Apparent pKa of the ionizable lipid.
    temperatures:
        List of temperature values in °C.

    Returns
    -------
    List of LNPStabilityResult sorted by t50_days descending (most stable first).
    """
    results = []
    for temp in temperatures:
        result = predict_lnp_stability(
            formulation_name=formulation_name,
            drug_name=drug_name,
            pka_lipid=pka_lipid,
            temperature_celsius=temp,
        )
        results.append(result)

    results.sort(key=lambda r: r.t50_days, reverse=True)
    return results
