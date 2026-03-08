"""Protein drug aggregation kinetics using nucleation-elongation model.

Models the time course of protein/peptide drug aggregation as a dynamic
process with cooperative nucleation and elongation phases.

References
----------
- Morris AM et al., Biochim Biophys Acta. 2009;1794(3):375-397 (aggregation kinetics)
- Arosio P et al., Trends Biotechnol. 2014;32(7):372-380 (nucleation-elongation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProteinAggregationKineticsResult:
    """Result container for protein aggregation kinetics simulation."""

    drug_name: str
    dose_mg_mL: float
    times_h: list[float] = field(default_factory=list)
    c_monomer_mg_mL: list[float] = field(default_factory=list)
    c_aggregate_mg_mL: list[float] = field(default_factory=list)
    c_nucleus_mg_mL: list[float] = field(default_factory=list)
    monomer_fraction: list[float] = field(default_factory=list)
    t_half_aggregation_h: float = 0.0
    lag_time_h: float = 0.0
    aggregation_rate_per_h: float = 0.0
    max_aggregate_pct: float = 0.0
    stability_class: str = ""
    notes: str = ""


def simulate_aggregation_kinetics(
    drug_name: str,
    initial_concentration_mg_mL: float,
    nucleation_rate_per_h: float,
    elongation_rate_per_mL_mg_h: float,
    critical_nucleus_size: int = 3,
    temperature_C: float = 25.0,
    ph: float = 7.4,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> ProteinAggregationKineticsResult:
    """Simulate protein drug aggregation kinetics.

    Parameters
    ----------
    drug_name : str
        Name of the protein drug.
    initial_concentration_mg_mL : float
        Initial monomer concentration (mg/mL).
    nucleation_rate_per_h : float
        Nucleation rate constant (per hour).
    elongation_rate_per_mL_mg_h : float
        Elongation rate constant (mL/mg/h).
    critical_nucleus_size : int
        Cooperativity of nucleation (>= 2).
    temperature_C : float
        Temperature in Celsius.
    ph : float
        Solution pH.
    t_end_h : float
        Simulation duration in hours.
    dt_h : float
        Time step for forward Euler integration.

    Returns
    -------
    ProteinAggregationKineticsResult
    """
    # Validation
    if initial_concentration_mg_mL <= 0:
        raise ValueError("initial_concentration_mg_mL must be positive")
    if nucleation_rate_per_h <= 0:
        raise ValueError("nucleation_rate_per_h must be positive")
    if elongation_rate_per_mL_mg_h <= 0:
        raise ValueError("elongation_rate_per_mL_mg_h must be positive")
    if critical_nucleus_size < 2:
        raise ValueError("critical_nucleus_size must be >= 2")

    c0 = initial_concentration_mg_mL

    # Temperature effect (Arrhenius-like)
    t_ref = 298.15  # 25 C in K
    t_k = temperature_C + 273.15
    temp_factor = math.exp(5000.0 * (1.0 / t_ref - 1.0 / t_k))

    # pH effect on nucleation
    ph_factor = 1.0 + abs(ph - 7.4) * 0.3

    # Adjusted rates
    nuc_rate = nucleation_rate_per_h * temp_factor * ph_factor
    elong_rate = elongation_rate_per_mL_mg_h * temp_factor

    # Forward Euler simulation
    n_steps = int(math.ceil(t_end_h / dt_h))
    c_m = c0  # monomer
    c_n = 0.0  # nucleus
    c_a = 0.0  # aggregate

    times: list[float] = []
    c_monomer_list: list[float] = []
    c_aggregate_list: list[float] = []
    c_nucleus_list: list[float] = []
    monomer_frac_list: list[float] = []
    agg_rate_list: list[float] = []

    lag_time = t_end_h
    t_half = t_end_h
    lag_found = False
    t_half_found = False

    for i in range(n_steps + 1):
        t = i * dt_h

        mf = c_m / c0 if c0 > 0 else 0.0

        times.append(round(t, 6))
        c_monomer_list.append(c_m)
        c_aggregate_list.append(c_a)
        c_nucleus_list.append(c_n)
        monomer_frac_list.append(mf)

        if not lag_found and mf < 0.99:
            lag_time = t
            lag_found = True

        if not t_half_found and mf <= 0.5:
            t_half = t
            t_half_found = True

        if i == n_steps:
            agg_rate_list.append(0.0)
            break

        # Rates
        rate_nuc = nuc_rate * (c_m**critical_nucleus_size)
        rate_elong = elong_rate * c_m * c_n

        # ODEs
        dc_m = -rate_nuc * critical_nucleus_size - rate_elong
        dc_n = rate_nuc - rate_elong * 0.1
        dc_a = rate_elong

        agg_rate_list.append(dc_a)

        # Euler step
        c_m += dc_m * dt_h
        c_n += dc_n * dt_h
        c_a += dc_a * dt_h

        # Clamp
        c_m = max(c_m, 0.0)
        c_n = max(c_n, 0.0)
        c_a = max(c_a, 0.0)

    # Post-processing
    max_agg_rate = max(agg_rate_list) if agg_rate_list else 0.0
    max_agg_pct = (max(c_aggregate_list) / c0 * 100.0) if c0 > 0 else 0.0

    if max_agg_pct < 5.0:
        stability = "stable"
    elif max_agg_pct < 20.0:
        stability = "borderline"
    else:
        stability = "unstable"

    notes_parts = []
    if temperature_C > 37.0:
        notes_parts.append("Elevated temperature accelerates aggregation")
    if abs(ph - 7.4) > 1.0:
        notes_parts.append("Non-physiological pH may destabilize protein")
    if max_agg_pct > 50.0:
        notes_parts.append("Significant aggregation observed")

    return ProteinAggregationKineticsResult(
        drug_name=drug_name,
        dose_mg_mL=c0,
        times_h=times,
        c_monomer_mg_mL=c_monomer_list,
        c_aggregate_mg_mL=c_aggregate_list,
        c_nucleus_mg_mL=c_nucleus_list,
        monomer_fraction=monomer_frac_list,
        t_half_aggregation_h=t_half,
        lag_time_h=lag_time,
        aggregation_rate_per_h=max_agg_rate,
        max_aggregate_pct=max_agg_pct,
        stability_class=stability,
        notes="; ".join(notes_parts),
    )
