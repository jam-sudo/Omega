"""
Phase 410 — Protein Aggregation Kinetics in Formulation

Predicts protein aggregation kinetics in pharmaceutical formulations as a
function of pH, ionic strength, temperature, and excipients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Excipient protection factors (lower = more protective)
_EXCIPIENT_FACTORS = {
    "none": 1.0,
    "sucrose": 0.3,  # strong stabilizer
    "polysorbate": 0.5,  # surfactant reduces interfacial aggregation
    "arginine": 0.4,  # charge repulsion
    "histidine": 0.6,  # buffer at low pH
}

_VALID_EXCIPIENTS = set(_EXCIPIENT_FACTORS.keys())

# Arrhenius parameters
_Ea_kJ_per_mol = 80.0  # activation energy, kJ/mol
_R_kJ_per_mol_K = 8.314e-3  # gas constant, kJ/(mol·K)
_T_REF_K = 298.15  # reference temperature (25 °C)
_T_STORAGE_K = 277.15  # storage temperature (4 °C)

_K_BASE = 0.001  # base aggregation rate at 25 °C, pH 7.0, IS=150 mM (/h)


@dataclass
class FormulationAggregationResult:
    drug_name: str
    formulation_ph: float
    ionic_strength_mM: float
    temperature_C: float
    excipient: str
    times_h: list
    monomer_pct: list
    aggregate_pct: list
    t50_h: float
    t10_h: float
    shelf_life_years: float
    optimal_ph: float
    aggregation_rate_per_h: float
    colloidal_stability_score: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ph_factor(ph: float) -> float:
    """U-shaped pH effect; minimum (most stable) at pH 7.0 (assumed pI)."""
    raw = 1.0 + 2.0 * (ph - 7.0) ** 2 / 16.0
    return max(1.0, min(10.0, raw))


def _is_factor(ionic_strength_mM: float) -> float:
    """Higher ionic strength beyond 150 mM increases aggregation."""
    return 1.0 + 0.003 * max(0.0, ionic_strength_mM - 150.0)


def _temp_factor(temperature_C: float) -> float:
    """Arrhenius temperature factor relative to 25 °C reference."""
    T_K = temperature_C + 273.15
    return math.exp(_Ea_kJ_per_mol / _R_kJ_per_mol_K * (1.0 / _T_REF_K - 1.0 / T_K))


def _compute_k_eff(
    ph: float,
    ionic_strength_mM: float,
    temperature_C: float,
    excipient: str,
) -> float:
    """Compute effective aggregation rate constant (/h)."""
    return (
        _K_BASE
        * _ph_factor(ph)
        * _is_factor(ionic_strength_mM)
        * _temp_factor(temperature_C)
        * _EXCIPIENT_FACTORS[excipient]
    )


def _compute_k_4C(ph: float, ionic_strength_mM: float, excipient: str) -> float:
    """Compute aggregation rate at 4 °C storage."""
    return (
        _K_BASE
        * _ph_factor(ph)
        * _is_factor(ionic_strength_mM)
        * _temp_factor(4.0)
        * _EXCIPIENT_FACTORS[excipient]
    )


def _simulate_monomer(k_eff: float, t_end_h: float, dt_h: float):
    """
    Forward Euler simulation of monomer depletion.
    dM/dt = -k_eff * M  → analytical: M(t) = exp(-k_eff * t)
    Uses Forward Euler for consistency with project conventions.
    Returns (times_h, monomer_fraction_list).
    """
    times = []
    monomers = []
    M = 1.0  # dimensionless monomer fraction
    t = 0.0
    while t <= t_end_h + 1e-9:
        times.append(t)
        monomers.append(M)
        # Forward Euler
        M = M - k_eff * M * dt_h
        M = max(0.0, M)
        t += dt_h
    return times, monomers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_formulation_aggregation(
    drug_name: str,
    formulation_ph: float,
    ionic_strength_mM: float,
    temperature_C: float = 25.0,
    excipient: str = "none",
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> FormulationAggregationResult:
    """
    Simulate protein aggregation kinetics in a pharmaceutical formulation.

    Parameters
    ----------
    drug_name : str
    formulation_ph : float  in [4, 10]
    ionic_strength_mM : float  >= 0
    temperature_C : float  in [0, 80]
    excipient : str  one of "none", "sucrose", "polysorbate", "arginine", "histidine"
    t_end_h : float  simulation end time (hours)
    dt_h : float  Forward Euler step size (hours)

    Returns
    -------
    FormulationAggregationResult
    """
    _validate_inputs(formulation_ph, ionic_strength_mM, temperature_C, excipient)

    k_eff = _compute_k_eff(formulation_ph, ionic_strength_mM, temperature_C, excipient)

    times, monomers = _simulate_monomer(k_eff, t_end_h, dt_h)

    monomer_pct = [m * 100.0 for m in monomers]
    aggregate_pct = [100.0 - m for m in monomer_pct]

    # t50: time when 50% monomer remains → ln(2)/k_eff
    t50_h = math.log(2.0) / k_eff if k_eff > 0 else float("inf")

    # t10: time when 10% lost (90% monomer remains) → -ln(0.9)/k_eff
    t10_h = -math.log(0.9) / k_eff if k_eff > 0 else float("inf")

    # Shelf life at 4 °C: t10 in years
    k_4C = _compute_k_4C(formulation_ph, ionic_strength_mM, excipient)
    if k_4C > 0:
        t10_4C_h = -math.log(0.9) / k_4C
        shelf_life_years = t10_4C_h / 8760.0
    else:
        shelf_life_years = float("inf")

    optimal_ph = 7.0  # minimum aggregation at pI

    colloidal_stability_score = max(0.0, 100.0 - k_eff * 1000.0)

    notes = (
        f"k_eff={k_eff:.6f}/h; pH factor={_ph_factor(formulation_ph):.3f}; "
        f"IS factor={_is_factor(ionic_strength_mM):.3f}; "
        f"temp factor={_temp_factor(temperature_C):.3f}; "
        f"excipient factor={_EXCIPIENT_FACTORS[excipient]}."
    )

    return FormulationAggregationResult(
        drug_name=drug_name,
        formulation_ph=formulation_ph,
        ionic_strength_mM=ionic_strength_mM,
        temperature_C=temperature_C,
        excipient=excipient,
        times_h=times,
        monomer_pct=monomer_pct,
        aggregate_pct=aggregate_pct,
        t50_h=t50_h,
        t10_h=t10_h,
        shelf_life_years=shelf_life_years,
        optimal_ph=optimal_ph,
        aggregation_rate_per_h=k_eff,
        colloidal_stability_score=colloidal_stability_score,
        notes=notes,
    )


def screen_excipients_aggregation(
    drug_name: str,
    formulation_ph: float,
    ionic_strength_mM: float = 150.0,
    temperature_C: float = 25.0,
) -> list[FormulationAggregationResult]:
    """
    Screen all 5 excipients and return results sorted by shelf_life_years descending.
    """
    results = [
        simulate_formulation_aggregation(
            drug_name=drug_name,
            formulation_ph=formulation_ph,
            ionic_strength_mM=ionic_strength_mM,
            temperature_C=temperature_C,
            excipient=excipient,
        )
        for excipient in _VALID_EXCIPIENTS
    ]
    return sorted(results, key=lambda r: r.shelf_life_years, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    formulation_ph: float,
    ionic_strength_mM: float,
    temperature_C: float,
    excipient: str,
) -> None:
    if not (4.0 <= formulation_ph <= 10.0):
        raise ValueError(f"formulation_ph must be in [4, 10], got {formulation_ph}.")
    if ionic_strength_mM < 0:
        raise ValueError(f"ionic_strength_mM must be >= 0, got {ionic_strength_mM}.")
    if not (0.0 <= temperature_C <= 80.0):
        raise ValueError(f"temperature_C must be in [0, 80], got {temperature_C}.")
    if excipient not in _VALID_EXCIPIENTS:
        raise ValueError(
            f"excipient must be one of {sorted(_VALID_EXCIPIENTS)}, got '{excipient}'."
        )
