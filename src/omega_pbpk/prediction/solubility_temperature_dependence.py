"""Phase 413 — Drug Solubility Temperature Dependence.

Predicts aqueous drug solubility as a function of temperature using the
van't Hoff equation.  Pure Python / stdlib only (math, dataclasses).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SolubilityTemperatureResult:
    """Result of a single solubility-temperature prediction."""

    drug_name: str
    temperature_C: float
    solubility_ref_mg_mL: float  # reference solubility at T_ref
    solubility_predicted_mg_mL: float  # at target temperature
    delta_H_solution_kJ_per_mol: float  # enthalpy of solution
    solubility_ratio: float  # predicted / reference
    temperature_sensitivity_class: str  # "low", "moderate", "high"
    storage_recommendation: str  # "refrigerate", "room_temperature", "freeze"
    intrinsic_solubility_mg_mL: float  # solubility at 25 C reference
    entropy_contribution_kJ_per_mol: float
    notes: str


# Gas constant  kJ / (mol K)
_R_KJ = 8.314e-3


def _estimate_delta_H(logp: float) -> float:
    """Rough estimate of enthalpy of solution from logP."""
    return max(5.0, 15.0 + 2.0 * logp)


def _van_t_hoff_ratio(delta_H: float, T_K: float, T_ref_K: float) -> float:
    """ln(S/S_ref) = -dH/R * (1/T - 1/T_ref)  =>  ratio = exp(...)."""
    exponent = -delta_H / _R_KJ * (1.0 / T_K - 1.0 / T_ref_K)
    return math.exp(exponent)


def _temperature_sensitivity_class(delta_H: float) -> str:
    """Classify sensitivity by 10-degree ratio S(35C)/S(25C)."""
    ratio_10 = _van_t_hoff_ratio(delta_H, 308.15, 298.15)
    if ratio_10 < 1.5:
        return "low"
    if ratio_10 < 3.0:
        return "moderate"
    return "high"


def _storage_recommendation(delta_H: float, solubility_predicted: float) -> str:
    """Derive simple storage recommendation."""
    if delta_H > 20.0:
        return "refrigerate"
    return "room_temperature"


def predict_solubility_temperature(
    drug_name: str,
    solubility_ref_mg_mL: float,
    temperature_C: float,
    logp: float,
    mw_da: float,
    t_ref_C: float = 25.0,
    delta_H_kJ_per_mol: float | None = None,
) -> SolubilityTemperatureResult:
    """Predict aqueous solubility at *temperature_C* via van't Hoff.

    Parameters
    ----------
    drug_name:
        Identifier / name of the drug.
    solubility_ref_mg_mL:
        Known solubility at the reference temperature *t_ref_C* (mg/mL).
    temperature_C:
        Target temperature (°C).
    logp:
        Octanol-water partition coefficient (dimensionless).
    mw_da:
        Molecular weight (Da).
    t_ref_C:
        Reference temperature (°C); default 25.
    delta_H_kJ_per_mol:
        Enthalpy of solution (kJ/mol).  Estimated from logP when *None*.

    Returns
    -------
    SolubilityTemperatureResult
    """
    # --- validation ---
    if solubility_ref_mg_mL <= 0:
        raise ValueError("solubility_ref_mg_mL must be > 0")
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if not (-20.0 <= temperature_C <= 100.0):
        raise ValueError("temperature_C must be in [-20, 100]")

    # --- enthalpy ---
    if delta_H_kJ_per_mol is None:
        delta_H = _estimate_delta_H(logp)
    else:
        delta_H = delta_H_kJ_per_mol

    T_K = temperature_C + 273.15
    T_ref_K = t_ref_C + 273.15

    # --- van't Hoff ---
    ratio = _van_t_hoff_ratio(delta_H, T_K, T_ref_K)
    solubility_predicted = solubility_ref_mg_mL * ratio

    # --- intrinsic solubility at 25 C ---
    if abs(t_ref_C - 25.0) < 1e-6:
        intrinsic_solubility = solubility_ref_mg_mL
    else:
        T_25_K = 298.15
        ratio_to_25 = _van_t_hoff_ratio(delta_H, T_25_K, T_ref_K)
        intrinsic_solubility = solubility_ref_mg_mL * ratio_to_25

    # --- entropy contribution (approximation) ---
    delta_S = delta_H / T_K  # kJ / (mol K)  simplified
    entropy_contribution = delta_H - delta_S * T_K / 1000.0  # rough

    # --- classification ---
    sens_class = _temperature_sensitivity_class(delta_H)
    storage_rec = _storage_recommendation(delta_H, solubility_predicted)

    notes = (
        f"Van't Hoff model; dH={delta_H:.1f} kJ/mol; "
        f"T_ref={t_ref_C:.1f} C; T_target={temperature_C:.1f} C"
    )

    return SolubilityTemperatureResult(
        drug_name=drug_name,
        temperature_C=temperature_C,
        solubility_ref_mg_mL=solubility_ref_mg_mL,
        solubility_predicted_mg_mL=solubility_predicted,
        delta_H_solution_kJ_per_mol=delta_H,
        solubility_ratio=ratio,
        temperature_sensitivity_class=sens_class,
        storage_recommendation=storage_rec,
        intrinsic_solubility_mg_mL=intrinsic_solubility,
        entropy_contribution_kJ_per_mol=entropy_contribution,
        notes=notes,
    )


def compare_temperatures(
    drug_name: str,
    solubility_ref_mg_mL: float,
    logp: float,
    mw_da: float,
    temperatures_C: list[float] | None = None,
) -> list[SolubilityTemperatureResult]:
    """Predict solubility at multiple temperatures and return sorted list.

    Default temperatures: [4.0, 15.0, 25.0, 37.0, 60.0].
    Results sorted by *solubility_predicted_mg_mL* ascending.
    """
    if temperatures_C is None:
        temperatures_C = [4.0, 15.0, 25.0, 37.0, 60.0]

    results = [
        predict_solubility_temperature(
            drug_name=drug_name,
            solubility_ref_mg_mL=solubility_ref_mg_mL,
            temperature_C=t,
            logp=logp,
            mw_da=mw_da,
        )
        for t in temperatures_C
    ]

    results.sort(key=lambda r: r.solubility_predicted_mg_mL)
    return results
