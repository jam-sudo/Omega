"""
Phase 384 — Drug Degradation in Biological Matrices

Predict drug degradation rates in different biological matrices:
blood, plasma, liver homogenate, intestinal fluid, microsomes, etc.

Pure Python, stdlib only (math, dataclasses).
"""

import math
from dataclasses import dataclass

# Base degradation rates per minute at 37 degrees C
_MATRIX_BASE_RATES = {
    "whole_blood": 0.005,
    "plasma": 0.002,
    "liver_homogenate": 0.05,
    "intestinal_fluid": 0.01,
    "microsomes": 0.08,
    "gastric_fluid": 0.02,
    "cerebrospinal_fluid": 0.0005,
}

_HYDROLYSIS_MATRICES = {"plasma", "whole_blood", "intestinal_fluid"}
_OXIDATION_MATRICES = {"microsomes", "liver_homogenate"}

# Arrhenius parameters
_ACTIVATION_ENERGY_kJ_per_mol = 50.0  # kJ/mol
_R_kJ_per_mol_K = 8.314e-3  # kJ/(mol*K)
_T_REF_K = 310.15  # 37 degrees C in Kelvin


@dataclass(frozen=True)
class MatrixDegradationResult:
    """Result of drug degradation prediction in a biological matrix."""

    drug_name: str
    matrix: str
    temperature_C: float
    initial_concentration_mg_mL: float
    degradation_rate_per_min: float
    t50_min: float
    t90_min: float
    percent_remaining_60min: float
    stability_class: str
    primary_degradation_pathway: str
    activation_energy_kJ_per_mol: float
    q10_factor: float
    notes: str


def _classify_stability(t50_min: float) -> str:
    """Classify stability based on matrix half-life."""
    if t50_min > 60.0:
        return "stable"
    elif t50_min >= 10.0:
        return "moderate"
    else:
        return "labile"


def _determine_pathway(
    matrix: str,
    logp: float,
    n_ester_bonds: int,
) -> str:
    """Determine primary degradation pathway."""
    # Ester bonds take priority for hydrolysis-prone matrices
    if n_ester_bonds > 0:
        return "hydrolysis"
    if matrix in _OXIDATION_MATRICES:
        if logp > 1 and n_ester_bonds == 0:
            return "oxidation"
        return "enzymatic"
    return "combined"


def predict_matrix_degradation(
    drug_name: str,
    matrix: str,
    logp: float,
    mw_da: float,
    n_ester_bonds: int,
    n_amide_bonds: int,
    temperature_C: float = 37.0,
    initial_concentration_mg_mL: float = 1.0,
) -> MatrixDegradationResult:
    """
    Predict drug degradation rate in a biological matrix.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    matrix : str
        One of: whole_blood, plasma, liver_homogenate, intestinal_fluid,
        microsomes, gastric_fluid, cerebrospinal_fluid.
    logp : float
        Octanol-water partition coefficient.
    mw_da : float
        Molecular weight in Daltons.
    n_ester_bonds : int
        Number of ester bonds (hydrolysis-prone functional groups).
    n_amide_bonds : int
        Number of amide bonds.
    temperature_C : float
        Temperature in Celsius (default 37.0, valid range 4-60).
    initial_concentration_mg_mL : float
        Initial drug concentration in mg/mL.

    Returns
    -------
    MatrixDegradationResult
    """
    # --- Validation ---
    if matrix not in _MATRIX_BASE_RATES:
        valid = ", ".join(sorted(_MATRIX_BASE_RATES.keys()))
        raise ValueError(f"matrix must be one of: {valid}")
    if not (4.0 <= temperature_C <= 60.0):
        raise ValueError("temperature_C must be in [4, 60]")
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if n_ester_bonds < 0:
        raise ValueError("n_ester_bonds must be >= 0")
    if n_amide_bonds < 0:
        raise ValueError("n_amide_bonds must be >= 0")

    # --- Base rate + structural contributions ---
    base_rate = _MATRIX_BASE_RATES[matrix]
    ester_contribution = n_ester_bonds * 0.015
    amide_contribution = n_amide_bonds * 0.003
    lipophilicity_contribution = max(0.0, (logp - 2.0) * 0.002)

    total_rate_37 = base_rate + ester_contribution + amide_contribution + lipophilicity_contribution

    # --- Arrhenius temperature adjustment ---
    T_K = temperature_C + 273.15
    exponent = (_ACTIVATION_ENERGY_kJ_per_mol / _R_kJ_per_mol_K) * (1.0 / _T_REF_K - 1.0 / T_K)
    rate_at_T = total_rate_37 * math.exp(exponent)

    # Q10 factor: rate increase per 10 degrees C rise from reference
    q10_exponent = (_ACTIVATION_ENERGY_kJ_per_mol / _R_kJ_per_mol_K) * (
        1.0 / _T_REF_K - 1.0 / (_T_REF_K + 10.0)
    )
    q10_factor = math.exp(q10_exponent)

    # --- Derived metrics ---
    t50_min = math.log(2.0) / rate_at_T
    t90_min = math.log(10.0) / rate_at_T
    percent_remaining_60min = math.exp(-rate_at_T * 60.0) * 100.0

    stability_class = _classify_stability(t50_min)
    primary_degradation_pathway = _determine_pathway(matrix, logp, n_ester_bonds)

    # --- Notes ---
    note_parts = [
        f"Matrix: {matrix}",
        f"t1/2 = {t50_min:.1f} min",
        f"Pathway: {primary_degradation_pathway}",
    ]
    if n_ester_bonds > 0:
        note_parts.append(f"{n_ester_bonds} ester bond(s) detected")
    notes = "; ".join(note_parts)

    return MatrixDegradationResult(
        drug_name=drug_name,
        matrix=matrix,
        temperature_C=temperature_C,
        initial_concentration_mg_mL=initial_concentration_mg_mL,
        degradation_rate_per_min=rate_at_T,
        t50_min=t50_min,
        t90_min=t90_min,
        percent_remaining_60min=percent_remaining_60min,
        stability_class=stability_class,
        primary_degradation_pathway=primary_degradation_pathway,
        activation_energy_kJ_per_mol=_ACTIVATION_ENERGY_kJ_per_mol,
        q10_factor=q10_factor,
        notes=notes,
    )


def compare_matrices(
    drug_name: str,
    logp: float,
    mw_da: float,
    n_ester_bonds: int,
    n_amide_bonds: int,
    temperature_C: float = 37.0,
) -> list:
    """
    Compare drug stability across all 7 biological matrices.

    Returns
    -------
    list of MatrixDegradationResult sorted by t50_min descending (most stable first).
    """
    results = []
    for matrix in _MATRIX_BASE_RATES:
        result = predict_matrix_degradation(
            drug_name=drug_name,
            matrix=matrix,
            logp=logp,
            mw_da=mw_da,
            n_ester_bonds=n_ester_bonds,
            n_amide_bonds=n_amide_bonds,
            temperature_C=temperature_C,
        )
        results.append(result)

    results.sort(key=lambda r: r.t50_min, reverse=True)
    return results
