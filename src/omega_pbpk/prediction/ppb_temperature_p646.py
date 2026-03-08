"""Phase 646 — Protein Binding Temperature Dependence.

Models the effect of temperature on plasma protein binding (fup) using
the Van't Hoff equation. Relevant for cryopreservation studies, fever
modeling, and ex vivo assays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_R_J_MOL_K = 8.314  # universal gas constant (J/mol/K)


@dataclass(frozen=True)
class PPBTemperatureResult:
    """Result from protein binding temperature prediction."""

    drug_name: str
    logP: float
    reference_fup: float
    reference_temp_C: float
    temp_C: float
    fup_at_temp: float
    delta_fup_pct: float
    binding_enthalpy_kJ_mol: float
    van_t_hoff_factor: float
    protein_conformation_factor: float
    clinical_relevance: str
    notes: str


def _validate_inputs(
    reference_fup: float,
    mw_Da: float,
) -> None:
    if not (0.0 < reference_fup < 1.0):
        raise ValueError(f"reference_fup must be in (0, 1), got {reference_fup}")
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")


def _estimate_enthalpy(logP: float) -> float:
    """Estimate binding enthalpy (kJ/mol) from logP."""
    delta_h = -20.0 - logP * 2.5
    return max(-60.0, min(-5.0, delta_h))


def _protein_conformation_factor(temp_C: float) -> float:
    """Protein conformation stability factor by temperature range."""
    if temp_C < 20.0:
        return 0.85
    if temp_C > 40.0:
        return 0.90
    return 1.0


def predict_ppb_temperature(
    drug_name: str,
    logP: float,
    mw_Da: float,
    reference_fup: float,
    reference_temp_C: float = 37.0,
    target_temp_C: float = 25.0,
) -> PPBTemperatureResult:
    """Predict plasma protein binding at a target temperature.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Log partition coefficient.
    mw_Da:
        Molecular weight (Da).
    reference_fup:
        Fraction unbound at reference temperature (0, 1).
    reference_temp_C:
        Reference temperature (Celsius). Default 37.0.
    target_temp_C:
        Target temperature (Celsius).

    Returns
    -------
    PPBTemperatureResult
    """
    _validate_inputs(reference_fup, mw_Da)

    delta_h = _estimate_enthalpy(logP)

    # Temperatures in Kelvin
    t1_k = reference_temp_C + 273.15
    t2_k = target_temp_C + 273.15

    # Equilibrium binding constant at reference temp
    k1 = (1.0 - reference_fup) / reference_fup

    # Van't Hoff: ln(K2/K1) = -delta_H / R * (1/T2 - 1/T1)
    exponent = -delta_h * 1000.0 / _R_J_MOL_K * (1.0 / t2_k - 1.0 / t1_k)
    k2 = k1 * math.exp(exponent)

    # fup at target temperature
    fup_at_temp = 1.0 / (1.0 + k2)
    fup_at_temp = max(0.001, min(0.999, fup_at_temp))

    delta_fup_pct = (fup_at_temp - reference_fup) / reference_fup * 100.0

    van_t_hoff_factor = k2 / k1

    conf_factor = _protein_conformation_factor(target_temp_C)

    abs_delta = abs(delta_fup_pct)
    if abs_delta > 20.0:
        clinical_relevance = "significant"
    elif abs_delta > 10.0:
        clinical_relevance = "moderate"
    else:
        clinical_relevance = "minor"

    notes = (
        f"Van't Hoff: dH={delta_h:.1f} kJ/mol; "
        f"K1={k1:.3f} (at {reference_temp_C:.0f}C); "
        f"K2={k2:.3f} (at {target_temp_C:.0f}C); "
        f"MW={mw_Da:.1f} Da; conformation_factor={conf_factor:.2f}"
    )

    return PPBTemperatureResult(
        drug_name=drug_name,
        logP=logP,
        reference_fup=reference_fup,
        reference_temp_C=reference_temp_C,
        temp_C=target_temp_C,
        fup_at_temp=fup_at_temp,
        delta_fup_pct=delta_fup_pct,
        binding_enthalpy_kJ_mol=delta_h,
        van_t_hoff_factor=van_t_hoff_factor,
        protein_conformation_factor=conf_factor,
        clinical_relevance=clinical_relevance,
        notes=notes,
    )


def compare_temperatures(
    drug_name: str,
    logP: float,
    mw_Da: float,
    reference_fup: float,
    temperatures: list[float],
) -> list[PPBTemperatureResult]:
    """Compare protein binding across multiple temperatures.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Log partition coefficient.
    mw_Da:
        Molecular weight (Da).
    reference_fup:
        Fraction unbound at reference temperature (0, 1).
    temperatures:
        List of target temperatures (Celsius).

    Returns
    -------
    List of PPBTemperatureResult sorted by temperature ascending.
    """
    results = [
        predict_ppb_temperature(
            drug_name=drug_name,
            logP=logP,
            mw_Da=mw_Da,
            reference_fup=reference_fup,
            target_temp_C=t,
        )
        for t in temperatures
    ]
    results.sort(key=lambda r: r.temp_C)
    return results
