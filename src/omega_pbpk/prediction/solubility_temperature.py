"""Drug Aqueous Solubility Temperature Dependence — Phase 1062.

Predicts how drug solubility changes with temperature using the van't Hoff
equation.  Important for manufacturing, storage, and IV formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_R_J_MOL_K = 8.314  # J / (mol · K)
_T_REF_K = 298.15  # 25 °C in kelvin

_VALID_IONIZATION_TYPES = {"neutral", "acid", "base"}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolubilityTemperatureResult:
    drug_name: str
    temperature_C: float
    solubility_mg_mL: float
    relative_to_25C: float
    dissolution_heat_kJ_mol: float
    enthalpy_driven: bool
    temperature_sensitivity: str
    predicted_crystal_habit: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dissolution_heat(logp: float, melting_point_C: float) -> float:
    """Estimate dissolution enthalpy (kJ/mol) from logP and melting point."""
    if logp > 3:
        dh = 50.0
    elif logp > 1:
        dh = 30.0
    else:
        dh = 10.0
    if melting_point_C > 200.0:
        dh += 20.0
    return dh


def _solubility_at_25(
    mw_Da: float,
    logp: float,
    pka: float,
    ionization_type: str,
    ph: float,
) -> float:
    """Intrinsic solubility at 25 °C (mg/mL), Henderson-Hasselbalch corrected."""
    s0 = 10.0 ** (-0.5 * logp) * 50.0 / mw_Da
    s0 = max(0.001, min(1000.0, s0))

    if ionization_type == "acid":
        s_ph = s0 * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base":
        s_ph = s0 * (1.0 + 10.0 ** (pka - ph))
    else:
        s_ph = s0

    return s_ph


def _crystal_habit(temperature_C: float, logp: float) -> str:
    if temperature_C < 20.0 and logp > 2.0:
        return "needle"
    if temperature_C > 40.0:
        return "prismatic"
    return "plate"


def _temperature_sensitivity(relative: float) -> str:
    deviation = abs(relative - 1.0)
    if deviation < 0.1:
        return "low"
    if deviation < 0.5:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    mw_Da: float,
    melting_point_C: float,
    temperature_C: float,
    ionization_type: str,
    pka: float,
) -> None:
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if melting_point_C <= 0:
        raise ValueError("melting_point_C must be > 0")
    if not (-20.0 <= temperature_C <= 120.0):
        raise ValueError("temperature_C must be in [-20, 120]")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError("pka must be in [0, 14]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_solubility_temperature(
    drug_name: str,
    mw_Da: float,
    logp: float,
    melting_point_C: float,
    temperature_C: float = 25.0,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    ph: float = 7.0,
) -> SolubilityTemperatureResult:
    """Predict aqueous drug solubility at a given temperature.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    mw_Da:
        Molecular weight (Da).
    logp:
        Octanol-water partition coefficient (log scale).
    melting_point_C:
        Melting point (°C); used for lattice-energy correction on ΔH_soln.
    temperature_C:
        Target temperature (°C), must be in [-20, 120].
    pka:
        Acid/base dissociation constant.
    ionization_type:
        "neutral", "acid", or "base".
    ph:
        pH of the dissolution medium.

    Returns
    -------
    SolubilityTemperatureResult (frozen dataclass)
    """
    _validate_inputs(mw_Da, melting_point_C, temperature_C, ionization_type, pka)

    dh_kj_mol = _dissolution_heat(logp, melting_point_C)

    s_25 = _solubility_at_25(mw_Da, logp, pka, ionization_type, ph)

    # Van't Hoff
    t_k = temperature_C + 273.15
    exponent = (dh_kj_mol * 1000.0 / _R_J_MOL_K) * (1.0 / _T_REF_K - 1.0 / t_k)
    s_t = s_25 * math.exp(exponent)
    s_t = max(1e-6, min(10000.0, s_t))

    relative = s_t / s_25

    enthalpy_driven = dh_kj_mol < 0
    sensitivity = _temperature_sensitivity(relative)
    habit = _crystal_habit(temperature_C, logp)

    notes = (
        f"Van't Hoff solubility prediction at {temperature_C:.1f} °C; "
        f"ΔH_soln = {dh_kj_mol:.1f} kJ/mol; "
        f"S(25°C) = {s_25:.4f} mg/mL; "
        f"ionization: {ionization_type} (pKa={pka}, pH={ph})."
    )

    return SolubilityTemperatureResult(
        drug_name=drug_name,
        temperature_C=temperature_C,
        solubility_mg_mL=s_t,
        relative_to_25C=relative,
        dissolution_heat_kJ_mol=dh_kj_mol,
        enthalpy_driven=enthalpy_driven,
        temperature_sensitivity=sensitivity,
        predicted_crystal_habit=habit,
        notes=notes,
    )


def solubility_temperature_profile(
    drug_name: str,
    mw_Da: float,
    logp: float,
    melting_point_C: float,
    temperatures_C: list,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    ph: float = 7.0,
) -> list:
    """Compute solubility predictions across a list of temperatures.

    Parameters
    ----------
    temperatures_C:
        List of temperatures (°C) to evaluate.

    Returns
    -------
    list[SolubilityTemperatureResult] in the same order as *temperatures_C*.
    """
    return [
        predict_solubility_temperature(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            melting_point_C=melting_point_C,
            temperature_C=t,
            pka=pka,
            ionization_type=ionization_type,
            ph=ph,
        )
        for t in temperatures_C
    ]
