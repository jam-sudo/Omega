"""Drug solubility prediction — thermodynamic and kinetic models."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SolubilityResult", "predict_solubility", "solubility_ph_profile"]


def _general_solubility_yalkowsky(logP: float, mp_celsius: float = 150.0) -> float:
    """Yalkowsky GSE: log10(S_mg_mL) = 0.5 - logP - 0.01 * max(mp - 25, 0)."""
    log_s = 0.5 - logP - 0.01 * max(mp_celsius - 25.0, 0.0)
    s = 10.0 ** log_s
    return max(0.0001, min(s, 1000.0))


def _ph_adjusted_solubility(S0: float, pka: float, drug_type: str, pH: float) -> float:
    """Henderson-Hasselbalch pH-adjusted solubility."""
    if drug_type == "acid":
        s = S0 * (1.0 + 10.0 ** (pH - pka))
    elif drug_type == "base":
        s = S0 * (1.0 + 10.0 ** (pka - pH))
    else:
        s = S0
    return max(0.001, min(s, 10000.0))


def _supersaturation_index(sol_intrinsic: float, sol_amorphous_factor: float = 1.5) -> float:
    """Amorphous form supersaturation index."""
    return sol_intrinsic * sol_amorphous_factor


@dataclass(frozen=True)
class SolubilityResult:
    drug_name: str
    logP: float
    mp_celsius: float
    pka: float
    drug_type: str
    sol_intrinsic_mg_mL: float
    sol_ph_stomach_mg_mL: float
    sol_ph_intestine_mg_mL: float
    sol_ph_colon_mg_mL: float
    supersaturation_potential: float
    bcs_solubility_class: str
    dose_number: float
    warnings: list[str]


def predict_solubility(
    drug_name: str,
    logP: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    mp_celsius: float = 150.0,
    dose_mg: float = 100.0,
    amorphous_factor: float = 1.5,
) -> SolubilityResult:
    """Predict drug solubility across GI tract pH values."""
    S0 = _general_solubility_yalkowsky(logP, mp_celsius)
    sol_stomach = _ph_adjusted_solubility(S0, pka, drug_type, 2.0)
    sol_intestine = _ph_adjusted_solubility(S0, pka, drug_type, 6.5)
    sol_colon = _ph_adjusted_solubility(S0, pka, drug_type, 7.4)
    supersaturation_potential = _supersaturation_index(sol_intestine, amorphous_factor)
    bcs_solubility_class = "high" if sol_intestine >= 0.1 else "low"
    dose_number = dose_mg / (sol_intestine * 250.0)

    warnings: list[str] = []
    if dose_number > 1.0:
        warnings.append("Low solubility (Do>1): dissolution may be rate-limiting")
    if S0 < 0.01:
        warnings.append("Very low solubility: formulation strategies recommended")

    return SolubilityResult(
        drug_name=drug_name,
        logP=logP,
        mp_celsius=mp_celsius,
        pka=pka,
        drug_type=drug_type,
        sol_intrinsic_mg_mL=S0,
        sol_ph_stomach_mg_mL=sol_stomach,
        sol_ph_intestine_mg_mL=sol_intestine,
        sol_ph_colon_mg_mL=sol_colon,
        supersaturation_potential=supersaturation_potential,
        bcs_solubility_class=bcs_solubility_class,
        dose_number=dose_number,
        warnings=warnings,
    )


def solubility_ph_profile(
    drug_name: str,
    logP: float,
    pka: float,
    drug_type: str = "neutral",
    mp_celsius: float = 150.0,
    ph_range: list[float] | None = None,
) -> list[dict]:
    """Return solubility at each pH in the given range."""
    if ph_range is None:
        ph_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.4, 8.0]
    S0 = _general_solubility_yalkowsky(logP, mp_celsius)
    return [
        {"pH": ph, "solubility_mg_mL": _ph_adjusted_solubility(S0, pka, drug_type, ph)}
        for ph in ph_range
    ]
