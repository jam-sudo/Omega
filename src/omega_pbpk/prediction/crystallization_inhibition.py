"""
Phase 1027 — Drug Crystallization Inhibition

Predict how excipients/polymers inhibit drug crystallization from supersaturated
solutions using an Avrami-type crystallization model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_POLYMERS = {"HPMC", "PVP", "PVA", "Eudragit", "none"}

_POLYMER_PARAMS: dict[str, dict] = {
    "HPMC": {"inhibition_factor": 0.8, "mechanism": "adsorption", "conc_exponent": 0.4},
    "PVP": {"inhibition_factor": 0.7, "mechanism": "adsorption", "conc_exponent": 0.5},
    "PVA": {"inhibition_factor": 0.85, "mechanism": "viscosity", "conc_exponent": 0.3},
    "Eudragit": {"inhibition_factor": 0.6, "mechanism": "complexation", "conc_exponent": 0.6},
    "none": {"inhibition_factor": 0.0, "mechanism": "mixed", "conc_exponent": 0.0},
}


@dataclass(frozen=True)
class CrystallizationInhibitionResult:
    drug_name: str
    polymer: str
    supersaturation_ratio: float
    induction_time_h: float
    crystallization_rate_per_h: float
    fraction_crystallized_at_1h: float
    fraction_crystallized_at_4h: float
    inhibition_efficiency: float
    mechanism: str
    notes: str


def _fraction_crystallized(k_cryst: float, induction_time_h: float, t: float) -> float:
    """Avrami-type fraction crystallized at time t."""
    dt = t - induction_time_h
    if dt <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-k_cryst * dt)))


def predict_crystallization_inhibition(
    drug_name: str,
    logp: float,
    mw_Da: float,
    supersaturation_ratio: float,
    polymer: str = "HPMC",
    polymer_conc_mg_mL: float = 1.0,
    temperature_C: float = 37.0,
) -> CrystallizationInhibitionResult:
    """
    Predict polymer inhibition of drug crystallization from supersaturated solution.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons. Must be > 0.
    supersaturation_ratio : float
        C / Cs (concentration / solubility). Must be > 1.0.
    polymer : str
        Polymer identifier: "HPMC", "PVP", "PVA", "Eudragit", or "none".
    polymer_conc_mg_mL : float
        Polymer concentration in solution (mg/mL). Must be >= 0.
    temperature_C : float
        Temperature in Celsius.

    Returns
    -------
    CrystallizationInhibitionResult
    """
    # --- Validation ---
    if supersaturation_ratio <= 1.0:
        raise ValueError(f"supersaturation_ratio must be > 1.0, got {supersaturation_ratio}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if polymer_conc_mg_mL < 0:
        raise ValueError(f"polymer_conc_mg_mL must be >= 0, got {polymer_conc_mg_mL}")
    if polymer not in _VALID_POLYMERS:
        raise ValueError(f"polymer must be one of {_VALID_POLYMERS}, got '{polymer}'")

    sr = supersaturation_ratio

    # --- Reference (no polymer) rates ---
    k_ref = 0.1 * sr**2 * math.exp(-logp * 0.1)
    induction_time_ref = 1.0 / (sr * 0.5)

    # --- Polymer inhibition ---
    params = _POLYMER_PARAMS[polymer]
    inhibition_factor = params["inhibition_factor"]
    mechanism = params["mechanism"]
    conc_exponent = params["conc_exponent"]

    if inhibition_factor == 0.0 or conc_exponent == 0.0:
        effective_inhibition = 0.0
    else:
        effective_inhibition = inhibition_factor * (polymer_conc_mg_mL**conc_exponent)

    effective_inhibition = min(0.99, effective_inhibition)

    k_cryst = k_ref * (1.0 - effective_inhibition)
    induction_time_h = induction_time_ref / (1.0 - effective_inhibition + 0.01)

    # --- Temperature correction (Arrhenius-like) ---
    temp_factor = math.exp(0.05 * (temperature_C - 37.0))
    k_cryst *= temp_factor
    induction_time_h /= temp_factor

    # --- Fraction crystallized ---
    f_1h = _fraction_crystallized(k_cryst, induction_time_h, 1.0)
    f_4h = _fraction_crystallized(k_cryst, induction_time_h, 4.0)

    inhibition_efficiency = effective_inhibition

    # --- Notes ---
    notes = (
        f"Drug: {drug_name}; Polymer: {polymer} @ {polymer_conc_mg_mL} mg/mL; "
        f"k_cryst={k_cryst:.4f}/h; induction={induction_time_h:.2f}h; "
        f"eff_inhibition={inhibition_efficiency:.3f}"
    )

    return CrystallizationInhibitionResult(
        drug_name=drug_name,
        polymer=polymer,
        supersaturation_ratio=sr,
        induction_time_h=induction_time_h,
        crystallization_rate_per_h=k_cryst,
        fraction_crystallized_at_1h=f_1h,
        fraction_crystallized_at_4h=f_4h,
        inhibition_efficiency=inhibition_efficiency,
        mechanism=mechanism,
        notes=notes,
    )
