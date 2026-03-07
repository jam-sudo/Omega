"""
Phase 896 — Adipocyte Cellular Drug Uptake

Predicts drug uptake into individual adipocytes (cellular-level model).
Relevant for adipokine drug interactions and lipid-soluble toxins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdipocyteUptakeResult:
    """Result of adipocyte uptake prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    lipid_droplet_kp: float
    membrane_permeability_cm_s: float
    intracellular_conc_factor: float
    uptake_rate_constant_per_min: float
    equilibration_time_min: float
    adipocyte_accumulation_class: str
    bioaccumulation_risk: bool
    notes: str


_VALID_IONIZATION_TYPES = {"neutral", "acid", "base"}


def predict_adipocyte_uptake(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 350.0,
    pka: float | None = None,
    ionization_type: str = "neutral",
    charge_at_ph7: int = 0,
) -> AdipocyteUptakeResult:
    """
    Predict drug uptake into adipocytes.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Lipophilicity (log octanol-water partition coefficient).
    mw_Da : float
        Molecular weight in Daltons.
    pka : float or None
        Ionization constant (optional).
    ionization_type : str
        One of "neutral", "acid", "base".
    charge_at_ph7 : int
        Net charge at pH 7.

    Returns
    -------
    AdipocyteUptakeResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )

    # --- Lipid droplet partition coefficient ---
    if logp > 0:
        raw_kp = math.pow(10.0, logp * 0.7)
    else:
        raw_kp = 1.0
    lipid_droplet_kp = max(1.0, min(1e4, raw_kp))

    # --- Membrane permeability ---
    raw_papp_exp = -6.0 + logp * 0.5 - mw_Da / 1000.0
    raw_papp = math.pow(10.0, raw_papp_exp)
    membrane_permeability_cm_s = max(1e-8, min(1e-4, raw_papp))

    # --- Uptake rate constant (per min) ---
    # SA/V ratio simplified: k_up = Papp * 60 * 1e4 / 10
    raw_k_up = membrane_permeability_cm_s * 60.0 * 1e4 / 10.0
    uptake_rate_constant_per_min = max(0.001, min(10.0, raw_k_up))

    # --- Intracellular concentration factor ---
    # neutral or cation: lipid_droplet_kp * 0.3 (30% lipid volume)
    raw_icf = lipid_droplet_kp * 0.3
    intracellular_conc_factor = max(1.0, min(500.0, raw_icf))

    # --- Equilibration time (min) ---
    # Time to 90% equilibrium: t90 = ln(10) / k_up
    raw_t_eq = math.log(10.0) / uptake_rate_constant_per_min
    equilibration_time_min = max(0.1, min(1440.0, raw_t_eq))

    # --- Classification ---
    if intracellular_conc_factor > 50:
        adipocyte_accumulation_class = "extensive"
    elif intracellular_conc_factor > 10:
        adipocyte_accumulation_class = "moderate"
    else:
        adipocyte_accumulation_class = "minimal"

    bioaccumulation_risk = intracellular_conc_factor > 20 and logp > 3

    # --- Notes ---
    if bioaccumulation_risk:
        notes = (
            "Bioaccumulation risk in adipose tissue — monitor for tissue toxicity in chronic dosing"
        )
    elif adipocyte_accumulation_class == "extensive":
        notes = "Extensive adipocyte accumulation — prolonged tissue retention expected"
    else:
        notes = "Acceptable adipocyte uptake profile"

    return AdipocyteUptakeResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        lipid_droplet_kp=lipid_droplet_kp,
        membrane_permeability_cm_s=membrane_permeability_cm_s,
        intracellular_conc_factor=intracellular_conc_factor,
        uptake_rate_constant_per_min=uptake_rate_constant_per_min,
        equilibration_time_min=equilibration_time_min,
        adipocyte_accumulation_class=adipocyte_accumulation_class,
        bioaccumulation_risk=bioaccumulation_risk,
        notes=notes,
    )


def screen_adipocyte_uptake(
    candidates: list[dict[str, Any]],
) -> list[AdipocyteUptakeResult]:
    """
    Screen a list of drug candidates for adipocyte uptake.

    Parameters
    ----------
    candidates : list of dict
        Each dict must have "name" and "logp"; optional "mw_Da" (default 350)
        and "ionization_type" (default "neutral").

    Returns
    -------
    list of AdipocyteUptakeResult sorted by intracellular_conc_factor descending
    """
    results: list[AdipocyteUptakeResult] = []
    for cand in candidates:
        name = cand["name"]
        logp = float(cand["logp"])
        mw_Da = float(cand.get("mw_Da", 350.0))
        ionization_type = cand.get("ionization_type", "neutral")
        result = predict_adipocyte_uptake(
            drug_name=name,
            logp=logp,
            mw_Da=mw_Da,
            ionization_type=ionization_type,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.intracellular_conc_factor, reverse=True)
