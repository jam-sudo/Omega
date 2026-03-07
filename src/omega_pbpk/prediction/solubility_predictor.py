"""Phase 665 — Aqueous solubility prediction using the General Solubility Equation (GSE).

Yalkowsky SH, He Y (2003). Handbook of Aqueous Solubility Data.
GSE: log S = 0.5 - 0.01*(MP - 25) - logP  (Yalkowsky 2001)
Extended with PSA (polar solvation) and MW (size penalty) corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SolubilityPrediction:
    compound_name: str
    logP: float
    mw: float
    psa: float
    predicted_log_s: float
    predicted_solubility_mg_mL: float
    predicted_solubility_ug_mL: float
    solubility_class: str
    bcs_solubility: str
    aqueous_solubility_flag: bool
    ionization_correction_applied: bool
    dissolution_rate_estimate: str
    notes: str


def predict_solubility(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    melting_point_c: float = 150.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    pH: float = 7.4,
    dose_mg: float = 100.0,
    dose_volume_mL: float = 250.0,
) -> SolubilityPrediction:
    """Predict aqueous solubility using the General Solubility Equation with corrections.

    Args:
        compound_name: Name of the compound.
        logP: Octanol-water partition coefficient (log).
        mw: Molecular weight (g/mol).
        psa: Polar surface area (Angstrom^2).
        melting_point_c: Melting point in degrees Celsius (default 150).
        pka_acid: Acidic pKa, if applicable.
        pka_base: Basic pKa, if applicable.
        pH: Target pH for solubility assessment (default 7.4).
        dose_mg: Dose in mg for BCS assessment (default 100).
        dose_volume_mL: Volume of dissolution medium in mL (default 250).

    Returns:
        SolubilityPrediction dataclass with solubility estimates and classification.

    Raises:
        ValueError: If mw <= 0, psa < 0, or pH outside [0, 14].
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if not (0 <= pH <= 14):
        raise ValueError(f"pH must be in [0, 14], got {pH}")

    # GSE base: log S (mol/L)
    log_s = 0.5 - 0.01 * (melting_point_c - 25.0) - logP

    # PSA correction: polar groups improve solvation
    psa_correction = psa * 0.008
    log_s += psa_correction

    # MW correction: large molecules less soluble
    mw_correction = -(mw - 300.0) * 0.002 if mw > 300.0 else 0.0
    log_s += mw_correction

    # Ionization correction at target pH (Henderson-Hasselbalch)
    ionization_applied = False
    if pka_acid is not None and abs(pH - pka_acid) < 3.0:
        ionization_factor = 1.0 + 10.0 ** (pH - pka_acid)
        log_s += math.log10(ionization_factor)
        ionization_applied = True
    elif pka_base is not None and abs(pH - pka_base) < 3.0:
        ionization_factor = 1.0 + 10.0 ** (pka_base - pH)
        log_s += math.log10(ionization_factor)
        ionization_applied = True

    sol_mol_per_L = 10.0**log_s
    sol_mg_mL = sol_mol_per_L * mw / 1000.0
    sol_ug_mL = sol_mg_mL * 1000.0

    # Solubility class (mg/mL)
    if sol_mg_mL < 0.01:
        solubility_class = "very_low"
    elif sol_mg_mL < 0.1:
        solubility_class = "low"
    elif sol_mg_mL < 1.0:
        solubility_class = "moderate"
    else:
        solubility_class = "high"

    # BCS solubility criterion: high if dose dissolves in dose_volume_mL
    bcs_solubility = "high" if sol_mg_mL * dose_volume_mL >= dose_mg else "low"

    # Aqueous solubility flag: potential issue if < 0.1 mg/mL
    aqueous_solubility_flag = sol_mg_mL < 0.1

    # Dissolution rate estimate
    if sol_mg_mL > 1.0:
        dissolution_rate = "fast"
    elif sol_mg_mL >= 0.1:
        dissolution_rate = "moderate"
    else:
        dissolution_rate = "slow"

    # Build notes
    note_parts = [
        f"logS={log_s:.2f} mol/L",
        f"Solubility={sol_mg_mL:.4f} mg/mL ({solubility_class})",
        f"BCS solubility: {bcs_solubility}",
        f"Dissolution: {dissolution_rate}",
    ]
    if ionization_applied:
        note_parts.append(f"Ionization correction applied at pH {pH:.1f}")
    if aqueous_solubility_flag:
        note_parts.append("Flag: low aqueous solubility may limit absorption")

    notes = "; ".join(note_parts)

    return SolubilityPrediction(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        predicted_log_s=log_s,
        predicted_solubility_mg_mL=sol_mg_mL,
        predicted_solubility_ug_mL=sol_ug_mL,
        solubility_class=solubility_class,
        bcs_solubility=bcs_solubility,
        aqueous_solubility_flag=aqueous_solubility_flag,
        ionization_correction_applied=ionization_applied,
        dissolution_rate_estimate=dissolution_rate,
        notes=notes,
    )


def screen_solubility(compounds: list[dict]) -> list[SolubilityPrediction]:
    """Screen multiple compounds for aqueous solubility, sorted by solubility descending.

    Each dict should contain keys matching predict_solubility arguments.
    Required keys: 'compound_name', 'logP', 'mw', 'psa'.
    Optional keys: 'melting_point_c', 'pka_acid', 'pka_base', 'pH', 'dose_mg', 'dose_volume_mL'.

    Returns:
        List of SolubilityPrediction sorted by predicted_solubility_mg_mL descending.
    """
    if not compounds:
        return []

    results: list[SolubilityPrediction] = []
    for c in compounds:
        pred = predict_solubility(
            compound_name=c["compound_name"],
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            melting_point_c=c.get("melting_point_c", 150.0),
            pka_acid=c.get("pka_acid", None),
            pka_base=c.get("pka_base", None),
            pH=c.get("pH", 7.4),
            dose_mg=c.get("dose_mg", 100.0),
            dose_volume_mL=c.get("dose_volume_mL", 250.0),
        )
        results.append(pred)

    results.sort(key=lambda r: r.predicted_solubility_mg_mL, reverse=True)
    return results
