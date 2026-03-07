"""Phase 319 — Amide Bond Stability Predictor.

Predict stability of amide bonds in drug molecules under acidic, basic,
and enzymatic conditions. Important for peptide drugs and amide-containing
small molecules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_CONDITIONS = {"acidic", "neutral", "basic", "enzymatic"}
_VALID_STERIC = {0, 1, 2, 3}
_EA_J_PER_MOL = 60_000.0  # 60 kJ/mol activation energy
_R_J_PER_MOL_K = 8.314
_T_REF_K = 310.15  # 37 degC


@dataclass(frozen=True)
class AmideStabilityResult:
    """Immutable result for a single amide bond stability prediction."""

    drug_name: str
    n_amide_bonds: int
    logp: float
    mw: float
    is_peptide: bool
    ph_condition: str
    temperature_C: float
    incubation_h: float
    effective_k_per_h: float
    stability_pct: float
    half_life_h: float
    stability_class: str
    notes: str


def _validate(
    n_amide_bonds: int,
    logp: float,
    mw: float,
    temperature_C: float,
    incubation_h: float,
    ph_condition: str,
    steric_protection_score: int,
) -> None:
    if n_amide_bonds < 1:
        raise ValueError("n_amide_bonds must be >= 1")
    if not (-5.0 < logp < 10.0):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (0.0 < temperature_C < 100.0):
        raise ValueError(f"temperature_C must be in (0, 100), got {temperature_C}")
    if incubation_h <= 0:
        raise ValueError("incubation_h must be > 0")
    if ph_condition not in _VALID_CONDITIONS:
        raise ValueError(f"ph_condition must be one of {_VALID_CONDITIONS}, got '{ph_condition}'")
    if steric_protection_score not in _VALID_STERIC:
        raise ValueError(
            f"steric_protection_score must be in {{0,1,2,3}}, got {steric_protection_score}"
        )


def _compute_k(
    n_amide_bonds: int,
    logp: float,
    is_peptide: bool,
    steric_protection_score: int,
    ph_condition: str,
    temperature_C: float,
) -> float:
    """Compute effective hydrolysis rate constant (per h)."""
    sps = steric_protection_score

    if ph_condition == "neutral":
        k_base = 0.001 * n_amide_bonds * max(0.1, 1.0 - sps * 0.2)
    elif ph_condition == "acidic":
        k_base = 0.005 * n_amide_bonds * max(0.1, 1.0 - sps * 0.15)
    elif ph_condition == "basic":
        k_base = 0.02 * n_amide_bonds * max(0.1, 1.0 - sps * 0.1)
    else:  # enzymatic
        k_base = 0.05 * n_amide_bonds * max(0.1, 1.0 - sps * 0.3)

    if is_peptide:
        k_base *= 2.0

    if logp > 2.0:
        k_base *= 0.7

    # Arrhenius temperature correction vs 37 degC
    temp_k = temperature_C + 273.15
    arrhenius = math.exp(_EA_J_PER_MOL / _R_J_PER_MOL_K * (1.0 / _T_REF_K - 1.0 / temp_k))
    k = k_base * arrhenius

    return k


def _stability_class(stability_pct: float) -> str:
    if stability_pct > 95.0:
        return "stable"
    if stability_pct >= 70.0:
        return "moderately_stable"
    return "labile"


def predict_amide_stability(
    drug_name: str,
    n_amide_bonds: int,
    logp: float,
    mw: float,
    is_peptide: bool,
    steric_protection_score: int,
    ph_condition: str,
    temperature_C: float,
    incubation_h: float,
) -> AmideStabilityResult:
    """Predict amide bond stability for a drug compound.

    Parameters
    ----------
    drug_name:
        Name / identifier of the drug.
    n_amide_bonds:
        Number of hydrolysable amide bonds (>= 1).
    logp:
        Octanol-water partition coefficient (-5 to 10).
    mw:
        Molecular weight in g/mol (> 0).
    is_peptide:
        True if the compound is a peptide (more susceptible).
    steric_protection_score:
        Integer 0-3; higher value means more sterically hindered amide bonds.
    ph_condition:
        One of "acidic", "neutral", "basic", "enzymatic".
    temperature_C:
        Incubation temperature in degrees Celsius (0-100 exclusive).
    incubation_h:
        Incubation duration in hours (> 0).

    Returns
    -------
    AmideStabilityResult
    """
    _validate(
        n_amide_bonds, logp, mw, temperature_C, incubation_h, ph_condition, steric_protection_score
    )

    k = _compute_k(
        n_amide_bonds, logp, is_peptide, steric_protection_score, ph_condition, temperature_C
    )

    stability_pct = max(0.0, 100.0 * math.exp(-k * incubation_h))
    half_life_h = 0.693 / max(k, 1e-10)
    cls = _stability_class(stability_pct)

    notes_parts = []
    if is_peptide:
        notes_parts.append("peptide substrate (2x susceptibility)")
    if logp > 2.0:
        notes_parts.append("lipophilic protection applied (0.7x)")
    if steric_protection_score >= 2:
        notes_parts.append(f"steric protection score={steric_protection_score}")
    notes = (
        "; ".join(notes_parts) if notes_parts else f"{ph_condition} condition at {temperature_C}C"
    )

    return AmideStabilityResult(
        drug_name=drug_name,
        n_amide_bonds=n_amide_bonds,
        logp=logp,
        mw=mw,
        is_peptide=is_peptide,
        ph_condition=ph_condition,
        temperature_C=temperature_C,
        incubation_h=incubation_h,
        effective_k_per_h=k,
        stability_pct=stability_pct,
        half_life_h=half_life_h,
        stability_class=cls,
        notes=notes,
    )


def compare_conditions(
    drug_name: str,
    n_amide_bonds: int,
    logp: float,
    mw: float,
    is_peptide: bool,
    steric_protection_score: int,
    incubation_h: float,
) -> list[AmideStabilityResult]:
    """Compare amide stability across all 4 pH/enzymatic conditions.

    Returns a list of AmideStabilityResult sorted by stability_pct descending
    (most stable condition first).
    """
    conditions = ["acidic", "neutral", "basic", "enzymatic"]
    results = [
        predict_amide_stability(
            drug_name=drug_name,
            n_amide_bonds=n_amide_bonds,
            logp=logp,
            mw=mw,
            is_peptide=is_peptide,
            steric_protection_score=steric_protection_score,
            ph_condition=cond,
            temperature_C=37.0,
            incubation_h=incubation_h,
        )
        for cond in conditions
    ]
    results.sort(key=lambda r: r.stability_pct, reverse=True)
    return results
