"""
Phase 219 — Esterase Lability Predictor

Predict ester bond lability to plasma, liver, and intestinal esterases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base plasma half-lives (minutes at 37°C)
_ESTER_BASE_T_HALF: dict[str, float] = {
    "methyl_ester": 5.0,
    "ethyl_ester": 15.0,
    "tert_butyl_ester": 240.0,
    "phenyl_ester": 2.0,
    "carbamate": 480.0,
    "carbonate": 10.0,
}

_STERIC_FACTORS: dict[str, float] = {
    "none": 1.0,
    "moderate": 3.0,
    "high": 10.0,
}

_ELECTRONIC_FACTORS: dict[str, float] = {
    "electron_withdrawing": 0.3,
    "neutral": 1.0,
    "electron_donating": 2.0,
}

# Liver hydrolysis is ~5× faster than plasma
_LIVER_SPEED_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EsteraseLabilityResult:
    drug_name: str
    ester_type: str
    steric_bulk: str
    electronic_effect: str
    t_half_plasma_min: float
    t_half_liver_min: float
    k_plasma_per_min: float
    stability_class: str
    prodrug_suitability: str  # "excellent" / "good" / "poor"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stability_class(t_half_plasma: float) -> str:
    if t_half_plasma < 10.0:
        return "very_labile"
    if t_half_plasma < 60.0:
        return "labile"
    if t_half_plasma <= 300.0:
        return "moderate"
    return "stable"


def _prodrug_suitability(stability_class: str) -> str:
    """Labile esters make good prodrugs (rapid activation after absorption)."""
    if stability_class == "very_labile":
        return "excellent"
    if stability_class == "labile":
        return "good"
    return "poor"


def _notes(ester_type: str, stability_class: str, prodrug_suitability: str) -> str:
    parts = [
        f"{ester_type.replace('_', ' ').title()} with {stability_class} plasma stability.",
        f"Prodrug suitability: {prodrug_suitability}.",
    ]
    if stability_class in ("very_labile", "labile"):
        parts.append("Rapid systemic hydrolysis expected after oral absorption.")
    elif stability_class == "stable":
        parts.append("High metabolic stability; may require enzymatic activation.")
    return " ".join(parts)


def _validate_inputs(ester_type: str, steric_bulk: str, electronic_effect: str) -> None:
    if ester_type not in _ESTER_BASE_T_HALF:
        valid = sorted(_ESTER_BASE_T_HALF.keys())
        raise ValueError(f"Invalid ester_type '{ester_type}'. Valid options: {valid}")
    if steric_bulk not in _STERIC_FACTORS:
        valid = sorted(_STERIC_FACTORS.keys())
        raise ValueError(f"Invalid steric_bulk '{steric_bulk}'. Valid options: {valid}")
    if electronic_effect not in _ELECTRONIC_FACTORS:
        valid = sorted(_ELECTRONIC_FACTORS.keys())
        raise ValueError(f"Invalid electronic_effect '{electronic_effect}'. Valid options: {valid}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_esterase_lability(
    drug_name: str,
    ester_type: str,
    steric_bulk: str = "none",
    electronic_effect: str = "neutral",
) -> EsteraseLabilityResult:
    """Predict esterase lability for a given ester bond.

    Parameters
    ----------
    drug_name:
        Name of the drug or prodrug.
    ester_type:
        One of "methyl_ester", "ethyl_ester", "tert_butyl_ester",
        "phenyl_ester", "carbamate", "carbonate".
    steric_bulk:
        Steric bulk around the ester bond: "none", "moderate", or "high".
    electronic_effect:
        Electronic environment: "electron_withdrawing", "neutral", or
        "electron_donating".

    Returns
    -------
    EsteraseLabilityResult
    """
    _validate_inputs(ester_type, steric_bulk, electronic_effect)

    base = _ESTER_BASE_T_HALF[ester_type]
    steric_factor = _STERIC_FACTORS[steric_bulk]
    electronic_factor = _ELECTRONIC_FACTORS[electronic_effect]

    t_half_plasma = base * steric_factor * electronic_factor
    t_half_liver = t_half_plasma / _LIVER_SPEED_FACTOR
    k_plasma = math.log(2.0) / t_half_plasma

    sc = _stability_class(t_half_plasma)
    ps = _prodrug_suitability(sc)
    note = _notes(ester_type, sc, ps)

    return EsteraseLabilityResult(
        drug_name=drug_name,
        ester_type=ester_type,
        steric_bulk=steric_bulk,
        electronic_effect=electronic_effect,
        t_half_plasma_min=t_half_plasma,
        t_half_liver_min=t_half_liver,
        k_plasma_per_min=k_plasma,
        stability_class=sc,
        prodrug_suitability=ps,
        notes=note,
    )


def compare_ester_types(
    drug_name: str,
    steric_bulk: str = "none",
    electronic_effect: str = "neutral",
) -> list[EsteraseLabilityResult]:
    """Compare all 6 ester types for the same drug scaffold.

    Returns a list of :class:`EsteraseLabilityResult` objects sorted by
    *t_half_plasma_min* in descending order (most stable first).
    """
    results = [
        predict_esterase_lability(drug_name, ester_type, steric_bulk, electronic_effect)
        for ester_type in _ESTER_BASE_T_HALF
    ]
    results.sort(key=lambda r: r.t_half_plasma_min, reverse=True)
    return results
