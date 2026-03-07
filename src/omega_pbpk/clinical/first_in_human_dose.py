"""Phase 362 — First-in-Human Dose Prediction.

Predict first-in-human (FIH) starting dose using multiple methods
consistent with FDA Guidance 2005.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Km factors (body surface area normalisation)
_KM_FACTORS: dict[str, float] = {
    "mouse": 3.0,
    "rat": 6.0,
    "monkey": 12.0,
    "dog": 20.0,
    "human": 37.0,
}

_HUMAN_BODY_WEIGHT_KG: float = 70.0
_DEFAULT_SF: float = 10.0
_TOX_SF: float = 50.0
_MABEL_DIVISOR: float = 10.0
_MTED_DIVISOR: float = 6.0

_RISK_CATEGORIES: dict[str, str] = {
    "low": "low",
    "moderate": "moderate",
    "high": "high",
}


@dataclass
class FIHDosePredictionResult:
    """Result of first-in-human dose prediction."""

    drug_name: str
    animal_species: str
    noael_mg_per_kg: float
    hed_mg_per_kg: float
    hed_mg: float
    safety_factor: float
    mrsd_mg: float
    mrsd_limiting_factor: str
    dose_escalation_mg: list = field(default_factory=list)
    risk_category: str = ""
    notes: str = ""


_VALID_ANIMAL_SPECIES: frozenset[str] = frozenset(k for k in _KM_FACTORS if k != "human")


def _validate_inputs(noael_mg_per_kg: float, animal_species: str) -> None:
    if noael_mg_per_kg <= 0.0:
        raise ValueError(f"noael_mg_per_kg must be > 0; got {noael_mg_per_kg}")
    if animal_species not in _VALID_ANIMAL_SPECIES:
        valid = ", ".join(sorted(_VALID_ANIMAL_SPECIES))
        raise ValueError(f"animal_species must be one of {valid}; got '{animal_species}'")


def predict_fih_dose(
    drug_name: str,
    noael_mg_per_kg_animal: float,
    animal_species: str,
    hsa_correction: bool = True,
    mabel_mg: float | None = None,
    mted_mg: float | None = None,
    target_organ_toxicity: bool = False,
) -> FIHDosePredictionResult:
    """Predict first-in-human starting dose.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    noael_mg_per_kg_animal:
        No-observed-adverse-effect level from animal study (mg/kg).
    animal_species:
        Species used: "mouse", "rat", "monkey", or "dog".
    hsa_correction:
        Human safety adjustment flag (default True; included in SF logic).
    mabel_mg:
        Minimum anticipated biological effect level in mg for 70 kg human.
        If provided, MRSD = min(MRSD, MABEL/10).
    mted_mg:
        Minimum toxic exposure dose in mg for 70 kg human.
        If provided, MRSD = min(MRSD, MTED/6).
    target_organ_toxicity:
        If True, SF is raised to 50 (high concern).

    Returns
    -------
    FIHDosePredictionResult
    """
    _validate_inputs(noael_mg_per_kg_animal, animal_species)

    km_animal = _KM_FACTORS[animal_species]
    km_human = _KM_FACTORS["human"]

    # Human equivalent dose
    hed_mg_per_kg = noael_mg_per_kg_animal * km_animal / km_human
    hed_mg = hed_mg_per_kg * _HUMAN_BODY_WEIGHT_KG

    # Safety factor
    if target_organ_toxicity:
        sf = _TOX_SF
    else:
        sf = _DEFAULT_SF

    # hsa_correction: already conservative by design; no further multiplication
    # (matches FDA guidance: SF=10 is already conservative for hsa)

    mrsd_from_hed = hed_mg / sf
    mrsd = mrsd_from_hed
    limiting_factor = "HED"

    if mabel_mg is not None:
        mabel_limit = mabel_mg / _MABEL_DIVISOR
        if mabel_limit < mrsd:
            mrsd = mabel_limit
            limiting_factor = "MABEL"

    if mted_mg is not None:
        mted_limit = mted_mg / _MTED_DIVISOR
        if mted_limit < mrsd:
            mrsd = mted_limit
            limiting_factor = "MTED"

    # Dose escalation scheme
    escalation_multipliers = [1.0, 3.0, 9.0, 20.0]
    dose_escalation_mg = [mrsd * m for m in escalation_multipliers]

    # Risk category based on safety factor
    if sf <= 10.0:
        risk_category = "low"
    elif sf <= 20.0:
        risk_category = "moderate"
    else:
        risk_category = "high"

    # Notes
    notes_parts: list[str] = []
    notes_parts.append(f"HED = {hed_mg:.2f} mg (Km {animal_species}={km_animal}, human={km_human})")
    notes_parts.append(f"SF = {sf:.0f}")
    if mabel_mg is not None:
        notes_parts.append(f"MABEL constraint = {mabel_mg / _MABEL_DIVISOR:.2f} mg")
    if mted_mg is not None:
        notes_parts.append(f"MTED constraint = {mted_mg / _MTED_DIVISOR:.2f} mg")
    notes_parts.append(f"limiting factor: {limiting_factor}")
    notes = "; ".join(notes_parts)

    return FIHDosePredictionResult(
        drug_name=drug_name,
        animal_species=animal_species,
        noael_mg_per_kg=noael_mg_per_kg_animal,
        hed_mg_per_kg=round(hed_mg_per_kg, 6),
        hed_mg=round(hed_mg, 4),
        safety_factor=sf,
        mrsd_mg=round(mrsd, 6),
        mrsd_limiting_factor=limiting_factor,
        dose_escalation_mg=[round(d, 6) for d in dose_escalation_mg],
        risk_category=risk_category,
        notes=notes,
    )
