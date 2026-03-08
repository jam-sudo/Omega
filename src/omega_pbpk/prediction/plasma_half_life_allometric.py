"""Phase 1123 — Allometric scaling of plasma half-life across species.

Uses the empirical exponent of 0.25 for t½ scaling:
    t_half_human = t_half_species * (BW_human / BW_species) ** 0.25

Reference: Boxenbaum (1982), Mahmood & Balian (1996).
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed species body weights (kg)
_SPECIES_BW_KG: dict[str, float] = {
    "mouse": 0.02,
    "rat": 0.25,
    "dog": 11.0,
    "monkey": 5.0,
    "human": 70.0,
}

# Scaling exponent for t½ (Vd~BW^1.0, CL~BW^0.75 → t½~BW^0.25)
_EXPONENT = 0.25

# Confidence levels by source species
_CONFIDENCE: dict[str, str] = {
    "rat": "high",
    "dog": "moderate",
    "monkey": "moderate",
    "mouse": "low",
    "human": "high",
}


@dataclass(frozen=True)
class AllometricHalfLifeResult:
    """Result of allometric t½ scaling between two species.

    Attributes
    ----------
    drug_name : Drug identifier.
    source_species : Species from which t½ data was taken.
    source_t_half_h : Observed t½ in source species (h).
    source_bw_kg : Body weight of source species (kg).
    target_species : Species for which t½ is predicted.
    target_bw_kg : Body weight of target species (kg).
    predicted_t_half_h : Predicted t½ in target species (h).
    scaling_exponent : Allometric exponent used (0.25).
    fold_change : predicted_t_half_h / source_t_half_h.
    confidence : Prediction confidence ("high", "moderate", "low").
    notes : Summary note string.
    """

    drug_name: str
    source_species: str
    source_t_half_h: float
    source_bw_kg: float
    target_species: str
    target_bw_kg: float
    predicted_t_half_h: float
    scaling_exponent: float
    fold_change: float
    confidence: str
    notes: str


def predict_human_half_life(
    drug_name: str,
    t_half_species_h: float,
    species: str,
    body_weight_kg: float | None = None,
) -> AllometricHalfLifeResult:
    """Predict human plasma t½ from a single animal species.

    Parameters
    ----------
    drug_name : Drug identifier.
    t_half_species_h : Observed plasma t½ in *species* (h).  Must be > 0.
    species : Source species; one of {"mouse", "rat", "dog", "monkey", "human"}.
    body_weight_kg : Override body weight for source species (kg).  Optional;
        defaults to the standard value for the named species.

    Returns
    -------
    AllometricHalfLifeResult
    """
    if t_half_species_h <= 0:
        raise ValueError("t_half_species_h must be > 0")
    species = species.lower()
    if species not in _SPECIES_BW_KG:
        raise ValueError(f"species must be one of {set(_SPECIES_BW_KG.keys())}, got '{species}'")

    bw_source = body_weight_kg if body_weight_kg is not None else _SPECIES_BW_KG[species]
    if bw_source <= 0:
        raise ValueError("body_weight_kg must be > 0")

    bw_human = _SPECIES_BW_KG["human"]
    predicted = t_half_species_h * (bw_human / bw_source) ** _EXPONENT
    fold_change = predicted / t_half_species_h

    confidence = _CONFIDENCE.get(species, "low")
    notes = (
        f"t_half_human = {t_half_species_h:.3f}h * "
        f"({bw_human:.1f}/{bw_source:.3f})^{_EXPONENT} = {predicted:.3f}h"
    )

    return AllometricHalfLifeResult(
        drug_name=drug_name,
        source_species=species,
        source_t_half_h=t_half_species_h,
        source_bw_kg=bw_source,
        target_species="human",
        target_bw_kg=bw_human,
        predicted_t_half_h=predicted,
        scaling_exponent=_EXPONENT,
        fold_change=fold_change,
        confidence=confidence,
        notes=notes,
    )


def scale_across_species(
    drug_name: str,
    t_half_rat_h: float,
) -> list[AllometricHalfLifeResult]:
    """Scale rat t½ to all five species, sorted by body weight ascending.

    Uses rat body weight (0.25 kg) as the reference and scales to each target
    species using the 0.25 allometric exponent.

    Parameters
    ----------
    drug_name : Drug identifier.
    t_half_rat_h : Plasma t½ in rat (h). Must be > 0.

    Returns
    -------
    list[AllometricHalfLifeResult]
        One result per species, sorted by target body weight ascending.
    """
    if t_half_rat_h <= 0:
        raise ValueError("t_half_rat_h must be > 0")

    bw_rat = _SPECIES_BW_KG["rat"]
    results: list[AllometricHalfLifeResult] = []

    for target_species, bw_target in _SPECIES_BW_KG.items():
        predicted = t_half_rat_h * (bw_target / bw_rat) ** _EXPONENT
        fold_change = predicted / t_half_rat_h
        confidence = _CONFIDENCE.get(target_species, "low")
        notes = (
            f"rat t_half={t_half_rat_h:.3f}h → {target_species} "
            f"(BW={bw_target}kg): predicted {predicted:.3f}h"
        )
        results.append(
            AllometricHalfLifeResult(
                drug_name=drug_name,
                source_species="rat",
                source_t_half_h=t_half_rat_h,
                source_bw_kg=bw_rat,
                target_species=target_species,
                target_bw_kg=bw_target,
                predicted_t_half_h=predicted,
                scaling_exponent=_EXPONENT,
                fold_change=fold_change,
                confidence=confidence,
                notes=notes,
            )
        )

    # Sort by target body weight ascending
    results.sort(key=lambda r: r.target_bw_kg)
    return results


__all__ = [
    "AllometricHalfLifeResult",
    "predict_human_half_life",
    "scale_across_species",
]
