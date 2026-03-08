"""Phase 479 — Microbiome Drug Metabolism Prediction.

Models the impact of gut microbiome on drug metabolism and bioavailability.
"""

from __future__ import annotations

from dataclasses import dataclass

# Microbiome activity factors by condition
_ACTIVITY_FACTORS: dict[str, float] = {
    "normal": 1.0,
    "dysbiosis": 0.4,
    "antibiotic_treated": 0.05,
    "enhanced": 1.5,
}

_VALID_INTERACTION_TYPES = {
    "glucuronide_hydrolysis",
    "azo_reduction",
    "nitroreduction",
    "inert",
}

_VALID_CONDITIONS = set(_ACTIVITY_FACTORS.keys())


@dataclass(frozen=True)
class MicrobiomeDrugMetabolismResult:
    """Result of microbiome drug metabolism prediction."""

    drug_name: str
    interaction_type: str
    microbiome_condition: str
    activity_factor: float
    fa_multiplier: float
    base_fa: float
    adjusted_fa: float
    dose_absorbed_mg: float
    systemic_exposure_ratio: float
    clinical_impact: str
    notes: str


def _compute_fa_multiplier(interaction_type: str, activity_factor: float) -> float:
    """Compute the fraction absorbed multiplier based on interaction type."""
    if interaction_type == "glucuronide_hydrolysis":
        return 1.0 + 0.3 * activity_factor
    elif interaction_type == "azo_reduction":
        return activity_factor
    elif interaction_type == "nitroreduction":
        return 0.5 + 0.5 * activity_factor
    else:  # inert
        return 1.0


def _classify_clinical_impact(systemic_exposure_ratio: float) -> str:
    """Classify clinical impact based on systemic exposure ratio."""
    pct_change = abs(systemic_exposure_ratio - 1.0) * 100.0
    if pct_change < 5.0:
        return "none"
    elif pct_change < 20.0:
        return "minor"
    elif pct_change < 50.0:
        return "significant"
    else:
        return "major"


def _build_notes(
    interaction_type: str,
    microbiome_condition: str,
    activity_factor: float,
    clinical_impact: str,
) -> str:
    """Build a descriptive notes string."""
    desc_map = {
        "glucuronide_hydrolysis": "Gut flora cleave glucuronide conjugates, increasing parent drug.",
        "azo_reduction": "Bacteria reduce prodrug to active form; activation requires viable flora.",
        "nitroreduction": "Bacterial nitroreduction partially activates drug.",
        "inert": "No significant microbiome interaction.",
    }
    cond_map = {
        "normal": "normal healthy microbiome",
        "dysbiosis": "dysbiotic reduced-diversity microbiome",
        "antibiotic_treated": "antibiotic-treated minimal-flora state",
        "enhanced": "probiotic-enhanced microbiome",
    }
    base = desc_map.get(interaction_type, "")
    cond = cond_map.get(microbiome_condition, microbiome_condition)
    return (
        f"{base} Condition: {cond} (activity factor={activity_factor:.2f}). "
        f"Clinical impact: {clinical_impact}."
    )


def predict_microbiome_metabolism(
    drug_name: str,
    interaction_type: str,
    microbiome_condition: str,
    dose_mg: float,
    base_fa: float = 0.7,
) -> MicrobiomeDrugMetabolismResult:
    """Predict microbiome impact on drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    interaction_type:
        One of "glucuronide_hydrolysis", "azo_reduction", "nitroreduction", "inert".
    microbiome_condition:
        One of "normal", "dysbiosis", "antibiotic_treated", "enhanced".
    dose_mg:
        Administered dose in mg (must be > 0).
    base_fa:
        Baseline fraction absorbed without microbiome modulation (0 < base_fa <= 1).

    Returns
    -------
    MicrobiomeDrugMetabolismResult
    """
    if interaction_type not in _VALID_INTERACTION_TYPES:
        raise ValueError(
            f"interaction_type must be one of {_VALID_INTERACTION_TYPES}, got {interaction_type!r}"
        )
    if microbiome_condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"microbiome_condition must be one of {_VALID_CONDITIONS}, got {microbiome_condition!r}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < base_fa <= 1):
        raise ValueError(f"base_fa must be in (0, 1], got {base_fa}")

    activity_factor = _ACTIVITY_FACTORS[microbiome_condition]
    fa_multiplier = _compute_fa_multiplier(interaction_type, activity_factor)

    # Clamp adjusted_fa to [0, 1]
    adjusted_fa = max(0.0, min(1.0, base_fa * fa_multiplier))
    dose_absorbed_mg = dose_mg * adjusted_fa
    systemic_exposure_ratio = adjusted_fa / base_fa
    clinical_impact = _classify_clinical_impact(systemic_exposure_ratio)
    notes = _build_notes(interaction_type, microbiome_condition, activity_factor, clinical_impact)

    return MicrobiomeDrugMetabolismResult(
        drug_name=drug_name,
        interaction_type=interaction_type,
        microbiome_condition=microbiome_condition,
        activity_factor=activity_factor,
        fa_multiplier=fa_multiplier,
        base_fa=base_fa,
        adjusted_fa=adjusted_fa,
        dose_absorbed_mg=dose_absorbed_mg,
        systemic_exposure_ratio=systemic_exposure_ratio,
        clinical_impact=clinical_impact,
        notes=notes,
    )


def compare_microbiome_conditions(
    drug_name: str,
    interaction_type: str,
    dose_mg: float,
    base_fa: float = 0.7,
) -> list[MicrobiomeDrugMetabolismResult]:
    """Screen all 4 microbiome conditions, sorted by adjusted_fa descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    interaction_type:
        One of "glucuronide_hydrolysis", "azo_reduction", "nitroreduction", "inert".
    dose_mg:
        Administered dose in mg.
    base_fa:
        Baseline fraction absorbed.

    Returns
    -------
    list[MicrobiomeDrugMetabolismResult]
        All 4 conditions sorted by adjusted_fa descending.
    """
    results = [
        predict_microbiome_metabolism(drug_name, interaction_type, condition, dose_mg, base_fa)
        for condition in _VALID_CONDITIONS
    ]
    return sorted(results, key=lambda r: r.adjusted_fa, reverse=True)
