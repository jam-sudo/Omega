"""
Phase 257: Pediatric Formulation Selector

Selects appropriate oral formulations for pediatric patients based on age
and swallowing ability. Pure Python (stdlib: math, dataclasses only).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Age group lookup table
# ---------------------------------------------------------------------------

_AGE_GROUPS = [
    {
        "name": "neonate",
        "age_range_months": (0, 1),
        "max_volume_mL": 1.0,
        "preferred_forms": ["solution", "suspension"],
    },
    {
        "name": "infant",
        "age_range_months": (1, 12),
        "max_volume_mL": 5.0,
        "preferred_forms": ["solution", "suspension", "drops"],
    },
    {
        "name": "toddler",
        "age_range_months": (12, 36),
        "max_volume_mL": 10.0,
        "preferred_forms": ["suspension", "chewable", "sprinkle"],
    },
    {
        "name": "child",
        "age_range_months": (36, 144),
        "max_volume_mL": 20.0,
        "preferred_forms": ["tablet_small", "chewable", "sprinkle", "syrup"],
    },
    {
        "name": "adolescent",
        "age_range_months": (144, 216),
        "max_volume_mL": 30.0,
        "preferred_forms": ["tablet", "capsule", "solution"],
    },
]

# Representative weights (kg) and midpoint ages (months) per group
_GROUP_REPR = {
    "neonate": (3.5, 0.5),
    "infant": (7.0, 6.5),
    "toddler": (12.0, 24.0),
    "child": (20.0, 90.0),
    "adolescent": (55.0, 180.0),
}

# Palatability scores (0-100, higher = better taste)
_PALATABILITY = {
    "solution": 60,
    "suspension": 65,
    "drops": 70,
    "chewable": 85,
    "sprinkle": 75,
    "tablet": 50,
    "capsule": 40,
    "tablet_small": 60,
    "syrup": 80,
}

# Solid forms (volume = 0, concentration = 0)
_SOLID_FORMS = {"tablet", "capsule", "tablet_small", "chewable", "sprinkle"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PediatricFormulationResult:
    drug_name: str
    age_months: float
    weight_kg: float
    age_group: str
    dose_mg: float
    recommended_formulation: str
    volume_mL: float
    concentration_mg_mL: float
    palatability_score: int
    age_appropriate: bool
    dose_accuracy_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_age_group(age_months: float):
    """Return the age group dict for the given age in months."""
    for group in _AGE_GROUPS:
        lo, hi = group["age_range_months"]
        if lo <= age_months < hi:
            return group
    # If age_months >= 216 treat as adolescent; < 0 caught earlier
    return _AGE_GROUPS[-1]


def _dose_accuracy_risk(volume_mL: float) -> str:
    """Classify dose accuracy risk by volume."""
    if volume_mL == 0.0:
        # Solid form — accuracy determined by tablet/capsule strength, low risk
        return "low"
    if volume_mL < 0.5:
        return "high"
    if volume_mL <= 2.0:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_pediatric_formulation(
    drug_name: str,
    age_months: float,
    weight_kg: float,
    dose_per_kg_mg: float,
    drug_solubility_mg_mL: float = 10.0,
) -> PediatricFormulationResult:
    """Select the most appropriate oral formulation for a pediatric patient.

    Parameters
    ----------
    drug_name : str
    age_months : float  Patient age in months (> 0).
    weight_kg : float   Patient weight in kg (> 0).
    dose_per_kg_mg : float  Dose per kg body weight (mg/kg, > 0).
    drug_solubility_mg_mL : float  Maximum drug solubility (default 10 mg/mL).

    Returns
    -------
    PediatricFormulationResult
    """
    if age_months <= 0:
        raise ValueError("age_months must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if dose_per_kg_mg <= 0:
        raise ValueError("dose_per_kg_mg must be > 0")

    dose_mg = dose_per_kg_mg * weight_kg

    group = _get_age_group(age_months)
    group_name = group["name"]
    max_vol = group["max_volume_mL"]
    preferred = group["preferred_forms"]

    # Pick the first preferred form
    chosen_form = preferred[0]

    # Compute volume and concentration
    notes_parts = []
    if chosen_form in _SOLID_FORMS:
        volume_mL = 0.0
        concentration_mg_mL = 0.0
    else:
        # Liquid: aim for concentration <= solubility, volume <= max_volume
        concentration_mg_mL = min(drug_solubility_mg_mL, dose_mg / max_vol)
        volume_mL = dose_mg / concentration_mg_mL
        if volume_mL > max_vol:
            # Saturate concentration so volume == max_vol
            concentration_mg_mL = dose_mg / max_vol
            volume_mL = max_vol
            notes_parts.append("Volume capped at max for age group; consider splitting dose.")

    palatability = _PALATABILITY.get(chosen_form, 50)
    risk = _dose_accuracy_risk(volume_mL)

    # Age appropriateness: chosen form is in the preferred list
    age_appropriate = chosen_form in preferred

    if not notes_parts:
        notes_parts.append(
            f"{chosen_form} selected for {group_name} ({age_months:.1f} mo, {weight_kg:.1f} kg)."
        )

    return PediatricFormulationResult(
        drug_name=drug_name,
        age_months=age_months,
        weight_kg=weight_kg,
        age_group=group_name,
        dose_mg=dose_mg,
        recommended_formulation=chosen_form,
        volume_mL=volume_mL,
        concentration_mg_mL=concentration_mg_mL,
        palatability_score=palatability,
        age_appropriate=age_appropriate,
        dose_accuracy_risk=risk,
        notes=" ".join(notes_parts),
    )


def compare_age_groups_for_drug(
    drug_name: str,
    dose_per_kg_mg: float,
    drug_solubility_mg_mL: float = 10.0,
) -> list[PediatricFormulationResult]:
    """Compare formulation recommendations across all pediatric age groups.

    Uses representative weight and midpoint age for each group.
    Results are sorted by palatability_score descending.

    Parameters
    ----------
    drug_name : str
    dose_per_kg_mg : float  Dose per kg (mg/kg, > 0).
    drug_solubility_mg_mL : float  Maximum solubility (default 10 mg/mL).

    Returns
    -------
    List[PediatricFormulationResult]  5 entries, sorted by palatability desc.
    """
    results = []
    for group in _AGE_GROUPS:
        name = group["name"]
        weight, age = _GROUP_REPR[name]
        result = select_pediatric_formulation(
            drug_name=drug_name,
            age_months=age,
            weight_kg=weight,
            dose_per_kg_mg=dose_per_kg_mg,
            drug_solubility_mg_mL=drug_solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.palatability_score, reverse=True)
    return results
