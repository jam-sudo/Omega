"""Formulation palatability scorer for pediatric and geriatric patients (Phase 273).

Scores drug formulation palatability based on taste category, masking strategy,
and formulation form. Provides patient acceptance estimates for children and elderly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Taste categories: (base_score, masking_effectiveness)
_TASTE_CATEGORIES: dict = {
    "bitter": {"base_score": 20, "masking_effectiveness": 0.6},
    "sweet": {"base_score": 90, "masking_effectiveness": 1.0},
    "sour": {"base_score": 40, "masking_effectiveness": 0.7},
    "salty": {"base_score": 60, "masking_effectiveness": 0.8},
    "tasteless": {"base_score": 95, "masking_effectiveness": 1.0},
}

# Masking strategies: masking_factor (lower = better masking)
_MASKING_STRATEGIES: dict = {
    "none": {"masking_factor": 1.0},
    "sweetener": {"masking_factor": 0.8},
    "flavor": {"masking_factor": 0.6},
    "cyclodextrin": {"masking_factor": 0.5},
    "coating": {"masking_factor": 0.3},
}

_VALID_FORMULATION_FORMS = {"liquid", "tablet", "capsule", "chewable", "powder"}


@dataclass(frozen=True)
class PalatabilityResult:
    """Result of palatability scoring for a drug formulation."""

    drug_name: str
    taste_category: str
    masking_strategy: str
    formulation_form: str
    base_palatability_score: float
    masked_palatability_score: float
    patient_acceptance_pct: float
    bitterness_index: float
    recommended_for_children: bool
    recommended_for_elderly: bool
    formulation_feasibility: str
    notes: str


def _score_to_acceptance(score: float) -> float:
    """Map palatability score to patient acceptance percentage."""
    if score < 40.0:
        return 20.0
    elif score <= 70.0:
        return 60.0
    else:
        return 90.0


def _compute_feasibility(score: float) -> str:
    """Determine formulation feasibility from masked score."""
    if score >= 50.0:
        return "feasible"
    elif score >= 30.0:
        return "challenging"
    else:
        return "not_recommended"


def _compute_bitterness_index(taste_category: str, base_score: float) -> float:
    """Bitterness index: 100 - base_score, 0 for sweet/tasteless."""
    if taste_category in ("sweet", "tasteless"):
        return 0.0
    return 100.0 - base_score


def _build_notes(
    taste_category: str,
    masking_strategy: str,
    masked_score: float,
    recommended_for_children: bool,
    recommended_for_elderly: bool,
) -> str:
    """Generate descriptive notes for the palatability result."""
    parts = []
    if taste_category == "bitter":
        if masking_strategy == "none":
            parts.append("Bitter taste with no masking — poor palatability expected.")
        else:
            parts.append(f"Bitter taste masked with {masking_strategy}.")
    elif taste_category == "sweet":
        parts.append("Sweet taste — naturally palatable.")
    elif taste_category == "sour":
        parts.append("Sour taste — masking may improve acceptance.")
    elif taste_category == "salty":
        parts.append("Salty taste — moderate palatability.")
    elif taste_category == "tasteless":
        parts.append("Tasteless — high natural palatability.")

    if recommended_for_children and recommended_for_elderly:
        parts.append("Suitable for both children and elderly.")
    elif recommended_for_children:
        parts.append("Suitable for children; marginal for elderly.")
    elif recommended_for_elderly:
        parts.append("Suitable for elderly; not recommended for children.")
    else:
        parts.append(
            "Not recommended for pediatric or geriatric patients without further optimization."
        )

    return " ".join(parts)


def score_palatability(
    drug_name: str,
    taste_category: str,
    masking_strategy: str = "none",
    formulation_form: str = "liquid",
) -> PalatabilityResult:
    """Score drug formulation palatability.

    Args:
        drug_name: Name of the drug.
        taste_category: Taste of the drug ("bitter", "sweet", "sour", "salty", "tasteless").
        masking_strategy: Strategy to mask taste ("none", "sweetener", "flavor",
                          "cyclodextrin", "coating").
        formulation_form: Dosage form ("liquid", "tablet", "capsule", "chewable", "powder").

    Returns:
        PalatabilityResult with palatability scores and patient suitability assessment.

    Raises:
        ValueError: If taste_category, masking_strategy, or formulation_form is invalid.
    """
    taste_category_lower = taste_category.lower()
    masking_strategy_lower = masking_strategy.lower()
    formulation_form_lower = formulation_form.lower()

    if taste_category_lower not in _TASTE_CATEGORIES:
        valid = ", ".join(sorted(_TASTE_CATEGORIES.keys()))
        raise ValueError(f"Invalid taste_category '{taste_category}'. Valid options: {valid}")
    if masking_strategy_lower not in _MASKING_STRATEGIES:
        valid = ", ".join(sorted(_MASKING_STRATEGIES.keys()))
        raise ValueError(f"Invalid masking_strategy '{masking_strategy}'. Valid options: {valid}")
    if formulation_form_lower not in _VALID_FORMULATION_FORMS:
        valid = ", ".join(sorted(_VALID_FORMULATION_FORMS))
        raise ValueError(f"Invalid formulation_form '{formulation_form}'. Valid options: {valid}")

    taste_info = _TASTE_CATEGORIES[taste_category_lower]
    masking_info = _MASKING_STRATEGIES[masking_strategy_lower]

    base_score = float(taste_info["base_score"])
    masking_factor = masking_info["masking_factor"]

    # masked_score = base_score + (100 - base_score) * (1 - masking_factor)
    # When masking_factor=1.0 (no masking), masked_score = base_score
    # When masking_factor=0.0 (perfect masking), masked_score = 100
    masked_score = base_score + (100.0 - base_score) * (1.0 - masking_factor)

    patient_acceptance_pct = _score_to_acceptance(masked_score)
    bitterness_index = _compute_bitterness_index(taste_category_lower, base_score)
    recommended_for_children = masked_score >= 70.0
    recommended_for_elderly = masked_score >= 60.0
    formulation_feasibility = _compute_feasibility(masked_score)

    notes = _build_notes(
        taste_category_lower,
        masking_strategy_lower,
        masked_score,
        recommended_for_children,
        recommended_for_elderly,
    )

    return PalatabilityResult(
        drug_name=drug_name,
        taste_category=taste_category_lower,
        masking_strategy=masking_strategy_lower,
        formulation_form=formulation_form_lower,
        base_palatability_score=base_score,
        masked_palatability_score=masked_score,
        patient_acceptance_pct=patient_acceptance_pct,
        bitterness_index=bitterness_index,
        recommended_for_children=recommended_for_children,
        recommended_for_elderly=recommended_for_elderly,
        formulation_feasibility=formulation_feasibility,
        notes=notes,
    )


def optimize_palatability(drug_name: str, taste_category: str) -> list:
    """Try all masking strategies and return results sorted by masked score descending.

    Args:
        drug_name: Name of the drug.
        taste_category: Taste of the drug.

    Returns:
        List of PalatabilityResult for all masking strategies, sorted by
        masked_palatability_score descending.
    """
    results = []
    for strategy in _MASKING_STRATEGIES:
        result = score_palatability(
            drug_name=drug_name,
            taste_category=taste_category,
            masking_strategy=strategy,
            formulation_form="liquid",
        )
        results.append(result)

    results.sort(key=lambda r: r.masked_palatability_score, reverse=True)
    return results
