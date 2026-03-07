"""Drug-food interaction modeling: quantitative PK impact of food on absorption.

This module provides quantitative PK food effect calculations distinct from
the existing food_drug_interaction.py qualitative assessments.
Follows FDA food effect guidance (2019).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FoodEffectResult", "assess_food_effect", "high_fat_meal_impact"]

_VALID_BCS = {"I", "II", "III", "IV"}
_VALID_FOOD_CONDITIONS = {"fasted", "fed_high_fat", "fed_low_fat"}
_VALID_FORMULATIONS = {"IR", "ER", "enteric_coated"}


@dataclass(frozen=True)
class FoodEffectResult:
    """Result of a drug-food interaction assessment."""

    drug_name: str
    bcs_class: str
    food_condition: str
    cmax_ratio_fed_fasted: float
    auc_ratio_fed_fasted: float
    tmax_delay_h: float
    food_effect_category: str
    clinical_relevance: str
    recommendation: str
    notes: str


def assess_food_effect(
    drug_name: str,
    bcs_class: str,
    logP: float,
    pka: float = 7.0,
    solubility_mg_mL: float = 1.0,
    dose_mg: float = 100.0,
    formulation: str = "IR",
    food_condition: str = "fed_high_fat",
) -> FoodEffectResult:
    """Assess quantitative food effect on drug PK based on BCS class and properties.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    bcs_class:
        BCS classification: "I", "II", "III", or "IV".
    logP:
        Octanol-water partition coefficient (log scale).
    pka:
        Drug pKa (used for solubility adjustments, reserved for extensions).
    solubility_mg_mL:
        Aqueous solubility (mg/mL).
    dose_mg:
        Dose in mg.
    formulation:
        Formulation type: "IR", "ER", or "enteric_coated".
    food_condition:
        Food condition being compared to fasted: "fasted", "fed_high_fat", "fed_low_fat".

    Returns
    -------
    FoodEffectResult
    """
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"Invalid BCS class '{bcs_class}'. Must be one of: {sorted(_VALID_BCS)}")
    if food_condition not in _VALID_FOOD_CONDITIONS:
        valid = sorted(_VALID_FOOD_CONDITIONS)
        raise ValueError(f"Invalid food_condition '{food_condition}'. Must be one of: {valid}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"Invalid formulation '{formulation}'. Must be one of: {sorted(_VALID_FORMULATIONS)}"
        )

    # Fasted vs fasted: ratios are 1.0, no delay
    if food_condition == "fasted":
        auc_ratio = 1.0
        cmax_ratio = 1.0
        tmax_delay = 0.0
    else:
        # FDA-based AUC ratio model
        if bcs_class == "I":
            # Minimal food effect; slight positive if lipophilic
            auc_ratio = 1.0 + 0.05 * logP / 5.0
        elif bcs_class == "II":
            # Lipophilic drugs benefit from fat: bile-mediated solubilization
            auc_ratio = (
                1.0
                + 0.3 * min(logP, 5.0) / 5.0
                + (0.1 if food_condition == "fed_high_fat" else 0.0)
            )
        elif bcs_class == "III":
            # Reduced motility generally reduces absorption
            auc_ratio = 0.85
        else:  # BCS IV
            # Variable; slight benefit if somewhat lipophilic
            auc_ratio = 0.9 + 0.2 * min(logP, 3.0) / 3.0

        # Cmax: food delays absorption more than it affects AUC
        cmax_ratio = auc_ratio * (0.8 if food_condition == "fed_high_fat" else 0.9)

        # Tmax delay relative to fasted
        tmax_delay = 1.0 if food_condition == "fed_high_fat" else 0.5

    # Enteric-coated formulations have inherent delay; blunt food effect on tmax
    if formulation == "enteric_coated" and food_condition != "fasted":
        tmax_delay = max(tmax_delay, 2.0)

    # Determine food effect category based on AUC ratio
    food_effect = auc_ratio - 1.0
    if abs(food_effect) < 0.1:
        food_effect_category = "neutral"
    elif food_effect > 0:
        food_effect_category = "positive"
    else:
        food_effect_category = "negative"

    # Clinical relevance: FDA threshold is >20% change
    max_change = max(abs(auc_ratio - 1.0), abs(cmax_ratio - 1.0))
    if max_change < 0.1:
        clinical_relevance = "none"
    elif max_change < 0.2:
        clinical_relevance = "low"
    elif max_change < 0.5:
        clinical_relevance = "moderate"
    else:
        clinical_relevance = "high"

    # Recommendation
    if food_effect_category == "positive" and clinical_relevance in ("moderate", "high"):
        recommendation = "Take with food"
    elif food_effect_category == "negative" and clinical_relevance in ("moderate", "high"):
        recommendation = "Take on empty stomach"
    else:
        recommendation = "May be taken with or without food"

    notes = (
        f"BCS {bcs_class} drug (logP={logP:.1f}); "
        f"AUC ratio (fed/fasted)={auc_ratio:.2f}, "
        f"Cmax ratio={cmax_ratio:.2f}; "
        f"formulation={formulation}."
    )

    return FoodEffectResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        food_condition=food_condition,
        cmax_ratio_fed_fasted=round(cmax_ratio, 4),
        auc_ratio_fed_fasted=round(auc_ratio, 4),
        tmax_delay_h=round(tmax_delay, 2),
        food_effect_category=food_effect_category,
        clinical_relevance=clinical_relevance,
        recommendation=recommendation,
        notes=notes,
    )


def high_fat_meal_impact(
    drug_name: str,
    bcs_class: str,
    logP: float,
    **kwargs,
) -> list[FoodEffectResult]:
    """Return food effect results for all three food conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    bcs_class:
        BCS classification: "I", "II", "III", or "IV".
    logP:
        Octanol-water partition coefficient (log scale).
    **kwargs:
        Additional keyword arguments forwarded to `assess_food_effect`
        (except `food_condition`).

    Returns
    -------
    list[FoodEffectResult]
        Results for "fasted", "fed_low_fat", and "fed_high_fat".
    """
    return [
        assess_food_effect(drug_name, bcs_class, logP, food_condition=cond, **kwargs)
        for cond in ("fasted", "fed_low_fat", "fed_high_fat")
    ]
