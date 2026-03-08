"""Phase 471 — Drug Interaction with Food Components.

Models specific drug-food interactions beyond simple food effect, including
grapefruit juice (CYP3A4 inhibition), dairy chelation, high-fiber absorption
reduction, alcohol effects, and tyramine/MAOI interactions.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugFoodInteractionResult",
    "predict_drug_food_interaction",
    "screen_food_interactions",
]

# Valid food items
VALID_FOOD_ITEMS = {
    "grapefruit",
    "dairy",
    "high_fiber",
    "alcohol_acute",
    "alcohol_chronic",
    "tyramine_foods",
}

# Food interaction database
FOOD_INTERACTION_DB: dict[str, dict] = {
    "grapefruit": {
        "cyp3a4_inhibition_factor": 3.0,
        "affected_drugs": ["CYP3A4_substrate"],
    },
    "dairy": {
        "chelation_fa_reduction": 0.5,
        "affected_drugs": ["fluoroquinolone", "tetracycline"],
    },
    "high_fiber": {
        "fa_reduction_factor": 0.8,
        "affected_drugs": ["all"],
    },
    "alcohol_acute": {
        "cyp2e1_inhibition": 0.5,
        "cns_additive": True,
        "affected_drugs": ["CYP2E1_substrate", "CNS_depressant"],
    },
    "alcohol_chronic": {
        "cyp2e1_induction": 2.0,
        "affected_drugs": ["CYP2E1_substrate"],
    },
    "tyramine_foods": {
        "mao_risk": True,
        "affected_drugs": ["MAOI"],
    },
}


@dataclass(frozen=True)
class DrugFoodInteractionResult:
    """Result of a drug-food interaction prediction."""

    drug_name: str
    food_item: str
    drug_class: str
    interaction_detected: bool
    auc_ratio: float
    fa_ratio: float
    clinical_severity: str
    mechanism: str
    recommendation: str
    notes: str


def predict_drug_food_interaction(
    drug_name: str,
    food_item: str,
    drug_class: str,
) -> DrugFoodInteractionResult:
    """Predict interaction between a drug and a food item.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    food_item:
        One of the 6 valid food items: grapefruit, dairy, high_fiber,
        alcohol_acute, alcohol_chronic, tyramine_foods.
    drug_class:
        Pharmacological/mechanistic class of the drug (e.g. "CYP3A4_substrate").

    Returns
    -------
    DrugFoodInteractionResult
    """
    if food_item not in VALID_FOOD_ITEMS:
        raise ValueError(
            f"Invalid food_item '{food_item}'. Must be one of: {sorted(VALID_FOOD_ITEMS)}"
        )
    if not drug_class or not drug_class.strip():
        raise ValueError("drug_class must be a non-empty string.")

    auc_ratio = 1.0
    fa_ratio = 1.0
    interaction_detected = False
    clinical_severity = "none"
    mechanism = "No known pharmacokinetic or pharmacodynamic interaction."
    recommendation = "no_action"
    notes = ""

    if food_item == "grapefruit":
        if drug_class == "CYP3A4_substrate":
            auc_ratio = 3.0
            interaction_detected = True
            clinical_severity = "major"
            mechanism = (
                "Grapefruit juice irreversibly inhibits intestinal CYP3A4, "
                "dramatically increasing bioavailability of CYP3A4 substrates."
            )
            recommendation = "avoid"
            notes = (
                "Even a single glass of grapefruit juice can inhibit CYP3A4 "
                "for up to 72 hours. Use alternative fruit juices."
            )

    elif food_item == "dairy":
        if drug_class in ("fluoroquinolone", "tetracycline"):
            fa_ratio = 0.5
            auc_ratio = 0.5
            interaction_detected = True
            clinical_severity = "moderate"
            mechanism = (
                "Divalent cations (Ca2+, Mg2+) in dairy products form insoluble "
                "chelate complexes with fluoroquinolones and tetracyclines, "
                "reducing gastrointestinal absorption."
            )
            recommendation = "separate_by_2h"
            notes = (
                "Take the drug at least 2 hours before or 4 hours after "
                "dairy products, antacids, or mineral supplements."
            )

    elif food_item == "high_fiber":
        fa_ratio = 0.8
        auc_ratio = 0.8
        interaction_detected = True
        clinical_severity = "minor"
        mechanism = (
            "High dietary fiber can bind drugs in the GI tract and reduce "
            "absorption rate and extent (bile acid sequestration effect)."
        )
        recommendation = "monitor"
        notes = (
            "Effect is generally minor. Monitor therapeutic response if "
            "patient has a very high-fiber diet."
        )

    elif food_item == "alcohol_acute":
        if drug_class == "CYP2E1_substrate":
            auc_ratio = 2.0
            interaction_detected = True
            clinical_severity = "moderate"
            mechanism = (
                "Acute alcohol competitively inhibits CYP2E1, reducing first-pass "
                "metabolism of CYP2E1 substrates and increasing systemic exposure."
            )
            recommendation = "avoid"
            notes = (
                "Avoid alcohol consumption while taking this medication. "
                "Even moderate alcohol can substantially increase drug levels."
            )
        elif drug_class == "CNS_depressant":
            auc_ratio = 1.0
            interaction_detected = True
            clinical_severity = "major"
            mechanism = (
                "Additive/synergistic CNS depression. No direct PK change, "
                "but pharmacodynamic potentiation of sedation, respiratory "
                "depression, and psychomotor impairment."
            )
            recommendation = "avoid"
            notes = (
                "Combination can cause dangerous over-sedation or respiratory "
                "depression. Alcohol is contraindicated with CNS depressants."
            )

    elif food_item == "alcohol_chronic":
        if drug_class == "CYP2E1_substrate":
            auc_ratio = 0.5
            interaction_detected = True
            clinical_severity = "moderate"
            mechanism = (
                "Chronic alcohol use induces CYP2E1, increasing first-pass "
                "metabolism of CYP2E1 substrates and reducing systemic exposure. "
                "May also increase formation of toxic metabolites."
            )
            recommendation = "monitor"
            notes = (
                "Chronic alcohol users may require higher doses, but induction "
                "also increases toxic metabolite formation (e.g., acetaminophen "
                "hepatotoxicity risk)."
            )

    elif food_item == "tyramine_foods":
        if drug_class == "MAOI":
            interaction_detected = True
            clinical_severity = "contraindicated"
            mechanism = (
                "MAO inhibitors block intestinal and hepatic MAO, preventing "
                "tyramine metabolism. Accumulated tyramine triggers massive "
                "norepinephrine release causing hypertensive crisis."
            )
            recommendation = "avoid"
            notes = (
                "Hypertensive crisis can be life-threatening. Foods high in "
                "tyramine (aged cheeses, cured meats, fermented products) are "
                "absolutely contraindicated with MAOIs. This restriction may "
                "persist 2 weeks after MAOI discontinuation."
            )

    return DrugFoodInteractionResult(
        drug_name=drug_name,
        food_item=food_item,
        drug_class=drug_class,
        interaction_detected=interaction_detected,
        auc_ratio=auc_ratio,
        fa_ratio=fa_ratio,
        clinical_severity=clinical_severity,
        mechanism=mechanism,
        recommendation=recommendation,
        notes=notes,
    )


def screen_food_interactions(
    drug_name: str,
    drug_class: str,
    food_items: list | None = None,
) -> list:
    """Screen a drug against multiple food items for interactions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_class:
        Pharmacological/mechanistic class of the drug.
    food_items:
        List of food items to screen. If None, screens all 6 valid items.

    Returns
    -------
    list[DrugFoodInteractionResult]
        Results sorted by auc_ratio descending (highest AUC increase first).
    """
    if food_items is None:
        items_to_screen = list(VALID_FOOD_ITEMS)
    else:
        items_to_screen = list(food_items)

    results = []
    for food_item in items_to_screen:
        result = predict_drug_food_interaction(drug_name, food_item, drug_class)
        results.append(result)

    # Sort by auc_ratio descending (highest AUC increase first)
    results.sort(key=lambda r: r.auc_ratio, reverse=True)
    return results
