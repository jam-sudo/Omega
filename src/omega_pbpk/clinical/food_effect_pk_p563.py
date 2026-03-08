"""Phase 563 - Drug Interaction with Food (Food Effect PK).

Models the effect of food on drug pharmacokinetics based on BCS classification.
"""

from dataclasses import dataclass

VALID_BCS_CLASSES = {"I", "II", "III", "IV"}


@dataclass(frozen=True)
class FoodEffectResult:
    """Result of food effect prediction."""

    drug_name: str
    bcs_class: str
    food_state: str
    cmax_ratio: float
    auc_ratio: float
    tmax_ratio: float
    ka_fed: float
    f_fed: float
    food_effect_class: str
    clinical_recommendation: str
    notes: str


def _classify_food_effect(auc_ratio: float) -> str:
    if auc_ratio > 1.25:
        return "positive"
    if auc_ratio < 0.8:
        return "negative"
    return "none"


def _clinical_recommendation(food_effect_class: str) -> str:
    if food_effect_class == "positive":
        return "take_with_food"
    if food_effect_class == "negative":
        return "take_without_food"
    return "no_preference"


def predict_food_effect(
    drug_name: str,
    bcs_class: str,
    ka_fasted_per_h: float = 1.0,
    f_fasted: float = 0.7,
    logP: float = 2.0,
) -> FoodEffectResult:
    """Predict food effect on drug PK based on BCS class.

    Args:
        drug_name: Name of the drug.
        bcs_class: BCS classification ("I", "II", "III", "IV").
        ka_fasted_per_h: Absorption rate constant in fasted state (1/h).
        f_fasted: Bioavailability in fasted state (0, 1].
        logP: Lipophilicity (log partition coefficient).

    Returns:
        FoodEffectResult with predicted ratios and recommendations.
    """
    if bcs_class not in VALID_BCS_CLASSES:
        raise ValueError(f"bcs_class must be one of {VALID_BCS_CLASSES}, got '{bcs_class}'")
    if f_fasted <= 0 or f_fasted > 1:
        raise ValueError(f"f_fasted must be in (0, 1], got {f_fasted}")
    if ka_fasted_per_h <= 0:
        raise ValueError(f"ka_fasted_per_h must be > 0, got {ka_fasted_per_h}")

    if bcs_class == "I":
        cmax_ratio = 0.9
        auc_ratio = 1.0
        tmax_ratio = 1.5
        notes = "BCS I: high solubility, high permeability; food slightly delays absorption"
    elif bcs_class == "II":
        auc_ratio = min(2.0, 1.0 + 0.2 * max(0.0, logP - 2.0))
        cmax_ratio = auc_ratio * 0.85
        tmax_ratio = 2.0
        notes = "BCS II: low solubility, high permeability; food enhances dissolution"
    elif bcs_class == "III":
        cmax_ratio = 0.75
        auc_ratio = 0.85
        tmax_ratio = 1.2
        notes = "BCS III: high solubility, low permeability; food reduces absorption"
    else:  # IV
        cmax_ratio = 1.1
        auc_ratio = 1.1
        tmax_ratio = 1.8
        notes = "BCS IV: low solubility, low permeability; slight positive food effect"

    ka_fed = ka_fasted_per_h / tmax_ratio
    f_fed_raw = f_fasted * auc_ratio
    f_fed = max(0.0, min(1.0, f_fed_raw))

    food_effect_cls = _classify_food_effect(auc_ratio)
    recommendation = _clinical_recommendation(food_effect_cls)

    return FoodEffectResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        food_state="fed",
        cmax_ratio=cmax_ratio,
        auc_ratio=auc_ratio,
        tmax_ratio=tmax_ratio,
        ka_fed=ka_fed,
        f_fed=f_fed,
        food_effect_class=food_effect_cls,
        clinical_recommendation=recommendation,
        notes=notes,
    )


def compare_bcs_classes(drug_name: str, logP: float = 2.0) -> list:
    """Compare food effects across all BCS classes, sorted by auc_ratio descending.

    Args:
        drug_name: Name of the drug.
        logP: Lipophilicity.

    Returns:
        List of FoodEffectResult sorted by auc_ratio descending.
    """
    results = []
    for bcs in ["I", "II", "III", "IV"]:
        result = predict_food_effect(drug_name, bcs, logP=logP)
        results.append(result)
    results.sort(key=lambda r: r.auc_ratio, reverse=True)
    return results
