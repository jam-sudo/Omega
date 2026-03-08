"""Phase 182 — Meal / food effect on drug pharmacokinetics.

Models how food intake modifies Cmax, Tmax and AUC for orally administered
drugs based on BCS class and lipophilicity.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid BCS classes
_VALID_BCS_CLASSES = {1, 2, 3, 4}

# logP bounds for reasonable inputs
_LOGP_MIN = -10.0
_LOGP_MAX = 15.0

# Food-effect classification thresholds
# (cmax_ratio relative to fasted)
_CLASS_THRESHOLDS = {
    "none": (0.85, 1.25),  # within 15-25 % → none
    "minor": (0.70, 1.50),  # within 30-50 % → minor
    "moderate": (0.50, 2.00),  # within 50-100 % → moderate
    # outside → significant
}


def _classify_food_effect(cmax_ratio: float, auc_ratio: float) -> str:
    """Classify the magnitude of the food effect."""
    max_ratio = max(cmax_ratio, auc_ratio)
    min_ratio = min(cmax_ratio, auc_ratio)

    # Use the larger deviation from 1.0 for classification
    deviation = max(abs(max_ratio - 1.0), abs(1.0 - min_ratio))

    if deviation <= 0.15:
        return "none"
    if deviation <= 0.30:
        return "minor"
    if deviation <= 0.60:
        return "moderate"
    return "significant"


def _administration_recommendation(
    food_effect_class: str, cmax_ratio: float, auc_ratio: float
) -> str:
    """Generate a dosing recommendation string."""
    if food_effect_class == "none":
        return "May be taken with or without food."
    if food_effect_class == "minor":
        return "Food effect is minor; administration with or without food is acceptable."
    if auc_ratio > 1.0:
        return (
            "Take with food to enhance absorption"
            f" (AUC increases ~{(auc_ratio - 1.0) * 100:.0f}% with food)."
        )
    return (
        "Take on an empty stomach to avoid reduced absorption"
        f" (AUC decreases ~{(1.0 - auc_ratio) * 100:.0f}% with food)."
    )


@dataclass(frozen=True)
class MealEffectResult:
    """Result of a food-effect assessment for a drug."""

    drug_name: str
    bcs_class: int
    logP: float
    fed_state: str  # "fed" or timing description
    cmax_ratio_fed_fasted: float
    tmax_shift_h: float
    auc_ratio_fed_fasted: float
    food_effect_class: str  # "none", "minor", "moderate", "significant"
    administration_recommendation: str
    notes: str


def assess_meal_effect(
    drug_name: str,
    bcs_class: int,
    logP: float,
    fed_state: str = "fed",
    meal_timing_h: float = 0.0,
) -> MealEffectResult:
    """Assess the effect of food on drug PK parameters.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    bcs_class:
        BCS classification (1, 2, 3, or 4).
    logP:
        Octanol-water partition coefficient (log scale).
    fed_state:
        Descriptive label for the fed state (stored in result).
    meal_timing_h:
        Hours between meal ingestion and drug administration.
        0 = taken simultaneously with food; positive values mean the meal
        was consumed before dosing.  Larger values → reduced food effect.

    Returns
    -------
    MealEffectResult
    """
    if bcs_class not in _VALID_BCS_CLASSES:
        raise ValueError(f"bcs_class must be one of {sorted(_VALID_BCS_CLASSES)}, got {bcs_class}")
    if not (_LOGP_MIN <= logP <= _LOGP_MAX):
        raise ValueError(f"logP must be in [{_LOGP_MIN}, {_LOGP_MAX}], got {logP}")

    # Timing attenuation: at t=0 (with food) full effect; decays with time
    # Use a simple exponential decay with half-life ~2 h
    attenuation = 1.0 / (1.0 + meal_timing_h / 2.0)

    # --- BCS-class-specific rules ---
    if bcs_class == 1:
        # High solubility, high permeability → minimal food effect
        cmax_ratio = 1.0
        tmax_shift_h = 0.5 * attenuation  # slight delay
        auc_ratio = 1.0
        notes = "BCS I: well-absorbed regardless of food state."

    elif bcs_class == 2:
        # Low solubility, high permeability → food increases dissolution
        if logP > 2:
            # Lipophilic: bile secretion aids dissolution significantly
            # logP effect: linear between 2 and 6
            logp_factor = min((logP - 2.0) / 4.0, 1.0)  # 0 to 1
            cmax_ratio = 1.2 + 0.6 * logp_factor  # 1.2–1.8
            auc_ratio = 1.15 + 0.5 * logp_factor  # 1.15–1.65
            # For high logP, faster apparent absorption due to solubilization
            tmax_shift_h = -0.5 * logp_factor * attenuation  # earlier
            notes = (
                "BCS II (lipophilic): food markedly increases solubilization "
                f"and absorption (logP={logP:.1f})."
            )
        else:
            # Hydrophilic BCS II: moderate food effect
            cmax_ratio = 1.1
            auc_ratio = 1.1
            tmax_shift_h = 0.5 * attenuation
            notes = "BCS II (low logP): moderate food effect via increased gastric emptying."

        # Apply timing attenuation to the food-effect delta
        cmax_ratio = 1.0 + (cmax_ratio - 1.0) * attenuation
        auc_ratio = 1.0 + (auc_ratio - 1.0) * attenuation

    elif bcs_class == 3:
        # High solubility, low permeability → food generally reduces absorption
        # (slowed gastric emptying reduces exposure to permeable region)
        reduction = 0.15 + 0.1 * max(0.0, -logP)  # more polar → more reduction
        reduction = min(reduction, 0.30)
        cmax_ratio = 1.0 - reduction * attenuation
        auc_ratio = 1.0 - reduction * attenuation
        tmax_shift_h = 1.0 * attenuation
        notes = (
            "BCS III: low permeability; food may reduce absorption by slowing "
            "transit through permeable GI segments."
        )

    else:  # bcs_class == 4
        # Low solubility, low permeability → variable; food can help via bile
        if logP > 0:
            benefit = min(0.5 * logP / 5.0, 0.5)  # 0 to 0.5 benefit
            cmax_ratio = 1.0 + benefit * attenuation
            auc_ratio = 1.0 + benefit * attenuation
            tmax_shift_h = 0.5 * attenuation
            notes = (
                "BCS IV (logP>0): food may modestly improve absorption "
                "via bile-mediated solubilization."
            )
        else:
            cmax_ratio = 0.9
            auc_ratio = 0.95
            tmax_shift_h = 1.0 * attenuation
            notes = "BCS IV (logP<=0): food effect is unpredictable; generally limited benefit."
            cmax_ratio = 1.0 + (cmax_ratio - 1.0) * attenuation
            auc_ratio = 1.0 + (auc_ratio - 1.0) * attenuation

    food_effect_class = _classify_food_effect(cmax_ratio, auc_ratio)
    recommendation = _administration_recommendation(food_effect_class, cmax_ratio, auc_ratio)

    return MealEffectResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        logP=logP,
        fed_state=fed_state,
        cmax_ratio_fed_fasted=cmax_ratio,
        tmax_shift_h=tmax_shift_h,
        auc_ratio_fed_fasted=auc_ratio,
        food_effect_class=food_effect_class,
        administration_recommendation=recommendation,
        notes=notes,
    )


def screen_food_timing(
    drug_name: str,
    bcs_class: int,
    logP: float,
    meal_timings_h: list[float],
) -> list[MealEffectResult]:
    """Assess food effect at multiple pre-dose meal timings.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    bcs_class:
        BCS classification (1, 2, 3, or 4).
    logP:
        Octanol-water partition coefficient.
    meal_timings_h:
        List of meal timings (hours before dose).  0 means taken with food.

    Returns
    -------
    List of MealEffectResult, one per timing in ``meal_timings_h``.
    """
    if not meal_timings_h:
        raise ValueError("meal_timings_h must not be empty")

    results: list = []
    for timing in meal_timings_h:
        label = "with food" if timing == 0.0 else f"{timing:.1f}h pre-dose"
        result = assess_meal_effect(
            drug_name=drug_name,
            bcs_class=bcs_class,
            logP=logP,
            fed_state=label,
            meal_timing_h=timing,
        )
        results.append(result)
    return results
