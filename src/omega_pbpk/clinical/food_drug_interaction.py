"""Food-drug interactions affecting oral absorption."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FoodEffectResult",
    "predict_food_effect",
    "food_effect_bcs_matrix",
]

_BCS_SOLUBILITY = {"I": 10.0, "II": 0.05, "III": 5.0, "IV": 0.01}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class FoodEffectResult:
    drug_name: str
    food_state: str  # 'fasted' or 'fed'
    logP: float
    pka: float
    gastric_pH: float
    solubility_mg_mL: float
    bile_factor: float
    motility_factor: float
    predicted_AUC_ratio: float
    predicted_Cmax_ratio: float
    food_effect_category: str  # 'positive', 'negative', or 'neutral'
    mechanism: str


def predict_food_effect(
    drug_name: str,
    logP: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    dose_mg: float = 100.0,
    bcs_class: str = "II",
    solubility_fasted_mg_mL: float = 0.1,
) -> FoodEffectResult:
    """Predict fed vs fasted pharmacokinetic ratio for a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Lipophilicity (log partition coefficient).
    pka:
        Acid/base dissociation constant.
    drug_type:
        One of 'neutral', 'acid', or 'base'.
    dose_mg:
        Dose in milligrams. Must be > 0.
    bcs_class:
        Biopharmaceutics Classification System class ('I', 'II', 'III', 'IV').
    solubility_fasted_mg_mL:
        Aqueous solubility in fasted state (mg/mL). Must be > 0.

    Returns
    -------
    FoodEffectResult
        Prediction for the *fed* state relative to fasted.

    Raises
    ------
    ValueError
        If ``dose_mg`` <= 0 or ``solubility_fasted_mg_mL`` <= 0.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if solubility_fasted_mg_mL <= 0:
        raise ValueError(
            f"solubility_fasted_mg_mL must be positive, got {solubility_fasted_mg_mL}"
        )

    gastric_pH_fasted = 1.5
    gastric_pH_fed = 4.5

    # Bile salt enhancement: higher logP → more bile micellar solubilization
    bile_factor_fed = 1.0 + 3.0 * _sigmoid(logP - 2.0)
    bile_factor_fasted = 1.0

    motility_factor_fed = 0.65
    motility_factor_fasted = 1.0

    # Solubility in fed state (bile micellar solubilization)
    solubility_fed = solubility_fasted_mg_mL * bile_factor_fed

    # pH-dependent solubility adjustment
    delta_pH = gastric_pH_fed - gastric_pH_fasted  # positive: fed is less acidic
    if drug_type == "acid":
        # Henderson-Hasselbalch: weak acids more ionized (soluble) at higher pH
        # fed pH is higher → more ionized → actually more soluble for weak acids
        # But for BCS framing we note absorption of un-ionized species may fall
        ph_factor = 1.0 + 0.1 * delta_pH  # mild positive for acid solubility
    elif drug_type == "base":
        # Weak bases less ionized at higher pH → less soluble in fed state
        ph_factor = max(0.5, 1.0 - 0.15 * delta_pH)
    else:
        ph_factor = 1.0

    solubility_fed *= ph_factor

    # Simplified AUC ratio: driven by solubility increase and motility reduction
    # predicted_AUC_ratio = (solubility_fed / solubility_fasted) * (motility_fed / motility_fasted)
    predicted_AUC_ratio = (
        (solubility_fed / solubility_fasted_mg_mL)
        * motility_factor_fed
        / motility_factor_fasted
    )
    predicted_Cmax_ratio = predicted_AUC_ratio * 0.8

    if predicted_AUC_ratio > 1.25:
        food_effect_category = "positive"
    elif predicted_AUC_ratio < 0.8:
        food_effect_category = "negative"
    else:
        food_effect_category = "neutral"

    # Describe dominant mechanism
    bile_contribution = bile_factor_fed - 1.0  # how much bile adds
    if bile_contribution > 1.0:
        mechanism = (
            f"Bile salt micellar solubilization dominates (logP={logP:.1f}); "
            f"bile_factor={bile_factor_fed:.2f} increases solubility and absorption. "
            f"Reduced GI motility (factor={motility_factor_fed}) moderates Cmax."
        )
    elif drug_type in ("acid", "base"):
        mechanism = (
            f"pH change (fasted {gastric_pH_fasted} → fed {gastric_pH_fed}) alters "
            f"ionization of {drug_type} (pKa={pka:.1f}). "
            f"Motility factor={motility_factor_fed} prolongs absorption window."
        )
    else:
        mechanism = (
            f"Reduced GI motility with food (factor={motility_factor_fed}) prolongs "
            f"residence time. Bile solubilization minor for logP={logP:.1f}."
        )

    return FoodEffectResult(
        drug_name=drug_name,
        food_state="fed",
        logP=logP,
        pka=pka,
        gastric_pH=gastric_pH_fed,
        solubility_mg_mL=solubility_fed,
        bile_factor=bile_factor_fed,
        motility_factor=motility_factor_fed,
        predicted_AUC_ratio=predicted_AUC_ratio,
        predicted_Cmax_ratio=predicted_Cmax_ratio,
        food_effect_category=food_effect_category,
        mechanism=mechanism,
    )


def food_effect_bcs_matrix(
    drug_name: str,
    logP: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
) -> dict[str, FoodEffectResult]:
    """Evaluate food effect across all four BCS classes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Lipophilicity.
    pka:
        Acid/base dissociation constant.
    drug_type:
        One of 'neutral', 'acid', or 'base'.

    Returns
    -------
    dict[str, FoodEffectResult]
        Mapping of BCS class ('I', 'II', 'III', 'IV') to predicted food effect.
    """
    return {
        bcs_class: predict_food_effect(
            drug_name=drug_name,
            logP=logP,
            pka=pka,
            drug_type=drug_type,
            bcs_class=bcs_class,
            solubility_fasted_mg_mL=solubility,
        )
        for bcs_class, solubility in _BCS_SOLUBILITY.items()
    }
