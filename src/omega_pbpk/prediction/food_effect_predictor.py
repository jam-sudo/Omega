"""Pharmacokinetic Food Effect Predictor — Phase 402.

Predicts the effect of food (fed vs fasted state) on drug pharmacokinetics
including changes in Cmax, AUC, and Tmax.

Model basis:
- BCS classification from dose number (Do) and permeability (logP)
- Lookup-table food effect ratios by BCS class and meal type
- Simplified analytical 1-compartment PK for fasted baseline
- FDA 2002 food-effect guidance for high-fat meal as standard condition

References:
- FDA CDER. Food-Effect Bioavailability and Fed Bioequivalence Studies. 2002.
- Amidon GL et al., Pharm Res. 1995;12(3):413-420 (BCS).
- Fleisher D et al., Clin Pharmacokinet. 1999;36(3):233-54.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FoodEffectResult",
    "predict_food_effect",
    "compare_food_states",
]

# ---------------------------------------------------------------------------
# Food effect lookup tables (AUC and Cmax ratios, fed/fasted)
# ---------------------------------------------------------------------------

_FOOD_AUC_RATIOS: dict = {
    "I": {"high_fat_meal": 1.05, "low_fat_meal": 1.02, "grapefruit": 1.5},
    "II": {"high_fat_meal": 2.0, "low_fat_meal": 1.4, "grapefruit": 1.3},
    "III": {"high_fat_meal": 0.85, "low_fat_meal": 0.9, "grapefruit": 1.0},
    "IV": {"high_fat_meal": 1.3, "low_fat_meal": 1.1, "grapefruit": 1.0},
}

_FOOD_CMAX_RATIOS: dict = {
    "I": {"high_fat_meal": 0.9, "low_fat_meal": 0.95, "grapefruit": 2.0},
    "II": {"high_fat_meal": 1.8, "low_fat_meal": 1.3, "grapefruit": 1.2},
    "III": {"high_fat_meal": 0.7, "low_fat_meal": 0.85, "grapefruit": 0.9},
    "IV": {"high_fat_meal": 1.2, "low_fat_meal": 1.0, "grapefruit": 0.9},
}

_VALID_FOOD_STATES = ("high_fat_meal", "low_fat_meal", "grapefruit")
_VALID_DRUG_TYPES = ("acid", "base", "neutral")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoodEffectResult:
    """Result from food effect prediction."""

    drug_name: str
    food_state: str  # "fasted", "high_fat_meal", "low_fat_meal", "grapefruit"
    logp: float
    bcs_class: str  # "I", "II", "III", "IV"
    fasted_cmax_mg_L: float
    fed_cmax_mg_L: float
    cmax_ratio_fed_fasted: float  # >1 means food increases Cmax
    fasted_auc_mg_h_per_L: float
    fed_auc_mg_h_per_L: float
    auc_ratio_fed_fasted: float
    fasted_tmax_h: float
    fed_tmax_h: float
    gastric_emptying_effect: str  # "delayed", "unchanged", "accelerated"
    dissolution_effect: str  # "enhanced", "unchanged", "reduced"
    bile_salt_effect: str  # "enhanced", "unchanged"
    clinical_food_recommendation: str  # "with_food", "without_food", "either", "avoid_fat"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_bcs(
    logp: float,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
) -> str:
    """Classify BCS class from dose number and logP-based permeability."""
    # Dose number: Do = dose / (250 mL * solubility)
    do_fasted = dose_mg / (solubility_fasted_mg_mL * 250.0)

    # High permeability: logP 0-5 (lipophilic enough to cross membranes)
    high_perm = 0.0 <= logp <= 5.0

    if do_fasted <= 1.0 and high_perm:
        return "I"
    elif do_fasted > 1.0 and logp >= 0.0:
        return "II"
    elif do_fasted <= 1.0 and logp < 0.0:
        return "III"
    else:
        return "IV"


def _fasted_pk(dose_mg: float) -> tuple:
    """Compute simplified fasted-state PK parameters.

    Uses a 1-compartment oral model with fixed ke and ka.

    Returns (cmax_mg_L, auc_mg_h_per_L, tmax_h)
    """
    ke = 0.2  # baseline elimination rate (1/h)
    ka = 1.0  # absorption rate (1/h)

    # Simplified Cmax (reference scale)
    cmax_fasted = dose_mg / 50.0 * 0.8

    # AUC = dose / (Vd * ke); using Vd = 50 L
    auc_fasted = dose_mg / (50.0 * ke)

    # Tmax = ln(ka/ke) / (ka - ke)
    tmax_fasted = math.log(ka / ke) / (ka - ke)

    return cmax_fasted, auc_fasted, tmax_fasted


def _gastric_emptying_effect(food_state: str) -> str:
    if food_state in ("high_fat_meal", "low_fat_meal"):
        return "delayed"
    return "unchanged"


def _dissolution_effect(bcs_class: str, food_state: str) -> str:
    if bcs_class == "II" and food_state == "high_fat_meal":
        return "enhanced"
    return "unchanged"


def _bile_salt_effect(food_state: str) -> str:
    if food_state == "high_fat_meal":
        return "enhanced"
    return "unchanged"


def _tmax_fed(tmax_fasted: float, food_state: str) -> float:
    if food_state == "high_fat_meal":
        return tmax_fasted * 1.5
    elif food_state == "low_fat_meal":
        return tmax_fasted * 1.2
    return tmax_fasted  # grapefruit: unchanged


def _clinical_recommendation(
    auc_ratio: float,
    cmax_ratio: float,
    bcs_class: str,
) -> str:
    if cmax_ratio > 2.0:
        return "avoid_fat"
    if auc_ratio > 1.5 and bcs_class == "II":
        return "with_food"
    if auc_ratio < 0.8:
        return "without_food"
    return "either"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_food_effect(
    drug_name: str,
    logp: float,
    mw_da: float,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
    pka: float,
    drug_type: str,
    bcs_class: str | None = None,
    food_state: str = "high_fat_meal",
) -> FoodEffectResult:
    """Predict pharmacokinetic food effect.

    Parameters
    ----------
    drug_name:
        Drug name.
    logp:
        Octanol-water partition coefficient (dimensionless). Range [-5, 10].
    mw_da:
        Molecular weight (Da). Must be > 0.
    dose_mg:
        Dose (mg). Must be > 0.
    solubility_fasted_mg_mL:
        Aqueous solubility in fasted state (mg/mL). Must be > 0.
    pka:
        Most acidic/basic pKa. Range [-5, 14].
    drug_type:
        Ionization type: "acid", "base", or "neutral".
    bcs_class:
        BCS class override ("I", "II", "III", "IV"). If None, auto-classified.
    food_state:
        Meal type: "high_fat_meal" (default), "low_fat_meal", or "grapefruit".

    Returns
    -------
    FoodEffectResult
    """
    # --- Validation ---
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_fasted_mg_mL <= 0:
        raise ValueError("solubility_fasted_mg_mL must be > 0")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got {drug_type!r}")
    if not (-5.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [-5, 14], got {pka}")
    if food_state not in _VALID_FOOD_STATES:
        raise ValueError(f"food_state must be one of {_VALID_FOOD_STATES}, got {food_state!r}")

    # --- BCS classification ---
    if bcs_class is None:
        bcs_class = _classify_bcs(logp, dose_mg, solubility_fasted_mg_mL)
    if bcs_class not in ("I", "II", "III", "IV"):
        raise ValueError(f"bcs_class must be I, II, III, or IV, got {bcs_class!r}")

    # --- Fasted state baseline PK ---
    cmax_fasted, auc_fasted, tmax_fasted = _fasted_pk(dose_mg)

    # --- Food effect ratios ---
    auc_ratio = _FOOD_AUC_RATIOS[bcs_class][food_state]
    cmax_ratio = _FOOD_CMAX_RATIOS[bcs_class][food_state]

    # --- Fed state PK ---
    cmax_fed = cmax_fasted * cmax_ratio
    auc_fed = auc_fasted * auc_ratio
    tmax_fed_val = _tmax_fed(tmax_fasted, food_state)

    # --- Mechanistic effects ---
    gastric_eff = _gastric_emptying_effect(food_state)
    dissolution_eff = _dissolution_effect(bcs_class, food_state)
    bile_salt_eff = _bile_salt_effect(food_state)

    # --- Clinical recommendation ---
    recommendation = _clinical_recommendation(auc_ratio, cmax_ratio, bcs_class)

    # --- Notes ---
    do_val = dose_mg / (solubility_fasted_mg_mL * 250.0)
    notes = (
        f"BCS {bcs_class}: Do={do_val:.2f}, logP={logp}; "
        f"food state={food_state}; "
        f"AUC ratio={auc_ratio:.2f}, Cmax ratio={cmax_ratio:.2f}"
    )

    return FoodEffectResult(
        drug_name=drug_name,
        food_state=food_state,
        logp=logp,
        bcs_class=bcs_class,
        fasted_cmax_mg_L=cmax_fasted,
        fed_cmax_mg_L=cmax_fed,
        cmax_ratio_fed_fasted=cmax_ratio,
        fasted_auc_mg_h_per_L=auc_fasted,
        fed_auc_mg_h_per_L=auc_fed,
        auc_ratio_fed_fasted=auc_ratio,
        fasted_tmax_h=tmax_fasted,
        fed_tmax_h=tmax_fed_val,
        gastric_emptying_effect=gastric_eff,
        dissolution_effect=dissolution_eff,
        bile_salt_effect=bile_salt_eff,
        clinical_food_recommendation=recommendation,
        notes=notes,
    )


def compare_food_states(
    drug_name: str,
    logp: float,
    mw_da: float,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
    pka: float,
    drug_type: str,
    bcs_class: str | None = None,
) -> list:
    """Compare high_fat_meal, low_fat_meal, and grapefruit food states.

    Returns list of FoodEffectResult sorted by auc_ratio_fed_fasted descending.
    """
    results = [
        predict_food_effect(
            drug_name=drug_name,
            logp=logp,
            mw_da=mw_da,
            dose_mg=dose_mg,
            solubility_fasted_mg_mL=solubility_fasted_mg_mL,
            pka=pka,
            drug_type=drug_type,
            bcs_class=bcs_class,
            food_state=fs,
        )
        for fs in _VALID_FOOD_STATES
    ]
    results.sort(key=lambda r: r.auc_ratio_fed_fasted, reverse=True)
    return results
