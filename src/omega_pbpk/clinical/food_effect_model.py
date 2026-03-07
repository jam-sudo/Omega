"""Phase 568 — Food Effect Model on Oral Drug Absorption.

Models how food state (fasted vs fed) alters solubility, Cmax, AUC, and
tmax for drugs of different physicochemical types.

References:
  - FDA 2002 Food-Effect Bioavailability and Fed Bioequivalence Studies guidance
  - Fleisher et al. (1999) Clin Pharmacokinet
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_DRUG_TYPES = {"acid", "base", "neutral", "lipophilic"}
_VALID_FOOD_STATES = {"fasted", "fed"}


def _validate_drug_type(drug_type: str) -> str:
    dt = drug_type.lower().strip()
    if dt not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")
    return dt


def _validate_food_state(food_state: str) -> str:
    fs = food_state.lower().strip()
    if fs not in _VALID_FOOD_STATES:
        raise ValueError(f"food_state must be one of {_VALID_FOOD_STATES}, got '{food_state}'")
    return fs


# ---------------------------------------------------------------------------
# food_effect_on_solubility
# ---------------------------------------------------------------------------


def food_effect_on_solubility(
    drug_type: str,
    logP: float,
    food_state: str = "fed",
) -> dict:
    """Estimate food-state effect on drug solubility.

    Fed state increases bile salt concentration which primarily benefits
    lipophilic drugs.  Acid drugs see a slight reduction in the fed stomach.

    Parameters
    ----------
    drug_type : 'acid', 'base', 'neutral', or 'lipophilic'
    logP      : octanol-water partition coefficient
    food_state: 'fasted' or 'fed'

    Returns
    -------
    dict with keys: drug_type, logP, food_state, solubility_fold_change, notes
    """
    dt = _validate_drug_type(drug_type)
    fs = _validate_food_state(food_state)

    if fs == "fasted":
        fold = 1.0
        notes = "Fasted state: reference solubility"
    else:
        # fed state
        if dt == "lipophilic" or (dt == "neutral" and logP > 3):
            # fold = 2 + logP/2, capped at 8
            fold = min(2.0 + logP / 2.0, 8.0)
            fold = max(fold, 1.0)
            notes = "Fed: bile salts enhance lipophilic drug solubility"
        elif dt == "acid":
            fold = 0.85
            notes = (
                "Fed: lower gastric pH reduces ionisation of weak acid, slight solubility decrease"
            )
        elif dt == "base":
            fold = 1.5
            notes = "Fed: bile salts and higher gastric pH increase base solubility"
        else:  # neutral (logP <= 3)
            fold = 1.2
            notes = "Fed: modest solubility increase for neutral drug"

    return {
        "drug_type": dt,
        "logP": logP,
        "food_state": fs,
        "solubility_fold_change": fold,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# food_effect_on_absorption
# ---------------------------------------------------------------------------


def food_effect_on_absorption(
    fa_fasted: float,
    cmax_fasted: float,
    auc_fasted: float,
    drug_type: str,
    logP: float,
) -> dict:
    """Apply fed-state food-effect multipliers to fa, Cmax, and AUC.

    Parameters
    ----------
    fa_fasted   : fraction absorbed in fasted state (0 to 1)
    cmax_fasted : Cmax in fasted state (any concentration unit)
    auc_fasted  : AUC in fasted state (any AUC unit)
    drug_type   : 'acid', 'base', 'neutral', or 'lipophilic'
    logP        : octanol-water log partition coefficient

    Returns
    -------
    dict: fa_fed, cmax_fed, auc_fed, food_effect_ratio_auc, clinical_significance
    """
    if not (0.0 <= fa_fasted <= 1.0):
        raise ValueError(f"fa_fasted must be in [0, 1], got {fa_fasted}")
    if cmax_fasted < 0:
        raise ValueError("cmax_fasted must be >= 0")
    if auc_fasted < 0:
        raise ValueError("auc_fasted must be >= 0")

    dt = _validate_drug_type(drug_type)

    sol_result = food_effect_on_solubility(drug_type=dt, logP=logP, food_state="fed")
    fold = sol_result["solubility_fold_change"]

    if dt == "lipophilic" or (dt == "neutral" and logP > 3):
        multiplier = fold
    elif dt == "acid":
        multiplier = 0.85
    elif dt == "base":
        multiplier = 1.5
    else:
        multiplier = 1.2

    # fa cannot exceed 1.0
    fa_fed = min(fa_fasted * multiplier, 1.0)
    cmax_fed = cmax_fasted * multiplier
    auc_fed = auc_fasted * multiplier
    food_effect_ratio_auc = auc_fed / auc_fasted if auc_fasted > 0 else multiplier

    # Clinical significance
    if food_effect_ratio_auc >= 2.0 or food_effect_ratio_auc <= 0.5:
        significance = "major"
    elif food_effect_ratio_auc >= 1.5 or food_effect_ratio_auc <= 0.67:
        significance = "moderate"
    elif food_effect_ratio_auc >= 1.25 or food_effect_ratio_auc <= 0.8:
        significance = "minor"
    else:
        significance = "none"

    return {
        "fa_fed": fa_fed,
        "cmax_fed": cmax_fed,
        "auc_fed": auc_fed,
        "food_effect_ratio_auc": food_effect_ratio_auc,
        "clinical_significance": significance,
    }


# ---------------------------------------------------------------------------
# high_fat_meal_effect
# ---------------------------------------------------------------------------


def high_fat_meal_effect(
    dose_mg: float,
    logP: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
) -> dict:
    """Model the effect of a high-fat meal on drug PK.

    High-fat meal:
      - Increases GI transit time by 1.5x (tmax delay ~1 h)
      - Increases bile acid pool by 3x
      - AUC_ratio = 1 + 2*(logP/5) for logP > 0, else 1.0; capped at 5

    Parameters
    ----------
    dose_mg  : dose in mg (not used in ratio calculation, kept for API)
    logP     : octanol-water partition coefficient
    pka_acid : pKa of acidic group (optional)
    pka_base : pKa of basic group (optional)

    Returns
    -------
    dict: auc_ratio, tmax_delay_h, clinical_significance
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    if logP > 0:
        auc_ratio = 1.0 + 2.0 * (logP / 5.0)
    else:
        auc_ratio = 1.0

    auc_ratio = min(auc_ratio, 5.0)
    auc_ratio = max(auc_ratio, 0.1)

    tmax_delay_h = 1.0

    if auc_ratio >= 2.0:
        significance = "major"
    elif auc_ratio >= 1.5:
        significance = "moderate"
    elif auc_ratio >= 1.25:
        significance = "minor"
    else:
        significance = "none"

    return {
        "auc_ratio": auc_ratio,
        "tmax_delay_h": tmax_delay_h,
        "clinical_significance": significance,
    }


# ---------------------------------------------------------------------------
# fasted_fed_recommendation
# ---------------------------------------------------------------------------


def fasted_fed_recommendation(
    drug_name: str,
    logP: float,
    solubility_mg_mL: float,
) -> str:
    """Return a clinical dosing recommendation based on logP and solubility.

    Parameters
    ----------
    drug_name        : drug identifier (used in message)
    logP             : octanol-water partition coefficient
    solubility_mg_mL : aqueous solubility (mg/mL)

    Returns
    -------
    Recommendation string.
    """
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")

    if logP > 3 and solubility_mg_mL < 0.1:
        return "Take with food to improve absorption"
    elif logP < 0 and solubility_mg_mL >= 0.1:
        return "Take on empty stomach for fastest absorption"
    else:
        return "May be taken with or without food"
