"""Phase 973 — Food effect on oral drug absorption (fasted vs fed state)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoodEffectResult:
    """Result of food effect prediction for oral drug absorption."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    fa_fasted: float
    fa_fed: float
    auc_ratio_fed_fasted: float
    food_effect_classification: str
    bile_enhancement_factor: float
    recommendation: str
    notes: str


def _ionization_fraction(ionization_type: str, pka: float, ph: float) -> float:
    """Return neutral fraction alpha at given pH using Henderson-Hasselbalch."""
    if ionization_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    elif ionization_type == "base":
        return 1.0 / (1.0 + 10.0 ** (ph - pka))
    else:  # neutral
        return 1.0


def _fa_from_dose_number(dn: float) -> float:
    """Compute fraction absorbed from dose number; capped at 0.99."""
    fa = 1.0 / (1.0 + dn)
    return min(fa, 0.99)


def predict_food_effect(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str = "neutral",
    dose_mg: float = 100.0,
    solubility_mg_mL: float = 1.0,
) -> FoodEffectResult:
    """Predict food effect on oral drug absorption.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Octanol-water partition coefficient (-5 to 10).
    pka:
        Acid/base dissociation constant (0 to 14).
    ionization_type:
        One of "acid", "base", or "neutral".
    dose_mg:
        Oral dose in mg.
    solubility_mg_mL:
        Intrinsic (un-ionized) aqueous solubility in mg/mL.

    Returns
    -------
    FoodEffectResult
    """
    # --- Validation ----------------------------------------------------------
    valid_ionization = {"acid", "base", "neutral"}
    if ionization_type not in valid_ionization:
        raise ValueError(
            f"ionization_type must be one of {valid_ionization}, got '{ionization_type}'"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be between 0 and 14, got {pka}")
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be between -5 and 10, got {logp}")

    # --- Gastric pH constants ------------------------------------------------
    ph_fasted = 2.0
    ph_fed = 4.0

    # --- Step 1: Neutral (un-ionized) fractions at each gastric pH -----------
    alpha_fasted = _ionization_fraction(ionization_type, pka, ph_fasted)
    alpha_fed = _ionization_fraction(ionization_type, pka, ph_fed)

    # --- Step 2: Bile enhancement factor (fed only) --------------------------
    bile_enhancement_factor = max(1.0, 1.0 + max(0.0, logp) * 0.5)

    # --- Step 3: Effective solubility ----------------------------------------
    sol_fasted = solubility_mg_mL * alpha_fasted
    sol_fed = solubility_mg_mL * alpha_fed * bile_enhancement_factor

    # Avoid division by zero for very insoluble compounds
    if sol_fasted <= 0.0:
        sol_fasted = 1e-9
    if sol_fed <= 0.0:
        sol_fed = 1e-9

    # --- Step 4: Dose number (250 mL gastric volume) -------------------------
    volume_mL = 250.0
    dn_fasted = dose_mg / (sol_fasted * volume_mL)
    dn_fed = dose_mg / (sol_fed * volume_mL)

    # --- Step 5: Fraction absorbed -------------------------------------------
    fa_fasted = _fa_from_dose_number(dn_fasted)
    fa_fed = _fa_from_dose_number(dn_fed)

    # --- Step 6: AUC ratio ---------------------------------------------------
    auc_ratio = fa_fed / fa_fasted if fa_fasted > 0.0 else 1.0

    # --- Step 7: Classification ----------------------------------------------
    if auc_ratio > 1.25:
        classification = "positive"
    elif auc_ratio < 0.8:
        classification = "negative"
    else:
        classification = "none"

    # --- Step 8: Recommendation ----------------------------------------------
    if classification == "positive":
        recommendation = "take with food"
    elif classification == "negative":
        recommendation = "take fasted"
    else:
        recommendation = "no preference"

    # --- Step 9: Notes -------------------------------------------------------
    notes_parts: list[str] = [
        f"Ionization type: {ionization_type}",
        f"Neutral fraction fasted (pH {ph_fasted}): {alpha_fasted:.3f}",
        f"Neutral fraction fed (pH {ph_fed}): {alpha_fed:.3f}",
        f"Bile enhancement factor: {bile_enhancement_factor:.3f}",
        f"Dose number fasted: {dn_fasted:.3f}",
        f"Dose number fed: {dn_fed:.3f}",
    ]
    notes = "; ".join(notes_parts)

    return FoodEffectResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fa_fasted=fa_fasted,
        fa_fed=fa_fed,
        auc_ratio_fed_fasted=auc_ratio,
        food_effect_classification=classification,
        bile_enhancement_factor=bile_enhancement_factor,
        recommendation=recommendation,
        notes=notes,
    )
