"""Phase 429 — Enteral Nutrition PK interactions.

Drug PK interactions with enteral nutrition (tube feeding):
binding to formula, absorption changes, gastric emptying effects.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Formula database
# ---------------------------------------------------------------------------

FORMULAS = {
    "standard": {
        "protein_g_per_L": 40,
        "ph": 6.5,
        "protein_binding_factor": 0.05,
        "emptying_factor": 0.9,
    },
    "high_protein": {
        "protein_g_per_L": 80,
        "ph": 6.7,
        "protein_binding_factor": 0.12,
        "emptying_factor": 0.85,
    },
    "elemental": {
        "protein_g_per_L": 20,
        "ph": 5.5,
        "protein_binding_factor": 0.02,
        "emptying_factor": 1.1,
    },
    "immune_modulating": {
        "protein_g_per_L": 60,
        "ph": 6.8,
        "protein_binding_factor": 0.08,
        "emptying_factor": 0.88,
    },
}

VALID_FEEDING_METHODS = {"continuous", "bolus", "cyclic"}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnteralNutritionPKResult:
    """Result of enteral nutrition PK interaction assessment."""

    drug_name: str
    formula_type: str
    feeding_method: str
    baseline_dose_mg: float
    adjusted_dose_mg: float
    bioavailability_baseline: float
    bioavailability_with_en: float
    bioavailability_ratio: float
    binding_loss_pct: float
    gastric_emptying_factor: float
    clinical_significance: str
    management_strategy: str
    notes: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def assess_enteral_nutrition_pk(
    drug_name: str,
    formula_type: str,
    feeding_method: str,
    baseline_dose_mg: float,
    baseline_bioavailability: float,
    drug_pka: float = 7.0,
    ionization_type: str = "neutral",
    drug_mw_Da: float = 300.0,
    logp: float = 1.0,
) -> EnteralNutritionPKResult:
    """Assess drug PK interaction with enteral nutrition formula.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    formula_type:
        One of "standard", "high_protein", "elemental", "immune_modulating".
    feeding_method:
        One of "continuous", "bolus", "cyclic".
    baseline_dose_mg:
        Dose without enteral nutrition adjustment (mg).
    baseline_bioavailability:
        Fraction absorbed without EN (0 < F <= 1).
    drug_pka:
        Drug pKa value.
    ionization_type:
        "acid", "base", or "neutral".
    drug_mw_Da:
        Molecular weight in Daltons.
    logp:
        Octanol-water partition coefficient.

    Returns
    -------
    EnteralNutritionPKResult
    """
    # --- Validation ---
    if formula_type not in FORMULAS:
        raise ValueError(
            f"formula_type must be one of {list(FORMULAS.keys())}, got '{formula_type}'"
        )
    if feeding_method not in VALID_FEEDING_METHODS:
        raise ValueError(
            f"feeding_method must be one of {sorted(VALID_FEEDING_METHODS)}, got '{feeding_method}'"
        )
    if baseline_dose_mg <= 0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")
    if not (0 < baseline_bioavailability <= 1):
        raise ValueError(
            f"baseline_bioavailability must be in (0, 1], got {baseline_bioavailability}"
        )

    formula = FORMULAS[formula_type]
    protein_binding_factor = formula["protein_binding_factor"]
    _formula_ph = formula["ph"]
    emptying_factor = formula["emptying_factor"]

    # --- Binding loss calculation ---
    # High MW increases binding
    if drug_mw_Da > 500:
        protein_binding_factor *= 1.5

    # Ionization adjustments
    if ionization_type == "acid":
        # Anionic drugs at high pH bind more
        protein_binding_factor *= 1.3
    elif ionization_type == "base":
        # Cationic drugs bind less
        protein_binding_factor *= 0.7

    binding_loss_pct = protein_binding_factor * 100.0

    # --- Gastric emptying factor ---
    gastric_emptying = emptying_factor
    if feeding_method == "continuous":
        # Continuous feeding slows emptying further
        gastric_emptying *= 0.95
    # bolus -> normal (no additional modification)
    # cyclic -> same as formula default

    # --- Bioavailability with EN ---
    bio_with_en = baseline_bioavailability * (1.0 - binding_loss_pct / 100.0) * gastric_emptying
    bio_with_en = max(0.05, min(1.0, bio_with_en))

    # --- Bioavailability ratio ---
    bio_ratio = bio_with_en / baseline_bioavailability

    # --- Adjusted dose ---
    if bio_ratio > 0:
        adjusted_dose = baseline_dose_mg / bio_ratio
    else:
        adjusted_dose = baseline_dose_mg * 2.0

    # Cap at 3x baseline
    adjusted_dose = min(adjusted_dose, baseline_dose_mg * 3.0)

    # --- Clinical significance ---
    if bio_ratio < 0.5:
        clinical_significance = "major"
    elif bio_ratio < 0.7:
        clinical_significance = "moderate"
    elif bio_ratio < 0.9:
        clinical_significance = "minor"
    else:
        clinical_significance = "none"

    # --- Management strategy ---
    if clinical_significance == "major":
        management_strategy = "hold_feeding_1h"
    elif clinical_significance == "moderate":
        management_strategy = "monitor"
    elif clinical_significance == "minor":
        management_strategy = "monitor"
    else:
        management_strategy = "no_adjustment"

    # --- Notes ---
    notes = (
        f"Bioavailability ratio {bio_ratio:.2f} (EN vs. fasted). "
        f"Binding loss: {binding_loss_pct:.1f}%. "
        f"Management: {management_strategy}."
    )

    return EnteralNutritionPKResult(
        drug_name=drug_name,
        formula_type=formula_type,
        feeding_method=feeding_method,
        baseline_dose_mg=baseline_dose_mg,
        adjusted_dose_mg=adjusted_dose,
        bioavailability_baseline=baseline_bioavailability,
        bioavailability_with_en=bio_with_en,
        bioavailability_ratio=bio_ratio,
        binding_loss_pct=binding_loss_pct,
        gastric_emptying_factor=gastric_emptying,
        clinical_significance=clinical_significance,
        management_strategy=management_strategy,
        notes=notes,
    )


def screen_formulas(
    drug_name: str,
    baseline_dose_mg: float,
    baseline_bioavailability: float,
    drug_pka: float = 7.0,
    ionization_type: str = "neutral",
    drug_mw_Da: float = 300.0,
    logp: float = 1.0,
) -> list[EnteralNutritionPKResult]:
    """Screen all 4 formulas with continuous feeding, sorted by bioavailability_ratio descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Dose without EN adjustment (mg).
    baseline_bioavailability:
        Fraction absorbed without EN (0 < F <= 1).
    drug_pka:
        Drug pKa value.
    ionization_type:
        "acid", "base", or "neutral".
    drug_mw_Da:
        Molecular weight in Daltons.
    logp:
        Octanol-water partition coefficient.

    Returns
    -------
    List of EnteralNutritionPKResult for all 4 formulas, sorted best first.
    """
    results = []
    for formula_type in FORMULAS:
        result = assess_enteral_nutrition_pk(
            drug_name=drug_name,
            formula_type=formula_type,
            feeding_method="continuous",
            baseline_dose_mg=baseline_dose_mg,
            baseline_bioavailability=baseline_bioavailability,
            drug_pka=drug_pka,
            ionization_type=ionization_type,
            drug_mw_Da=drug_mw_Da,
            logp=logp,
        )
        results.append(result)

    # Sort descending by bioavailability_ratio (best = least interaction first)
    results.sort(key=lambda r: r.bioavailability_ratio, reverse=True)
    return results
