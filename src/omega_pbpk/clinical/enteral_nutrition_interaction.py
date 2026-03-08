"""Phase 313 — Enteral Nutrition Drug Interaction Predictor.

Predicts PK changes when drugs are co-administered with enteral nutrition
(tube feeding). Models absorption changes due to physical binding, pH
effects, and gastric motility changes.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_FORMULATIONS = {"tablet", "capsule", "liquid", "crushable"}
_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class EnteralNutritionResult:
    """Result from enteral nutrition drug interaction prediction."""

    drug_name: str
    formulation_type: str
    en_rate_mL_per_h: float
    protein_binding_risk: bool
    relative_bioavailability: float
    auc_ratio: float
    cmax_ratio: float
    tmax_delay_h: float
    interaction_severity: str
    clinical_recommendation: str
    notes: str


def predict_en_interaction(
    drug_name: str,
    drug_pka: float,
    drug_ionization_type: str,
    drug_logp: float,
    formulation_type: str,
    en_rate_mL_per_h: float,
    protein_binding_risk: bool,
) -> EnteralNutritionResult:
    """Predict PK changes when a drug is co-administered with enteral nutrition.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_pka:
        pKa of the drug (0, 14].
    drug_ionization_type:
        Ionization type: "acid", "base", or "neutral".
    drug_logp:
        Lipophilicity (log P).
    formulation_type:
        Dosage form: "tablet", "capsule", "liquid", or "crushable".
    en_rate_mL_per_h:
        Enteral nutrition infusion rate in mL/h; 0 means bolus (normal meal).
    protein_binding_risk:
        True if drug binds to EN proteins (e.g., phenytoin, warfarin).

    Returns
    -------
    EnteralNutritionResult
    """
    # Validate inputs
    if en_rate_mL_per_h < 0:
        raise ValueError("en_rate_mL_per_h must be >= 0")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {_VALID_FORMULATIONS}, got '{formulation_type}'"
        )
    if drug_ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"drug_ionization_type must be one of {_VALID_IONIZATION_TYPES}, "
            f"got '{drug_ionization_type}'"
        )
    if not (0 < drug_pka <= 14):
        raise ValueError("drug_pka must be in (0, 14]")

    # Formulation pH effect
    if formulation_type == "tablet":
        ph_effect = 0.1
    elif formulation_type == "crushable":
        ph_effect = 0.05
    else:
        # liquid or capsule
        ph_effect = 0.0

    # Protein binding effect
    binding_reduction = 0.3 if protein_binding_risk else 0.0

    # EN dilution effect
    dilution_effect = min(0.2, en_rate_mL_per_h / 500.0) * 0.3

    # Motility acceleration effect
    motility_effect = 0.1 if en_rate_mL_per_h > 100 else 0.0

    # pH correction for basic drugs (EN raises gastric pH → less ionized → better)
    ph_correction = 0.0
    if drug_pka > 7.4 and drug_ionization_type == "base":
        ph_correction = 0.1

    # Relative bioavailability
    relative_bioavailability = max(
        0.1,
        1.0 - ph_effect - binding_reduction - dilution_effect - motility_effect + ph_correction,
    )

    auc_ratio = relative_bioavailability
    cmax_ratio = relative_bioavailability * 0.85
    tmax_delay_h = motility_effect * 2.0 + dilution_effect * 1.0

    # Interaction severity
    if auc_ratio < 0.7:
        interaction_severity = "major"
    elif auc_ratio < 0.85:
        interaction_severity = "moderate"
    else:
        interaction_severity = "minor"

    # Clinical recommendation (priority order)
    if auc_ratio < 0.7:
        clinical_recommendation = "hold_EN_around_dose"
    elif auc_ratio < 0.85:
        clinical_recommendation = "monitor_drug_levels"
    elif protein_binding_risk:
        clinical_recommendation = "separate_by_2h"
    else:
        clinical_recommendation = "no_special_precaution"

    # Notes
    notes_parts: list[str] = []
    if protein_binding_risk:
        notes_parts.append("Drug binds EN proteins; separate administration by 2h.")
    if motility_effect > 0:
        notes_parts.append("High EN rate accelerates gastric motility.")
    if ph_correction > 0:
        notes_parts.append("Basic drug may have improved absorption due to EN-mediated pH rise.")
    if formulation_type == "tablet":
        notes_parts.append("Tablet crushing may affect release profile.")
    notes = " ".join(notes_parts) if notes_parts else "No special notes."

    return EnteralNutritionResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        en_rate_mL_per_h=en_rate_mL_per_h,
        protein_binding_risk=protein_binding_risk,
        relative_bioavailability=relative_bioavailability,
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        tmax_delay_h=tmax_delay_h,
        interaction_severity=interaction_severity,
        clinical_recommendation=clinical_recommendation,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    drug_pka: float,
    drug_ionization_type: str,
    drug_logp: float,
    en_rate_mL_per_h: float,
) -> list[EnteralNutritionResult]:
    """Compare all formulation types for a drug under the same EN conditions.

    Returns a list of EnteralNutritionResult sorted by
    relative_bioavailability descending.
    """
    results: list[EnteralNutritionResult] = []
    for formulation in _VALID_FORMULATIONS:
        result = predict_en_interaction(
            drug_name=drug_name,
            drug_pka=drug_pka,
            drug_ionization_type=drug_ionization_type,
            drug_logp=drug_logp,
            formulation_type=formulation,
            en_rate_mL_per_h=en_rate_mL_per_h,
            protein_binding_risk=False,
        )
        results.append(result)
    results.sort(key=lambda r: r.relative_bioavailability, reverse=True)
    return results
