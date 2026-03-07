"""Phase 304 — Drug Metabolite Safety Assessment.

Assess safety risk of drug metabolites per FDA MIST guidance.
Determines if metabolites require standalone nonclinical studies based on their
exposure relative to parent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MIST_THRESHOLD = 0.10  # FDA 10% of parent AUC
_UNIQUE_HUMAN_THRESHOLD = 0.05  # Lower threshold for unique human metabolites


@dataclass(frozen=True)
class MetaboliteSafetyResult:
    """Result of metabolite safety assessment per FDA MIST guidance."""

    drug_name: str
    metabolite_name: str
    parent_auc_mg_h_per_L: float
    metabolite_auc_mg_h_per_L: float
    exposure_ratio: float
    above_mist_threshold: bool
    standalone_study_required: bool
    risk_score: float
    risk_class: str
    safety_margin: float
    regulatory_recommendation: str
    notes: str


def _validate_inputs(
    parent_auc_mg_h_per_L: float,
    metabolite_auc_mg_h_per_L: float,
    metabolite_mw: float,
    parent_dose_mg: float,
) -> None:
    if parent_auc_mg_h_per_L <= 0:
        raise ValueError(f"parent_auc_mg_h_per_L must be > 0, got {parent_auc_mg_h_per_L}")
    if metabolite_auc_mg_h_per_L < 0:
        raise ValueError(f"metabolite_auc_mg_h_per_L must be >= 0, got {metabolite_auc_mg_h_per_L}")
    if metabolite_mw <= 0:
        raise ValueError(f"metabolite_mw must be > 0, got {metabolite_mw}")
    if parent_dose_mg <= 0:
        raise ValueError(f"parent_dose_mg must be > 0, got {parent_dose_mg}")


def assess_metabolite_safety(
    drug_name: str,
    metabolite_name: str,
    parent_auc_mg_h_per_L: float,
    metabolite_auc_mg_h_per_L: float,
    metabolite_mw: float,
    metabolite_logp: float,
    parent_dose_mg: float,
    is_unique_human_metabolite: bool,
    has_pharmacological_activity: bool,
) -> MetaboliteSafetyResult:
    """Assess safety risk of a drug metabolite per FDA MIST guidance.

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    metabolite_name:
        Name of the metabolite.
    parent_auc_mg_h_per_L:
        Parent drug AUC (mg*h/L). Must be > 0.
    metabolite_auc_mg_h_per_L:
        Metabolite AUC (mg*h/L). Must be >= 0.
    metabolite_mw:
        Metabolite molecular weight (Da). Must be > 0.
    metabolite_logp:
        Metabolite logP.
    parent_dose_mg:
        Parent drug dose (mg). Must be > 0.
    is_unique_human_metabolite:
        True if the metabolite is unique to humans (not observed in animal species).
    has_pharmacological_activity:
        True if the metabolite has pharmacological activity.

    Returns
    -------
    MetaboliteSafetyResult
    """
    _validate_inputs(
        parent_auc_mg_h_per_L, metabolite_auc_mg_h_per_L, metabolite_mw, parent_dose_mg
    )

    # Exposure ratio
    exposure_ratio = metabolite_auc_mg_h_per_L / parent_auc_mg_h_per_L

    # MIST threshold assessment
    above_mist_threshold = exposure_ratio >= _MIST_THRESHOLD

    # Standalone study requirement
    standalone_study_required = above_mist_threshold or (
        is_unique_human_metabolite and exposure_ratio >= _UNIQUE_HUMAN_THRESHOLD
    )

    # Risk score (0-100)
    risk_score = min(50.0, exposure_ratio * 50.0)

    if has_pharmacological_activity:
        risk_score += 20.0
    if is_unique_human_metabolite:
        risk_score += 15.0
    if metabolite_logp > 3 and metabolite_mw < 300:
        risk_score += 10.0  # Small lipophilic → reactive risk
    if above_mist_threshold:
        risk_score += 5.0

    risk_score = min(100.0, risk_score)

    # Risk class
    if risk_score >= 60.0:
        risk_class = "high"
    elif risk_score >= 30.0:
        risk_class = "moderate"
    else:
        risk_class = "low"

    # Safety margin
    if metabolite_auc_mg_h_per_L == 0:
        safety_margin = math.inf
    else:
        safety_margin = parent_auc_mg_h_per_L / metabolite_auc_mg_h_per_L

    # Regulatory recommendation
    if standalone_study_required:
        regulatory_recommendation = "standalone_nonclinical_required"
    elif risk_score >= 30.0:
        regulatory_recommendation = "monitor_only"
    else:
        regulatory_recommendation = "no_action_required"

    # Notes
    note_parts: list[str] = []
    if above_mist_threshold:
        note_parts.append(f"Exposure ratio {exposure_ratio:.2f} exceeds FDA MIST 10% threshold")
    if is_unique_human_metabolite:
        note_parts.append("Unique human metabolite: lower threshold (5%) applies")
    if has_pharmacological_activity:
        note_parts.append("Pharmacologically active metabolite: elevated risk")
    if metabolite_logp > 3 and metabolite_mw < 300:
        note_parts.append("Small lipophilic structure: reactive metabolite alert")
    notes = "; ".join(note_parts) if note_parts else "No significant safety flags identified"

    return MetaboliteSafetyResult(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        parent_auc_mg_h_per_L=parent_auc_mg_h_per_L,
        metabolite_auc_mg_h_per_L=metabolite_auc_mg_h_per_L,
        exposure_ratio=exposure_ratio,
        above_mist_threshold=above_mist_threshold,
        standalone_study_required=standalone_study_required,
        risk_score=risk_score,
        risk_class=risk_class,
        safety_margin=safety_margin,
        regulatory_recommendation=regulatory_recommendation,
        notes=notes,
    )


def screen_metabolites(
    drug_name: str,
    parent_auc_mg_h_per_L: float,
    parent_dose_mg: float,
    metabolites_data: list[dict],
) -> list[MetaboliteSafetyResult]:
    """Screen multiple metabolites and rank by risk score.

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    parent_auc_mg_h_per_L:
        Parent drug AUC (mg*h/L).
    parent_dose_mg:
        Parent drug dose (mg).
    metabolites_data:
        List of dicts, each with keys:
        - metabolite_name (str)
        - metabolite_auc_mg_h_per_L (float)
        - metabolite_mw (float)
        - metabolite_logp (float)
        - is_unique_human_metabolite (bool)
        - has_pharmacological_activity (bool)

    Returns
    -------
    list[MetaboliteSafetyResult] sorted by risk_score descending.
    """
    results: list[MetaboliteSafetyResult] = []
    for m in metabolites_data:
        result = assess_metabolite_safety(
            drug_name=drug_name,
            metabolite_name=m["metabolite_name"],
            parent_auc_mg_h_per_L=parent_auc_mg_h_per_L,
            metabolite_auc_mg_h_per_L=m["metabolite_auc_mg_h_per_L"],
            metabolite_mw=m["metabolite_mw"],
            metabolite_logp=m["metabolite_logp"],
            parent_dose_mg=parent_dose_mg,
            is_unique_human_metabolite=m["is_unique_human_metabolite"],
            has_pharmacological_activity=m["has_pharmacological_activity"],
        )
        results.append(result)

    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
