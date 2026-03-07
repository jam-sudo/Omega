"""Drug Purity and Impurity Impact Predictor (Phase 760).

Assesses pharmaceutical impurities against ICH Q3A/Q3B reporting,
qualification, and identification thresholds.  Also evaluates genotoxic
impurities against the ICH M7 threshold of toxicological concern (TTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# ICH Q3A thresholds
# ---------------------------------------------------------------------------

_REPORTING_PCT = 0.05  # %
_QUALIFICATION_PCT = 0.10  # %
_REPORTING_MG = 1.0  # mg (TDI)
_QUALIFICATION_MG = 1.0  # mg (TDI)

# ICH M7 threshold of toxicological concern for genotoxic impurities
_TTC_UG_PER_DAY = 1.5  # µg/day  (= 0.0015 mg/day)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpurityImpactResult:
    """Result from impurity impact assessment."""

    drug_name: str
    impurity_name: str
    daily_dose_drug_mg: float
    impurity_pct: float
    daily_dose_impurity_mg: float
    effective_pharmacological_dose_mg: float
    reporting_threshold_exceeded: bool
    qualification_threshold_exceeded: bool
    ich_classification: str
    genotoxic_risk: str
    regulatory_action_required: str
    notes: str


# ---------------------------------------------------------------------------
# Core assessment
# ---------------------------------------------------------------------------


def assess_impurity_impact(
    drug_name: str,
    impurity_name: str,
    daily_dose_drug_mg: float,
    impurity_pct: float,
    is_genotoxic: bool = False,
    potency_relative_to_parent: float = 1.0,
) -> ImpurityImpactResult:
    """Assess the impact of a drug impurity against ICH Q3A/M7 guidelines.

    Parameters
    ----------
    drug_name:
        Name of the parent drug substance.
    impurity_name:
        Name or identifier of the impurity.
    daily_dose_drug_mg:
        Daily dose of the drug substance (mg).
    impurity_pct:
        Impurity level as a percentage of drug substance (0–100 %).
    is_genotoxic:
        Whether the impurity is known/suspected genotoxic (ICH M7).
    potency_relative_to_parent:
        Pharmacological potency of the impurity relative to the parent drug.

    Returns
    -------
    ImpurityImpactResult
    """
    # --- Validation ---
    if daily_dose_drug_mg <= 0:
        raise ValueError(f"daily_dose_drug_mg must be > 0, got {daily_dose_drug_mg}")
    if not (0.0 <= impurity_pct <= 100.0):
        raise ValueError(f"impurity_pct must be between 0 and 100, got {impurity_pct}")
    if potency_relative_to_parent < 0:
        raise ValueError(
            f"potency_relative_to_parent must be >= 0, got {potency_relative_to_parent}"
        )

    # --- Impurity dose ---
    daily_dose_impurity_mg = daily_dose_drug_mg * (impurity_pct / 100.0)
    effective_pharmacological_dose_mg = daily_dose_impurity_mg * potency_relative_to_parent

    # --- ICH Q3A threshold checks ---
    reporting_threshold_exceeded = (
        impurity_pct > _REPORTING_PCT or daily_dose_impurity_mg > _REPORTING_MG
    )
    qualification_threshold_exceeded = (
        impurity_pct > _QUALIFICATION_PCT or daily_dose_impurity_mg > _QUALIFICATION_MG
    )

    # --- ICH classification ---
    if not reporting_threshold_exceeded:
        ich_classification = "below_reporting"
    elif not qualification_threshold_exceeded:
        ich_classification = "reporting"
    else:
        ich_classification = "qualification"

    # --- Genotoxic risk (ICH M7) ---
    if is_genotoxic:
        daily_dose_impurity_ug = daily_dose_impurity_mg * 1000.0
        if daily_dose_impurity_ug > _TTC_UG_PER_DAY:
            genotoxic_risk = "ttc_exceeded"
        else:
            genotoxic_risk = "none"
    else:
        genotoxic_risk = "not_applicable"

    # --- Regulatory action ---
    if is_genotoxic and genotoxic_risk == "ttc_exceeded":
        regulatory_action_required = "immediate_action_genotoxic"
    elif ich_classification == "below_reporting":
        regulatory_action_required = "none"
    elif ich_classification == "reporting":
        regulatory_action_required = "report_to_authority"
    else:
        regulatory_action_required = "qualification_study_required"

    # --- Notes ---
    notes = (
        f"Impurity '{impurity_name}' at {impurity_pct:.3f}% in {drug_name}. "
        f"Daily impurity dose: {daily_dose_impurity_mg:.4f} mg "
        f"(effective pharmacological dose: {effective_pharmacological_dose_mg:.4f} mg). "
        f"ICH Q3A classification: {ich_classification}. "
        f"Genotoxic risk: {genotoxic_risk}. "
        f"Regulatory action: {regulatory_action_required}."
    )

    return ImpurityImpactResult(
        drug_name=drug_name,
        impurity_name=impurity_name,
        daily_dose_drug_mg=daily_dose_drug_mg,
        impurity_pct=impurity_pct,
        daily_dose_impurity_mg=daily_dose_impurity_mg,
        effective_pharmacological_dose_mg=effective_pharmacological_dose_mg,
        reporting_threshold_exceeded=reporting_threshold_exceeded,
        qualification_threshold_exceeded=qualification_threshold_exceeded,
        ich_classification=ich_classification,
        genotoxic_risk=genotoxic_risk,
        regulatory_action_required=regulatory_action_required,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def screen_impurities(
    drug_name: str,
    daily_dose_mg: float,
    impurities: list[dict[str, Any]],
) -> list[ImpurityImpactResult]:
    """Screen a list of impurities and return results sorted by impurity_pct descending.

    Parameters
    ----------
    drug_name:
        Name of the parent drug substance.
    daily_dose_mg:
        Daily dose of the drug substance (mg).
    impurities:
        List of dicts, each with keys:
        - impurity_name (str, required)
        - impurity_pct (float, required)
        - is_genotoxic (bool, optional, default False)
        - potency (float, optional, default 1.0)

    Returns
    -------
    List of ImpurityImpactResult sorted by impurity_pct descending.
    """
    results = []
    for imp in impurities:
        result = assess_impurity_impact(
            drug_name=drug_name,
            impurity_name=imp["impurity_name"],
            daily_dose_drug_mg=daily_dose_mg,
            impurity_pct=imp["impurity_pct"],
            is_genotoxic=imp.get("is_genotoxic", False),
            potency_relative_to_parent=imp.get("potency", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.impurity_pct, reverse=True)
    return results
