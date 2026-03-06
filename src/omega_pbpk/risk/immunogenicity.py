"""Immunogenicity prediction for biologic drugs (mAbs, peptides, proteins).

References
----------
Jiskoot 2012, De Groot 2009, EMA guideline on immunogenicity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ImmunogenicityResult",
    "assess_immunogenicity",
    "compare_routes_immunogenicity",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROTEIN_TYPE_MULTIPLIERS: dict[str, float] = {
    "mab": 1.0,
    "adac": 1.5,
    "peptide": 2.0,
    "enzyme": 3.0,
    "cytokine": 2.5,
}

_ROUTE_MULTIPLIERS: dict[str, float] = {
    "iv": 0.7,
    "sc": 1.2,
    "im": 1.3,
}

_CONDITION_MULTIPLIERS: dict[str, float] = {
    "oncology": 0.7,
    "autoimmune": 1.2,
    "healthy": 1.5,
    "rare_disease": 1.0,
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImmunogenicityResult:
    """Complete immunogenicity assessment result."""

    protein_name: str
    protein_type: str
    mw_kDa: float
    humanization_pct: float
    route: str
    ada_risk_pct: float
    ada_risk_category: str
    clinical_impact_prediction: str
    mitigation_strategies: list[str]
    risk_drivers: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_immunogenicity(
    protein_name: str,
    mw_kDa: float,
    protein_type: str = "mab",
    peg_conjugated: bool = False,
    aggregation_score: float = 0.0,
    humanization_pct: float = 100.0,
    glycosylation: bool = True,
    route: str = "sc",
    treatment_duration_weeks: int = 52,
    patient_condition: str = "autoimmune",
) -> ImmunogenicityResult:
    """Assess immunogenicity risk for a biologic drug.

    Parameters
    ----------
    protein_name : str
        Name or identifier of the protein/biologic.
    mw_kDa : float
        Molecular weight in kDa.
    protein_type : str
        Type of biologic: "mab", "adac", "peptide", "enzyme", "cytokine".
    peg_conjugated : bool
        Whether the biologic is PEGylated.
    aggregation_score : float
        Aggregation propensity score (0-100).
    humanization_pct : float
        Percentage of human sequence (0-100).
    glycosylation : bool
        Whether the biologic is glycosylated.
    route : str
        Route of administration: "iv", "sc", "im".
    treatment_duration_weeks : int
        Duration of treatment in weeks.
    patient_condition : str
        Patient condition: "oncology", "autoimmune", "rare_disease", "healthy".

    Returns
    -------
    ImmunogenicityResult
    """
    if mw_kDa <= 0:
        raise ValueError("Molecular weight must be positive")
    if humanization_pct < 0 or humanization_pct > 100:
        raise ValueError("Humanization percentage must be between 0 and 100")
    if aggregation_score < 0:
        raise ValueError("Aggregation score must be non-negative")

    risk_drivers: list[str] = []

    # Base risk
    base_risk = 5.0

    # Protein type multiplier
    type_mult = _PROTEIN_TYPE_MULTIPLIERS.get(protein_type, 1.0)
    if type_mult > 1.1:
        risk_drivers.append(f"protein_type={protein_type} (x{type_mult})")
    base_risk *= type_mult

    # Humanization multiplier
    human_mult = 1 + (100 - humanization_pct) / 100 * 2
    if human_mult > 1.1:
        risk_drivers.append(f"humanization={humanization_pct}% (x{human_mult:.2f})")
    base_risk *= human_mult

    # Aggregation addition
    agg_addition = aggregation_score * 0.2
    if agg_addition > 1.0:
        risk_drivers.append(f"aggregation_score={aggregation_score} (+{agg_addition:.1f}%)")
    base_risk += agg_addition

    # PEGylation
    if peg_conjugated:
        base_risk *= 0.7
    elif base_risk > 10:
        risk_drivers.append("not PEGylated")

    # Glycosylation
    if glycosylation:
        base_risk *= 0.8
    else:
        risk_drivers.append("not glycosylated")
        base_risk *= 1.0  # no multiplier, just no reduction

    # Route
    route_mult = _ROUTE_MULTIPLIERS.get(route, 1.0)
    if route_mult > 1.1:
        risk_drivers.append(f"route={route} (x{route_mult})")
    base_risk *= route_mult

    # Treatment duration
    duration_addition = math.log(max(1, treatment_duration_weeks)) * 1.0
    if duration_addition > 2.0:
        risk_drivers.append(
            f"treatment_duration={treatment_duration_weeks}wk (+{duration_addition:.1f}%)"
        )
    base_risk += duration_addition

    # Patient condition
    cond_mult = _CONDITION_MULTIPLIERS.get(patient_condition, 1.0)
    if cond_mult > 1.1:
        risk_drivers.append(f"patient_condition={patient_condition} (x{cond_mult})")
    base_risk *= cond_mult

    # Clamp
    ada_risk_pct = min(95.0, max(1.0, base_risk))

    # Categories
    if ada_risk_pct < 5:
        ada_risk_category = "low"
    elif ada_risk_pct <= 20:
        ada_risk_category = "moderate"
    elif ada_risk_pct <= 50:
        ada_risk_category = "high"
    else:
        ada_risk_category = "very_high"

    if ada_risk_pct < 5:
        clinical_impact_prediction = "unlikely"
    elif ada_risk_pct <= 20:
        clinical_impact_prediction = "possible"
    elif ada_risk_pct <= 50:
        clinical_impact_prediction = "probable"
    else:
        clinical_impact_prediction = "expected"

    # Mitigation strategies
    mitigation_strategies: list[str] = []
    if ada_risk_pct < 5:
        mitigation_strategies = ["Immunogenicity risk is acceptable; routine monitoring sufficient"]
    else:
        if humanization_pct < 90:
            mitigation_strategies.append("Increase humanization; consider fully human sequence")
        if aggregation_score > 30:
            mitigation_strategies.append("Optimize formulation to reduce aggregation")
        if not peg_conjugated and ada_risk_pct > 20:
            mitigation_strategies.append("Consider PEGylation for immunogenicity reduction")
        if route == "sc" and ada_risk_pct > 20:
            mitigation_strategies.append("IV route may reduce immunogenicity")
        if patient_condition == "autoimmune":
            mitigation_strategies.append("Co-administration with immunosuppressant")
        if not mitigation_strategies:
            mitigation_strategies = [
                "Immunogenicity risk is acceptable; routine monitoring sufficient"
            ]

    # Notes
    note_parts: list[str] = []
    if risk_drivers:
        note_parts.append(f"Key risk drivers: {', '.join(risk_drivers[:3])}")
    if ada_risk_category in ("high", "very_high"):
        note_parts.append("Recommend ADA monitoring in clinical trials")
    notes = "; ".join(note_parts) if note_parts else "Low immunogenicity risk"

    return ImmunogenicityResult(
        protein_name=protein_name,
        protein_type=protein_type,
        mw_kDa=mw_kDa,
        humanization_pct=humanization_pct,
        route=route,
        ada_risk_pct=ada_risk_pct,
        ada_risk_category=ada_risk_category,
        clinical_impact_prediction=clinical_impact_prediction,
        mitigation_strategies=mitigation_strategies,
        risk_drivers=risk_drivers,
        notes=notes,
    )


def compare_routes_immunogenicity(
    protein_name: str,
    mw_kDa: float,
    **kwargs,
) -> list[ImmunogenicityResult]:
    """Compare immunogenicity risk across IV, SC, and IM routes.

    Returns
    -------
    list[ImmunogenicityResult]
        Results for IV, SC, IM routes.
    """
    results = []
    for route in ("iv", "sc", "im"):
        result = assess_immunogenicity(
            protein_name=protein_name,
            mw_kDa=mw_kDa,
            route=route,
            **kwargs,
        )
        results.append(result)
    return results
