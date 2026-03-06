"""Drug-drug interaction severity scoring module."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "InteractionScoreResult",
    "score_cyp_perpetrator",
    "score_cyp_victim",
    "calculate_interaction_score",
]


@dataclass
class InteractionScoreResult:
    drug_a: str
    drug_b: str
    cyp_perpetrator_score: float
    cyp_victim_score: float
    transporter_score: float
    combined_risk_score: float
    severity: str
    aucr_estimate: float
    cmax_ratio_estimate: float
    clinical_significance: str
    management_recommendation: str


def score_cyp_perpetrator(
    drug_name: str,
    ki_uM: float | None = None,
    induction_fold: float = 1.0,
    logP: float = 2.0,
    MW: float = 300.0,
) -> float:
    """Score CYP inhibition/induction perpetrator risk (0-10).

    High inhibition: ki_uM <= 1 → base 7
    Medium inhibition: ki_uM <= 10 → base 4
    Low inhibition: ki_uM > 10 → base 1
    No ki provided → base 0
    Induction: += 3 if induction_fold > 2
    LogP risk: += 1 if logP > 3
    Capped at 10.0.
    """
    score = 0.0

    if ki_uM is not None:
        if ki_uM <= 1.0:
            score += 7.0
        elif ki_uM <= 10.0:
            score += 4.0
        else:
            score += 1.0

    if induction_fold > 2.0:
        score += 3.0

    if logP > 3.0:
        score += 1.0

    return min(score, 10.0)


def score_cyp_victim(
    drug_name: str,
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.0,
    fm_cyp2c9: float = 0.0,
    narrow_therapeutic_index: bool = False,
) -> float:
    """Score CYP victim susceptibility (0-10).

    Weighted sum: fm_cyp3a4*4 + fm_cyp2d6*3 + fm_cyp2c9*2
    NTI multiplier: 1.5 if narrow_therapeutic_index
    Capped at 10.0.
    """
    score = fm_cyp3a4 * 4.0 + fm_cyp2d6 * 3.0 + fm_cyp2c9 * 2.0

    if narrow_therapeutic_index:
        score *= 1.5

    return min(score, 10.0)


def calculate_interaction_score(
    drug_a: str,
    drug_b: str,
    perpetrator_score: float,
    victim_score: float,
    transporter_score: float = 0.0,
) -> InteractionScoreResult:
    """Calculate overall drug-drug interaction severity score.

    combined_risk = (perpetrator_score + victim_score + transporter_score) / 3
    severity: >=8 → contraindicated, >=6 → major, >=4 → moderate, else minor
    aucr_estimate = 1 + combined_risk * 0.3
    cmax_ratio_estimate = 1 + combined_risk * 0.2

    Raises ValueError if any score is outside [0, 10].
    """
    for name, value in [
        ("perpetrator_score", perpetrator_score),
        ("victim_score", victim_score),
        ("transporter_score", transporter_score),
    ]:
        if value < 0.0 or value > 10.0:
            raise ValueError(f"{name} must be between 0 and 10, got {value}")

    combined_risk = (perpetrator_score + victim_score + transporter_score) / 3.0

    if combined_risk >= 8.0:
        severity = "contraindicated"
        clinical_significance = (
            "Combination poses unacceptable risk; pharmacokinetic interaction "
            "likely to cause serious adverse events or treatment failure."
        )
        management_recommendation = "avoid combination"
    elif combined_risk >= 6.0:
        severity = "major"
        clinical_significance = (
            "Clinically significant interaction expected; benefit-risk assessment "
            "required before co-administration."
        )
        management_recommendation = "monitor closely"
    elif combined_risk >= 4.0:
        severity = "moderate"
        clinical_significance = (
            "Moderate interaction potential; dosage adjustment or enhanced "
            "monitoring may be warranted."
        )
        management_recommendation = "monitor"
    else:
        severity = "minor"
        clinical_significance = (
            "Minimal clinical impact expected; interaction unlikely to require "
            "intervention under standard dosing."
        )
        management_recommendation = "routine monitoring"

    aucr_estimate = 1.0 + combined_risk * 0.3
    cmax_ratio_estimate = 1.0 + combined_risk * 0.2

    return InteractionScoreResult(
        drug_a=drug_a,
        drug_b=drug_b,
        cyp_perpetrator_score=perpetrator_score,
        cyp_victim_score=victim_score,
        transporter_score=transporter_score,
        combined_risk_score=combined_risk,
        severity=severity,
        aucr_estimate=aucr_estimate,
        cmax_ratio_estimate=cmax_ratio_estimate,
        clinical_significance=clinical_significance,
        management_recommendation=management_recommendation,
    )
