"""Drug-induced cardiotoxicity risk predictor.

Phase 840: Predicts cardiotoxicity risk from physicochemical and pharmacological
properties using a scoring model covering hERG, QT prolongation, negative inotropy,
arrhythmia, and vascular risks.

References
----------
- Redfern WS et al., Cardiovasc Res. 2003;58:32-45 (hERG/QT safety pharmacology)
- Laverty HG et al., Br J Pharmacol. 2011;163:675-693 (cardiac safety testing)
- FDA Guidance: E14 Clinical Evaluation of QT/QTc Interval Prolongation. 2005.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CardiotoxicityResult",
    "predict_cardiotoxicity",
    "screen_cardiotoxicity",
]

_VALID_DRUG_CLASSES = {"antiarrhythmic", "antihistamine", "antipsychotic", "antibiotic", "other"}

_DRUG_CLASS_SCORES: dict[str, float] = {
    "antiarrhythmic": 15.0,
    "antihistamine": 10.0,
    "antipsychotic": 10.0,
    "antibiotic": 5.0,
    "other": 0.0,
}


@dataclass(frozen=True)
class CardiotoxicityResult:
    """Comprehensive cardiotoxicity risk assessment result."""

    drug_name: str
    overall_risk: str  # "low", "moderate", "high", "critical"
    risk_score: float  # 0-100
    herg_risk: str  # "low", "moderate", "high"
    structural_alert_count: int
    structural_alerts: str  # comma-separated list
    qt_prolongation_risk: str  # "low", "moderate", "high"
    negative_inotropy_risk: str  # "low", "moderate", "high"
    arrhythmia_risk: str  # "low", "moderate", "high"
    vascular_risk: str  # "low", "moderate", "high"
    monitoring_recommendation: str
    contraindications: str
    notes: str


def _classify_risk(score: float, high_thresh: float, moderate_thresh: float) -> str:
    """Classify sub-risk as high/moderate/low based on score thresholds."""
    if score > high_thresh:
        return "high"
    if score > moderate_thresh:
        return "moderate"
    return "low"


def predict_cardiotoxicity(
    drug_name: str,
    logp: float,
    mw_da: float,
    pka: float,
    has_amino_group: bool,
    has_chloro: bool,
    has_aromatic: bool,
    drug_class: str = "other",
    cyp_inhibitor: bool = False,
) -> CardiotoxicityResult:
    """Predict drug-induced cardiotoxicity risk.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient (log P). Must be in [-5, 10].
    mw_da:
        Molecular weight in Daltons.
    pka:
        pKa of the most basic ionizable group. Must be in [0, 14].
    has_amino_group:
        Whether the drug contains an amino (amine) group.
    has_chloro:
        Whether the drug contains a chloro substituent.
    has_aromatic:
        Whether the drug has aromatic ring(s).
    drug_class:
        Therapeutic class; one of "antiarrhythmic", "antihistamine",
        "antipsychotic", "antibiotic", "other".
    cyp_inhibitor:
        Whether the drug inhibits CYP enzymes (increases own exposure via DDI).

    Returns
    -------
    CardiotoxicityResult
        Comprehensive cardiotoxicity risk assessment.
    """
    if logp < -5 or logp > 10:
        raise ValueError(f"logp must be in [-5, 10], got {logp}")
    if pka < 0 or pka > 14:
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"Invalid drug_class '{drug_class}'. Choose from: {sorted(_VALID_DRUG_CLASSES)}"
        )

    score = 10.0  # base score

    # ---------------------------------------------------------------------------
    # hERG risk scoring (physicochemical proxy for hERG channel binding)
    # ---------------------------------------------------------------------------
    if logp > 3 and pka > 7:
        score += 20.0
    elif logp > 2 and pka > 7:
        score += 10.0

    # ---------------------------------------------------------------------------
    # Structural alerts
    # ---------------------------------------------------------------------------
    alerts: list[str] = []

    if has_amino_group and logp > 2:
        alerts.append("aminoalkyl_side_chain")
        score += 15.0

    if has_chloro and has_aromatic:
        alerts.append("halogenated_aromatic")
        score += 10.0

    if mw_da > 500 and pka > 7:
        alerts.append("high_mw_basic")
        score += 10.0

    # ---------------------------------------------------------------------------
    # Drug class risk
    # ---------------------------------------------------------------------------
    score += _DRUG_CLASS_SCORES[drug_class]

    # ---------------------------------------------------------------------------
    # CYP inhibitor (raises own plasma exposure → greater cardiac exposure)
    # ---------------------------------------------------------------------------
    if cyp_inhibitor:
        score += 10.0

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))

    # ---------------------------------------------------------------------------
    # Sub-risk classification
    # ---------------------------------------------------------------------------
    herg_risk = _classify_risk(score, 50.0, 25.0)
    qt_prolongation_risk = _classify_risk(score, 50.0, 25.0)
    negative_inotropy_risk = _classify_risk(score, 60.0, 35.0)
    arrhythmia_risk = _classify_risk(score, 55.0, 30.0)
    vascular_risk = _classify_risk(score, 70.0, 45.0)

    # ---------------------------------------------------------------------------
    # Overall risk category
    # ---------------------------------------------------------------------------
    if score > 75:
        overall_risk = "critical"
    elif score > 50:
        overall_risk = "high"
    elif score > 25:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    # ---------------------------------------------------------------------------
    # Monitoring recommendation
    # ---------------------------------------------------------------------------
    if score > 50:
        monitoring_recommendation = "cardiac monitoring required"
    elif score > 25:
        monitoring_recommendation = "ECG at baseline"
    else:
        monitoring_recommendation = "standard monitoring"

    # ---------------------------------------------------------------------------
    # Contraindications
    # ---------------------------------------------------------------------------
    if herg_risk == "high":
        contraindications = "QT-prolonging drugs"
    else:
        contraindications = ""

    structural_alerts_str = ", ".join(alerts) if alerts else ""

    notes = (
        f"logP={logp:.2f}; MW={mw_da:.1f} Da; pKa={pka:.2f}; "
        f"class={drug_class}; CYP_inhibitor={cyp_inhibitor}"
    )

    return CardiotoxicityResult(
        drug_name=drug_name,
        overall_risk=overall_risk,
        risk_score=score,
        herg_risk=herg_risk,
        structural_alert_count=len(alerts),
        structural_alerts=structural_alerts_str,
        qt_prolongation_risk=qt_prolongation_risk,
        negative_inotropy_risk=negative_inotropy_risk,
        arrhythmia_risk=arrhythmia_risk,
        vascular_risk=vascular_risk,
        monitoring_recommendation=monitoring_recommendation,
        contraindications=contraindications,
        notes=notes,
    )


def screen_cardiotoxicity(drug_candidates: list[dict]) -> list[CardiotoxicityResult]:
    """Screen a list of drug candidates for cardiotoxicity risk.

    Parameters
    ----------
    drug_candidates:
        List of dicts, each containing the keyword arguments for
        :func:`predict_cardiotoxicity`.

    Returns
    -------
    list[CardiotoxicityResult]
        Results sorted by risk_score descending (highest risk first).
    """
    results = [predict_cardiotoxicity(**candidate) for candidate in drug_candidates]
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
