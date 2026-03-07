"""
Phase 604 — Drug-Induced Liver Injury (DILI) Risk Prediction

Predicts DILI risk from structural and metabolic features.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DILIResult:
    drug_name: str
    dili_risk_score: float  # 0-100
    dili_concern_level: str  # "no_concern","low","moderate","high","most_concerned"
    reactive_metabolite_risk: bool
    mitochondrial_risk: bool
    bile_salt_transporter_risk: bool  # BSEP inhibition risk
    oxidative_stress_risk: bool
    dose_dependent_risk: bool
    structural_alerts: list  # list of alert names triggered
    daily_dose_mg: float
    hepatic_auc_proxy: float  # daily_dose * f_oral * (1 - fu)
    confidence: str  # "high","medium","low"
    notes: str


def _compute_structural_alerts(
    mw: float,
    logP: float,
    n_aromatic_rings: int,
    psa: float,
    n_hbd: int,
) -> list:
    alerts = []
    if mw < 250 and n_aromatic_rings >= 1 and logP < 2:
        alerts.append("aniline")
    if logP < 0 and n_aromatic_rings >= 1:
        alerts.append("nitro_group")
    if logP > 3 and n_aromatic_rings >= 2:
        alerts.append("halogenated_aromatic")
    if n_aromatic_rings >= 2 and psa < 60:
        alerts.append("quinone_precursor")
    if logP > 1 and mw > 200 and n_hbd == 0:
        alerts.append("michael_acceptor")
    if n_aromatic_rings >= 3:
        alerts.append("epoxide_risk")
    return alerts


def dili_score_from_features(features: dict) -> float:
    """
    Compute DILI risk score 0-100 from feature dict.

    Expected keys:
        reactive_metabolite (bool), mitochondrial (bool),
        bsep_ic50 (float | None), n_alerts (int),
        daily_dose (float), logP (float)
    """
    score = 0.0

    if features.get("reactive_metabolite", False):
        score += 25
    if features.get("mitochondrial", False):
        score += 20

    bsep = features.get("bsep_ic50", None)
    if bsep is not None:
        if bsep < 10:
            score += 20
        elif bsep < 100:
            score += 10

    n_alerts = features.get("n_alerts", 0)
    score += n_alerts * 5

    daily_dose = features.get("daily_dose", 0.0)
    if daily_dose > 100:
        score += 15
    if daily_dose > 500:
        score += 10

    logP = features.get("logP", 0.0)
    if logP > 3:
        score += 5

    return max(0.0, min(100.0, score))


def predict_dili_risk(
    drug_name: str,
    daily_dose_mg: float,
    logP: float,
    mw: float,
    n_aromatic_rings: int = 0,
    has_reactive_metabolite: bool = False,
    has_mitochondrial_liability: bool = False,
    bsep_ic50_uM: float | None = None,
    fu_plasma: float = 0.5,
    f_oral: float = 0.8,
    psa: float = 80.0,
    n_hbd: int = 2,
) -> DILIResult:
    """Predict DILI risk from structural and metabolic features."""
    # Structural alerts
    structural_alerts = _compute_structural_alerts(mw, logP, n_aromatic_rings, psa, n_hbd)

    # Risk flags
    reactive_metabolite_risk = has_reactive_metabolite
    mitochondrial_risk = has_mitochondrial_liability

    # Scoring
    score = 0.0

    if reactive_metabolite_risk:
        score += 25
    if mitochondrial_risk:
        score += 20

    # BSEP inhibition
    bile_salt_transporter_risk = False
    if bsep_ic50_uM is not None:
        if bsep_ic50_uM < 10:
            bile_salt_transporter_risk = True
            score += 20
        elif bsep_ic50_uM < 100:
            score += 10

    # Structural alerts
    score += len(structural_alerts) * 5

    # Dose-dependent
    dose_dependent_risk = daily_dose_mg > 100
    if daily_dose_mg > 100:
        score += 15
    if daily_dose_mg > 500:
        score += 10

    # Lipophilicity
    if logP > 3:
        score += 5

    # Clamp
    dili_risk_score = max(0.0, min(100.0, score))

    # Derived flags
    oxidative_stress_risk = reactive_metabolite_risk or (n_aromatic_rings >= 2)
    hepatic_auc_proxy = daily_dose_mg * f_oral * (1.0 - fu_plasma)

    # Concern level
    if dili_risk_score < 20:
        dili_concern_level = "no_concern"
    elif dili_risk_score < 40:
        dili_concern_level = "low"
    elif dili_risk_score < 60:
        dili_concern_level = "moderate"
    elif dili_risk_score < 80:
        dili_concern_level = "high"
    else:
        dili_concern_level = "most_concerned"

    # Confidence
    if has_reactive_metabolite or has_mitochondrial_liability:
        confidence = "high"
    elif bsep_ic50_uM is not None:
        confidence = "medium"
    else:
        confidence = "low"

    notes = (
        f"DILI score={dili_risk_score:.1f}; concern={dili_concern_level}; "
        f"alerts={structural_alerts}; confidence={confidence}"
    )

    return DILIResult(
        drug_name=drug_name,
        dili_risk_score=dili_risk_score,
        dili_concern_level=dili_concern_level,
        reactive_metabolite_risk=reactive_metabolite_risk,
        mitochondrial_risk=mitochondrial_risk,
        bile_salt_transporter_risk=bile_salt_transporter_risk,
        oxidative_stress_risk=oxidative_stress_risk,
        dose_dependent_risk=dose_dependent_risk,
        structural_alerts=structural_alerts,
        daily_dose_mg=daily_dose_mg,
        hepatic_auc_proxy=hepatic_auc_proxy,
        confidence=confidence,
        notes=notes,
    )
