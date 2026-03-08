"""Phase 314 — Drug-Induced Kidney Injury (DIKI) Risk Predictor.

Predicts DIKI risk using structural, PK, and pharmacological features.
Identifies nephrotoxicity alerts and generates monitoring recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DIKIRiskResult:
    """Result from drug-induced kidney injury risk prediction."""

    drug_name: str
    logp: float
    mw: float
    renal_clearance_fraction: float
    daily_dose_mg: float
    risk_score: float
    aki_likelihood_pct: float
    risk_class: str
    monitoring_recommendation: str
    mitigation_strategies: list = field(default_factory=list)
    notes: str = ""


def predict_diki_risk(
    drug_name: str,
    logp: float,
    mw: float,
    renal_clearance_fraction: float,
    plasma_protein_binding_pct: float,
    is_nephrotoxic_class: bool,
    structural_alert_count_kidney: int,
    daily_dose_mg: float,
    fu_plasma: float,
) -> DIKIRiskResult:
    """Predict drug-induced kidney injury risk.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Lipophilicity (log P); range (-5, 10).
    mw:
        Molecular weight in Da; must be > 0.
    renal_clearance_fraction:
        Fraction of drug cleared renally [0, 1].
    plasma_protein_binding_pct:
        Plasma protein binding percentage [0, 100].
    is_nephrotoxic_class:
        True if the drug belongs to a known nephrotoxic class
        (NSAIDs, aminoglycosides, contrast agents, calcineurin inhibitors).
    structural_alert_count_kidney:
        Number of structural alerts associated with nephrotoxicity (>= 0).
    daily_dose_mg:
        Total daily dose in mg; must be > 0.
    fu_plasma:
        Unbound fraction in plasma (0, 1].

    Returns
    -------
    DIKIRiskResult
    """
    # Validate inputs
    if not (-5 < logp < 10):
        raise ValueError("logp must be in (-5, 10)")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (0 <= renal_clearance_fraction <= 1):
        raise ValueError("renal_clearance_fraction must be in [0, 1]")
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if structural_alert_count_kidney < 0:
        raise ValueError("structural_alert_count_kidney must be >= 0")

    # Scoring components
    # Renal elimination score
    if renal_clearance_fraction > 0.6:
        renal_score = 30
    elif renal_clearance_fraction >= 0.3:
        renal_score = 20
    else:
        renal_score = 5

    # Nephrotoxic class score
    class_score = 35 if is_nephrotoxic_class else 0

    # Structural alerts score
    alert_score = min(25, structural_alert_count_kidney * 10)

    # Dose score
    if daily_dose_mg > 1000:
        dose_score = 15
    elif daily_dose_mg >= 100:
        dose_score = 10
    elif daily_dose_mg >= 10:
        dose_score = 5
    else:
        dose_score = 0

    # Protein binding score
    ppb_score = 5 if plasma_protein_binding_pct > 95 else 0

    # Low fu_plasma score
    fu_score = 5 if fu_plasma < 0.05 else 0

    # Lipophilicity score
    lipophilicity_score = 5 if logp > 4 else 0

    # Total risk score
    risk_score = float(
        renal_score
        + class_score
        + alert_score
        + dose_score
        + ppb_score
        + fu_score
        + lipophilicity_score
    )

    # Risk class
    if risk_score >= 60:
        risk_class = "high"
    elif risk_score >= 30:
        risk_class = "moderate"
    else:
        risk_class = "low"

    # AKI likelihood percentage
    aki_likelihood_pct = min(90.0, risk_score * 100.0 / 120.0)

    # Monitoring recommendation
    if risk_class == "high":
        monitoring_recommendation = "daily_renal_panel"
    elif risk_class == "moderate":
        monitoring_recommendation = "weekly_creatinine"
    else:
        monitoring_recommendation = "monthly_creatinine"

    # Mitigation strategies
    mitigation_strategies: list[str] = []
    if is_nephrotoxic_class:
        mitigation_strategies.append("dose_reduce_in_renal_impairment")
    if renal_clearance_fraction > 0.6:
        mitigation_strategies.append("adjust_dose_for_gfr")
    mitigation_strategies.append("adequate_hydration")
    if daily_dose_mg > 1000:
        mitigation_strategies.append("consider_dose_split")

    # Notes
    notes_parts: list[str] = []
    if is_nephrotoxic_class:
        notes_parts.append("Drug belongs to a known nephrotoxic class.")
    if renal_clearance_fraction > 0.6:
        notes_parts.append("High renal elimination fraction increases exposure risk.")
    if structural_alert_count_kidney > 0:
        notes_parts.append(f"{structural_alert_count_kidney} structural alert(s) detected.")
    if logp > 4:
        notes_parts.append("High lipophilicity may cause tubular accumulation.")
    notes = " ".join(notes_parts) if notes_parts else "No specific nephrotoxicity flags."

    return DIKIRiskResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        renal_clearance_fraction=renal_clearance_fraction,
        daily_dose_mg=daily_dose_mg,
        risk_score=risk_score,
        aki_likelihood_pct=aki_likelihood_pct,
        risk_class=risk_class,
        monitoring_recommendation=monitoring_recommendation,
        mitigation_strategies=mitigation_strategies,
        notes=notes,
    )


def screen_compounds(compound_list: list[dict]) -> list[DIKIRiskResult]:
    """Screen multiple compounds and return results sorted by risk_score descending.

    Each dict in compound_list should contain the keyword arguments for
    predict_diki_risk.
    """
    results: list[DIKIRiskResult] = []
    for compound in compound_list:
        result = predict_diki_risk(**compound)
        results.append(result)
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
