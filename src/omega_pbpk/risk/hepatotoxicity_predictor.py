"""Phase 299 — Hepatotoxicity Risk Predictor.

Predicts DILI (drug-induced liver injury) risk based on structural alerts,
lipophilicity, reactive metabolite potential, and daily dose, aligned with
the DILIst database scoring approach.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepatotoxicityRiskResult:
    """Result of hepatotoxicity risk prediction."""

    drug_name: str
    logp: float
    mw: float
    daily_dose_mg: float
    fu_plasma: float
    has_reactive_metabolite: bool
    has_mitochondrial_liability: bool
    structural_alert_count: int
    risk_score: float
    dili_likelihood_pct: float
    risk_class: str
    mitigation_strategies: list
    notes: str


def _validate_inputs(
    logp: float,
    mw: float,
    daily_dose_mg: float,
    fu_plasma: float,
    structural_alert_count: int,
) -> None:
    """Validate input parameters."""
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if structural_alert_count < 0:
        raise ValueError(f"structural_alert_count must be >= 0, got {structural_alert_count}")


def predict_hepatotoxicity_risk(
    drug_name: str,
    logp: float,
    mw: float,
    daily_dose_mg: float,
    fu_plasma: float,
    has_reactive_metabolite: bool,
    has_mitochondrial_liability: bool,
    structural_alert_count: int,
) -> HepatotoxicityRiskResult:
    """Predict DILI risk for a single compound.

    Parameters
    ----------
    drug_name:
        Name of the drug or compound.
    logp:
        Octanol-water partition coefficient (must be in (-5, 10)).
    mw:
        Molecular weight in g/mol (must be > 0).
    daily_dose_mg:
        Total daily dose in mg (must be > 0).
    fu_plasma:
        Fraction unbound in plasma (must be in (0, 1]).
    has_reactive_metabolite:
        Whether the compound forms reactive metabolites.
    has_mitochondrial_liability:
        Whether the compound has mitochondrial liability.
    structural_alert_count:
        Number of structural alerts present (>= 0).

    Returns
    -------
    HepatotoxicityRiskResult
        Full risk assessment result.
    """
    _validate_inputs(logp, mw, daily_dose_mg, fu_plasma, structural_alert_count)

    # Dose score
    if daily_dose_mg > 100:
        dose_score = 30.0
    elif daily_dose_mg >= 50:
        dose_score = 20.0
    elif daily_dose_mg >= 10:
        dose_score = 10.0
    else:
        dose_score = 5.0

    # Lipophilicity score
    if logp > 3:
        lipo_score = 20.0
    elif logp >= 1:
        lipo_score = 10.0
    else:
        lipo_score = 0.0

    # Reactive metabolite score
    rm_score = 25.0 if has_reactive_metabolite else 0.0

    # Mitochondrial liability score
    mito_score = 20.0 if has_mitochondrial_liability else 0.0

    # Structural alerts score (capped at 30)
    alert_score = min(30.0, structural_alert_count * 10.0)

    # MW penalty
    mw_penalty = 5.0 if mw > 500 else 0.0

    # PPB penalty (extensive binding limits hepatic elimination)
    ppb_penalty = 5.0 if fu_plasma < 0.01 else 0.0

    # Total risk score (max 135)
    risk_score = (
        dose_score + lipo_score + rm_score + mito_score + alert_score + mw_penalty + ppb_penalty
    )

    # Risk classification
    if risk_score >= 60:
        risk_class = "high"
    elif risk_score >= 30:
        risk_class = "moderate"
    else:
        risk_class = "low"

    # DILI likelihood percentage
    dili_likelihood_pct = min(95.0, risk_score * 100.0 / 135.0)

    # Mitigation strategies
    mitigation_strategies: list = []
    if has_reactive_metabolite:
        mitigation_strategies.append("avoid_bioactivation_pathways")
    if daily_dose_mg > 100:
        mitigation_strategies.append("consider_dose_reduction")
    if logp > 3:
        mitigation_strategies.append("improve_hydrophilicity")
    if has_mitochondrial_liability:
        mitigation_strategies.append("assess_mtDNA_integrity")
    mitigation_strategies.append("monitor_liver_enzymes")

    # Build notes
    notes_parts = [
        f"risk_score={risk_score:.1f}",
        f"dose_score={dose_score:.0f}",
        f"lipo_score={lipo_score:.0f}",
        f"rm_score={rm_score:.0f}",
        f"mito_score={mito_score:.0f}",
        f"alert_score={alert_score:.0f}",
    ]
    notes = "; ".join(notes_parts)

    return HepatotoxicityRiskResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        fu_plasma=fu_plasma,
        has_reactive_metabolite=has_reactive_metabolite,
        has_mitochondrial_liability=has_mitochondrial_liability,
        structural_alert_count=structural_alert_count,
        risk_score=risk_score,
        dili_likelihood_pct=dili_likelihood_pct,
        risk_class=risk_class,
        mitigation_strategies=mitigation_strategies,
        notes=notes,
    )


def screen_compounds(compound_list: list) -> list:
    """Screen multiple compounds and return results sorted by risk_score descending.

    Parameters
    ----------
    compound_list:
        List of dicts with keys: drug_name, logp, mw, daily_dose_mg,
        fu_plasma, has_reactive_metabolite, has_mitochondrial_liability,
        structural_alert_count.

    Returns
    -------
    list of HepatotoxicityRiskResult
        Sorted by risk_score descending (highest risk first).
    """
    results = []
    for compound in compound_list:
        result = predict_hepatotoxicity_risk(
            drug_name=compound["drug_name"],
            logp=compound["logp"],
            mw=compound["mw"],
            daily_dose_mg=compound["daily_dose_mg"],
            fu_plasma=compound["fu_plasma"],
            has_reactive_metabolite=compound["has_reactive_metabolite"],
            has_mitochondrial_liability=compound["has_mitochondrial_liability"],
            structural_alert_count=compound["structural_alert_count"],
        )
        results.append(result)

    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
