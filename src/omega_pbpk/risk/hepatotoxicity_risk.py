"""Phase 582 - Drug Hepatotoxicity Risk Score.

Predicts hepatotoxicity risk from structural and PK features.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HepatotoxicityRiskResult:
    drug_name: str
    logP: float
    mw_Da: float
    daily_dose_mg: float
    reactive_metabolite_alert: bool
    mitochondrial_alert: bool
    bile_salt_export_pump_alert: bool
    total_body_burden_mg: float
    risk_score: float
    dili_class: str
    severity_prediction: str
    monitoring_recommendation: str
    notes: str


def _validate_inputs(daily_dose_mg: float, mw_Da: float) -> None:
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")


def predict_hepatotoxicity_risk(
    drug_name: str,
    logP: float,
    mw_Da: float,
    daily_dose_mg: float,
    reactive_metabolite_alert: bool = False,
    mitochondrial_alert: bool = False,
    bsep_alert: bool = False,
) -> HepatotoxicityRiskResult:
    _validate_inputs(daily_dose_mg, mw_Da)

    bsep_alert_final = bsep_alert or (mw_Da > 400 and logP > 2)

    # Score components
    dose_pts = min(30.0, daily_dose_mg / 10.0 * 3.0)
    lipo_pts = min(20.0, max(0.0, logP - 1.0) * 5.0)
    burden_pts = min(20.0, daily_dose_mg * max(1.0, logP) / 100.0)
    rm_pts = 15.0 if reactive_metabolite_alert else 0.0
    mito_pts = 10.0 if mitochondrial_alert else 0.0
    bsep_pts = 10.0 if bsep_alert_final else 0.0

    raw_score = dose_pts + lipo_pts + burden_pts + rm_pts + mito_pts + bsep_pts
    risk_score = max(0.0, min(100.0, raw_score))

    # Classifications
    if risk_score > 60:
        dili_class = "most_concern"
    elif risk_score > 30:
        dili_class = "less_concern"
    else:
        dili_class = "no_concern"

    if risk_score > 75:
        severity_prediction = "fatal"
    elif risk_score > 55:
        severity_prediction = "severe"
    elif risk_score > 35:
        severity_prediction = "moderate"
    else:
        severity_prediction = "mild"

    if dili_class == "most_concern":
        monitoring = "monthly LFT monitoring for first year"
    elif dili_class == "less_concern":
        monitoring = "quarterly LFT monitoring"
    else:
        monitoring = "routine annual monitoring"

    total_body_burden = daily_dose_mg * max(1.0, logP)

    alerts = []
    if reactive_metabolite_alert:
        alerts.append("reactive_metabolite")
    if mitochondrial_alert:
        alerts.append("mitochondrial")
    if bsep_alert_final:
        alerts.append("BSEP_inhibition")

    notes = (
        f"Risk score {risk_score:.1f}/100 ({dili_class}); "
        f"alerts: {', '.join(alerts) if alerts else 'none'}"
    )

    return HepatotoxicityRiskResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        daily_dose_mg=daily_dose_mg,
        reactive_metabolite_alert=reactive_metabolite_alert,
        mitochondrial_alert=mitochondrial_alert,
        bile_salt_export_pump_alert=bsep_alert_final,
        total_body_burden_mg=total_body_burden,
        risk_score=risk_score,
        dili_class=dili_class,
        severity_prediction=severity_prediction,
        monitoring_recommendation=monitoring,
        notes=notes,
    )


def screen_hepatotoxicity(compounds: list) -> list:
    results = []
    for compound in compounds:
        result = predict_hepatotoxicity_risk(**compound)
        results.append(result)
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
