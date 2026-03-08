"""Phase 663 - Drug Ocular Toxicity Prediction.

Predict ocular toxicity risk from drug properties for safety assessment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OcularToxicityResult:
    drug_name: str
    logP: float
    mw_Da: float
    charge_at_ph7: float
    melanin_binding_score: float
    corneal_deposition_risk: float
    retinal_toxicity_score: float
    lens_accumulation_risk: float
    overall_ocular_risk_score: float
    risk_class: str
    primary_toxicity_mechanism: str
    mitigation_strategy: str
    notes: str


def predict_ocular_toxicity(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pka: float = 7.4,
    is_base: bool = True,
    charge_at_ph7: float = 0.0,
    amphiphilic: bool = False,
) -> OcularToxicityResult:
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # Melanin binding
    if is_base and pka > 7.4:
        melanin_binding = min(10.0, (pka - 7.4) * 2.0 + logP * 0.5)
    else:
        melanin_binding = max(0.0, logP * 0.3)
    melanin_binding_score = max(0.0, min(10.0, melanin_binding))

    # Corneal deposition risk
    if mw_Da > 500 and logP < 0:
        corneal_dep = 8.0
    elif logP > 3 and mw_Da > 400:
        corneal_dep = 6.0
    else:
        corneal_dep = max(0.0, min(5.0, mw_Da / 200.0 - logP * 0.5))
    corneal_deposition_risk = max(0.0, min(10.0, corneal_dep))

    # Retinal toxicity
    retinal = melanin_binding_score * 0.5 + (logP - 2.0) * 1.0 + amphiphilic * 3.0
    retinal_toxicity_score = max(0.0, min(10.0, retinal))

    # Lens accumulation
    lens = max(0.0, min(10.0, logP * 1.5 + mw_Da / 400.0 - 1.0))
    lens_accumulation_risk = max(0.0, min(10.0, lens))

    # Overall risk
    overall = (
        melanin_binding_score * 0.3
        + corneal_deposition_risk * 0.2
        + retinal_toxicity_score * 0.4
        + lens_accumulation_risk * 0.1
    )
    overall_ocular_risk_score = max(0.0, min(10.0, overall))

    # Risk class
    if overall_ocular_risk_score > 6:
        risk_class = "high"
    elif overall_ocular_risk_score > 3:
        risk_class = "moderate"
    else:
        risk_class = "low"

    # Primary toxicity mechanism
    scores = {
        "melanin_binding": melanin_binding_score,
        "corneal_deposition": corneal_deposition_risk,
        "retinal": retinal_toxicity_score,
        "lens": lens_accumulation_risk,
    }
    primary_toxicity_mechanism = max(scores, key=scores.get)

    # Mitigation strategy
    if risk_class == "high":
        mitigation_strategy = "avoid"
    elif risk_class == "moderate":
        mitigation_strategy = "monitoring_required"
    else:
        mitigation_strategy = "standard_surveillance"

    notes = (
        f"Ocular toxicity assessment for {drug_name}: "
        f"logP={logP}, MW={mw_Da} Da, pKa={pka}, "
        f"is_base={is_base}, amphiphilic={amphiphilic}."
    )

    return OcularToxicityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        charge_at_ph7=charge_at_ph7,
        melanin_binding_score=melanin_binding_score,
        corneal_deposition_risk=corneal_deposition_risk,
        retinal_toxicity_score=retinal_toxicity_score,
        lens_accumulation_risk=lens_accumulation_risk,
        overall_ocular_risk_score=overall_ocular_risk_score,
        risk_class=risk_class,
        primary_toxicity_mechanism=primary_toxicity_mechanism,
        mitigation_strategy=mitigation_strategy,
        notes=notes,
    )


def screen_ocular_toxicity(compounds: list) -> list:
    results = [predict_ocular_toxicity(**c) for c in compounds]
    results.sort(key=lambda r: r.overall_ocular_risk_score, reverse=True)
    return results
