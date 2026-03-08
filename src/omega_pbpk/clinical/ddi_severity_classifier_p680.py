"""Phase 680 — Drug-Drug Interaction Severity Classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDISeverityResult:
    """Result of DDI severity classification."""

    drug_a: str
    drug_b: str
    interaction_type: str
    mechanism: str
    severity_score: float
    severity_class: str
    contraindicated: bool
    clinical_recommendation: str
    aucr_estimate: float
    cmaxr_estimate: float
    onset_h: float
    duration_h: float
    management_strategy: str
    notes: str


_VALID_INTERACTION_TYPES = {
    "CYP3A4_inhibition",
    "CYP2D6_inhibition",
    "CYP1A2_inhibition",
    "P_glycoprotein",
    "QT_prolongation",
    "additive_toxicity",
    "pharmacodynamic_synergy",
    "protein_binding_displacement",
}

_TYPE_WEIGHTS = {
    "CYP3A4_inhibition": 8,
    "CYP2D6_inhibition": 7,
    "CYP1A2_inhibition": 6,
    "P_glycoprotein": 5,
    "QT_prolongation": 15,
    "additive_toxicity": 12,
    "pharmacodynamic_synergy": 8,
    "protein_binding_displacement": 4,
}

_MECHANISM_DESC = {
    "CYP3A4_inhibition": "Inhibition of CYP3A4-mediated metabolism increasing substrate exposure",
    "CYP2D6_inhibition": "Inhibition of CYP2D6-mediated metabolism increasing substrate exposure",
    "CYP1A2_inhibition": "Inhibition of CYP1A2-mediated metabolism increasing substrate exposure",
    "P_glycoprotein": "P-glycoprotein transporter inhibition altering drug efflux",
    "QT_prolongation": "Additive QT interval prolongation increasing cardiac arrhythmia risk",
    "additive_toxicity": "Additive toxicity from combined pharmacological effects",
    "pharmacodynamic_synergy": "Synergistic pharmacodynamic effects amplifying drug response",
    "protein_binding_displacement": "Displacement from plasma protein binding sites increasing free drug",
}

_ONSET_DEFAULTS = {
    "CYP3A4_inhibition": 1.0,
    "CYP2D6_inhibition": 2.0,
    "CYP1A2_inhibition": 2.0,
    "P_glycoprotein": 2.0,
    "QT_prolongation": 0.5,
    "additive_toxicity": 2.0,
    "pharmacodynamic_synergy": 2.0,
    "protein_binding_displacement": 0.1,
}

_CYP_TYPES = {"CYP3A4_inhibition", "CYP2D6_inhibition", "CYP1A2_inhibition"}


def classify_ddi_severity(
    drug_a: str,
    drug_b: str,
    interaction_type: str,
    aucr_estimate: float = 1.0,
    cmaxr_estimate: float = 1.0,
    onset_h: float | None = None,
    duration_h: float | None = None,
) -> DDISeverityResult:
    """Classify severity of a drug-drug interaction."""
    if aucr_estimate <= 0:
        raise ValueError("aucr_estimate must be > 0")
    if cmaxr_estimate <= 0:
        raise ValueError("cmaxr_estimate must be > 0")
    if interaction_type not in _VALID_INTERACTION_TYPES:
        raise ValueError(f"interaction_type must be one of {sorted(_VALID_INTERACTION_TYPES)}")

    # Score components
    aucr_component = max(0.0, min(40.0, abs(aucr_estimate - 1.0) * 20.0))
    cmax_component = max(0.0, min(20.0, abs(cmaxr_estimate - 1.0) * 10.0))
    type_weight = _TYPE_WEIGHTS[interaction_type]
    severity_score = min(100.0, aucr_component + cmax_component + type_weight * 2)

    # Classification
    if severity_score >= 70:
        severity_class = "contraindicated"
    elif severity_score >= 40:
        severity_class = "major"
    elif severity_score >= 20:
        severity_class = "moderate"
    else:
        severity_class = "minor"

    contraindicated = severity_class == "contraindicated" or (
        interaction_type == "QT_prolongation" and severity_score >= 50
    )

    if contraindicated:
        clinical_recommendation = "avoid combination"
    elif severity_class == "major":
        clinical_recommendation = "use alternative or monitor closely"
    elif severity_class == "moderate":
        clinical_recommendation = "adjust dose, monitor"
    else:
        clinical_recommendation = "standard monitoring"

    # Onset / duration defaults
    if onset_h is None:
        onset_h = _ONSET_DEFAULTS.get(interaction_type, 2.0)
    if duration_h is None:
        duration_h = 12.0 if interaction_type in _CYP_TYPES else 2.0

    mechanism = _MECHANISM_DESC[interaction_type]

    management_strategy = (
        "dose reduction 50% if cannot avoid"
        if severity_class in ("contraindicated", "major")
        else "monitor clinical response and adjust as needed"
    )

    return DDISeverityResult(
        drug_a=drug_a,
        drug_b=drug_b,
        interaction_type=interaction_type,
        mechanism=mechanism,
        severity_score=severity_score,
        severity_class=severity_class,
        contraindicated=contraindicated,
        clinical_recommendation=clinical_recommendation,
        aucr_estimate=aucr_estimate,
        cmaxr_estimate=cmaxr_estimate,
        onset_h=onset_h,
        duration_h=duration_h,
        management_strategy=management_strategy,
        notes=f"DDI: {drug_a} + {drug_b} via {interaction_type}",
    )


def batch_classify_ddi(
    interactions: list,
) -> list:
    """Classify multiple DDIs and return sorted by severity_score descending."""
    results = []
    for item in interactions:
        result = classify_ddi_severity(
            drug_a=item["drug_a"],
            drug_b=item["drug_b"],
            interaction_type=item["interaction_type"],
            aucr_estimate=item.get("aucr_estimate", 1.0),
            cmaxr_estimate=item.get("cmaxr_estimate", 1.0),
            onset_h=item.get("onset_h"),
            duration_h=item.get("duration_h"),
        )
        results.append(result)
    results.sort(key=lambda r: r.severity_score, reverse=True)
    return results
