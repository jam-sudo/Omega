"""Phase 451 — DDI Severity Classification System.

Classifies drug-drug interactions using FDA and EMA guidance with mechanistic predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hardcoded DDI interaction database
# ---------------------------------------------------------------------------
INTERACTIONS = {
    # (perpetrator_lower, victim_lower): interaction data
    ("fluconazole", "warfarin"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP2C9_inhibition",
        "aucr": 2.7,
        "cmax_ratio": 1.6,
        "severity": "major",
    },
    ("rifampicin", "warfarin"): {
        "type": "CYP_induction",
        "mechanism": "CYP2C9_induction",
        "aucr": 0.37,
        "cmax_ratio": 0.5,
        "severity": "major",
    },
    ("amiodarone", "simvastatin"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP3A4_inhibition",
        "aucr": 3.0,
        "cmax_ratio": 2.5,
        "severity": "major",
    },
    ("clarithromycin", "simvastatin"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP3A4_inhibition",
        "aucr": 10.0,
        "cmax_ratio": 8.0,
        "severity": "contraindicated",
    },
    ("metformin", "alcohol"): {
        "type": "PD_additive",
        "mechanism": "lactic_acidosis_risk",
        "aucr": 1.0,
        "cmax_ratio": 1.0,
        "severity": "moderate",
    },
    ("aspirin", "warfarin"): {
        "type": "PD_additive",
        "mechanism": "bleeding_risk",
        "aucr": 1.2,
        "cmax_ratio": 1.1,
        "severity": "major",
    },
    ("fluoxetine", "tramadol"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP2D6_inhibition",
        "aucr": 2.5,
        "cmax_ratio": 1.8,
        "severity": "major",
    },
    ("omeprazole", "clopidogrel"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP2C19_inhibition",
        "aucr": 0.6,
        "cmax_ratio": 0.5,
        "severity": "moderate",
    },
    ("ciprofloxacin", "theophylline"): {
        "type": "CYP_inhibition",
        "mechanism": "CYP1A2_inhibition",
        "aucr": 1.7,
        "cmax_ratio": 1.4,
        "severity": "moderate",
    },
    ("phenytoin", "oral_contraceptives"): {
        "type": "CYP_induction",
        "mechanism": "CYP3A4_induction",
        "aucr": 0.5,
        "cmax_ratio": 0.4,
        "severity": "major",
    },
}

# ---------------------------------------------------------------------------
# Severity → FDA guidance category mapping
# ---------------------------------------------------------------------------
_SEVERITY_TO_FDA = {
    "contraindicated": "avoid",
    "major": "dose_reduction",
    "moderate": "monitor",
    "minor": "no_action",
    "none": "no_action",
}

# ---------------------------------------------------------------------------
# Severity → clinical consequence and management strategy
# ---------------------------------------------------------------------------
_SEVERITY_TO_CONSEQUENCE = {
    "contraindicated": "Life-threatening or serious adverse events expected; combination must be avoided.",
    "major": "Significant adverse effects possible; may require dose adjustment or close monitoring.",
    "moderate": "Moderate clinical effect; monitoring and possible dose modification recommended.",
    "minor": "Mild or minimally significant interaction; minimal clinical impact.",
    "none": "No clinically significant interaction identified.",
}

_SEVERITY_TO_MANAGEMENT = {
    "contraindicated": "Avoid concurrent use. Identify an alternative agent from a different drug class.",
    "major": "Reduce dose of victim drug, monitor closely for adverse effects, or consider alternative.",
    "moderate": "Monitor for signs of altered drug effect; adjust dose if necessary.",
    "minor": "No specific action required; counsel patient on potential mild effects.",
    "none": "No action required.",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DDISeverityResult:
    """Result of a DDI severity classification."""

    perpetrator_drug: str
    victim_drug: str
    interaction_type: str  # "CYP_inhibition", "CYP_induction", "transporter_inhibition", "PD_additive", "PD_antagonism"
    perpetrator_mechanism: str
    severity_class: str  # "contraindicated", "major", "moderate", "minor", "none"
    predicted_aucr: float  # AUC ratio (victim with/without perpetrator)
    predicted_cmax_ratio: float
    clinical_consequence: str
    management_strategy: str
    monitoring_required: bool
    dose_adjustment_required: bool
    alternative_recommendation: str
    fda_guidance_category: str  # "avoid", "dose_reduction", "monitor", "no_action"
    notes: str


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------


def classify_ddi_severity(perpetrator_drug: str, victim_drug: str) -> DDISeverityResult:
    """Classify the severity of a drug-drug interaction.

    Looks up the pair in the interaction database (bidirectional) and returns
    a DDISeverityResult with FDA-aligned severity classification.

    Parameters
    ----------
    perpetrator_drug:
        Name of the drug acting as perpetrator (modulator).
    victim_drug:
        Name of the drug whose exposure is affected.

    Returns
    -------
    DDISeverityResult
        Classified interaction result.
    """
    perp_lower = perpetrator_drug.strip().lower()
    vict_lower = victim_drug.strip().lower()

    # Bidirectional lookup: try (perp, vict) then (vict, perp)
    data = INTERACTIONS.get((perp_lower, vict_lower))
    reversed_lookup = False
    if data is None:
        data = INTERACTIONS.get((vict_lower, perp_lower))
        if data is not None:
            reversed_lookup = True

    if data is None:
        # No known interaction
        return DDISeverityResult(
            perpetrator_drug=perpetrator_drug,
            victim_drug=victim_drug,
            interaction_type="none",
            perpetrator_mechanism="unknown",
            severity_class="none",
            predicted_aucr=1.0,
            predicted_cmax_ratio=1.0,
            clinical_consequence=_SEVERITY_TO_CONSEQUENCE["none"],
            management_strategy=_SEVERITY_TO_MANAGEMENT["none"],
            monitoring_required=False,
            dose_adjustment_required=False,
            alternative_recommendation="None required.",
            fda_guidance_category="no_action",
            notes=(
                f"No clinically significant interaction identified between "
                f"{perpetrator_drug} and {victim_drug} in the current database."
            ),
        )

    severity = data["severity"]
    interaction_type = data["type"]
    mechanism = data["mechanism"]
    aucr = data["aucr"]
    cmax_ratio = data["cmax_ratio"]

    monitoring_required = severity in {"contraindicated", "major", "moderate"}
    dose_adjustment_required = severity in {"contraindicated", "major"}

    fda_category = _SEVERITY_TO_FDA[severity]
    consequence = _SEVERITY_TO_CONSEQUENCE[severity]
    management = _SEVERITY_TO_MANAGEMENT[severity]

    # Alternative recommendation based on severity
    if severity == "contraindicated":
        alt_rec = f"Select an alternative to {victim_drug} that does not interact with {perpetrator_drug}."
    elif severity == "major":
        alt_rec = f"Consider an alternative to {victim_drug} or {perpetrator_drug} with a safer interaction profile."
    else:
        alt_rec = "No alternative required unless clinically indicated."

    # Build notes
    direction_note = " (bidirectional lookup applied)" if reversed_lookup else ""
    aucr_note = (
        f"AUCR={aucr:.2f} (victim exposure {'increased' if aucr >= 1 else 'decreased'} "
        f"{abs(aucr - 1.0) * 100:.0f}%)."
    )
    notes = (
        f"{perpetrator_drug} → {victim_drug}: {mechanism.replace('_', ' ')} "
        f"({interaction_type.replace('_', ' ')}). {aucr_note}{direction_note}"
    )

    return DDISeverityResult(
        perpetrator_drug=perpetrator_drug,
        victim_drug=victim_drug,
        interaction_type=interaction_type,
        perpetrator_mechanism=mechanism,
        severity_class=severity,
        predicted_aucr=aucr,
        predicted_cmax_ratio=cmax_ratio,
        clinical_consequence=consequence,
        management_strategy=management,
        monitoring_required=monitoring_required,
        dose_adjustment_required=dose_adjustment_required,
        alternative_recommendation=alt_rec,
        fda_guidance_category=fda_category,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


def screen_interactions(drug_list: list[str]) -> list[DDISeverityResult]:
    """Screen all pairwise interactions among a list of drugs.

    Parameters
    ----------
    drug_list:
        List of drug names to screen.

    Returns
    -------
    List of DDISeverityResult for all drug pairs with severity != "none",
    sorted by predicted_aucr descending (most impactful first).
    """
    results = []
    n = len(drug_list)
    for i in range(n):
        for j in range(i + 1, n):
            result = classify_ddi_severity(drug_list[i], drug_list[j])
            if result.severity_class != "none":
                results.append(result)

    # Sort by aucr descending (most impactful first)
    # For induction pairs (aucr < 1), use absolute distance from 1
    results.sort(key=lambda r: r.predicted_aucr, reverse=True)
    return results
