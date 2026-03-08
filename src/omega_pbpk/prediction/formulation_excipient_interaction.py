"""
Phase 994 — Drug-excipient interaction prediction for oral solid dosage forms.

Rule-based compatibility scoring for 8 common pharmaceutical excipients.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_EXCIPIENTS = {
    "MCC",
    "lactose",
    "HPMC",
    "PVP",
    "magnesium_stearate",
    "croscarmellose",
    "starch",
    "talc",
}

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}

_VALID_INTERACTION_TYPES = {"none", "physical", "chemical", "electrostatic"}
_VALID_SEVERITIES = {"none", "minor", "moderate", "major"}
_VALID_RECOMMENDATIONS = {"compatible", "use with caution", "avoid"}


@dataclass(frozen=True)
class ExcipientInteractionResult:
    """Result of a drug-excipient compatibility prediction."""

    drug_name: str
    excipient_name: str
    drug_ionization_type: str
    interaction_type: str
    interaction_severity: str
    compatibility_score: float
    mechanism: str
    recommendation: str
    notes: str


def _score_to_recommendation(score: float) -> str:
    if score >= 80:
        return "compatible"
    if score >= 60:
        return "use with caution"
    return "avoid"


def predict_excipient_interaction(
    drug_name: str,
    excipient_name: str,
    drug_logp: float = 1.0,
    drug_pka: float = 7.0,
    drug_ionization_type: str = "neutral",
    has_primary_amine: bool = False,
    has_carboxylic_acid: bool = False,
) -> ExcipientInteractionResult:
    """Predict drug-excipient compatibility for oral solid dosage forms.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    excipient_name:
        Name of the excipient (must be one of the 8 supported excipients).
    drug_logp:
        Octanol-water partition coefficient of the drug.
    drug_pka:
        Ionization pKa of the drug.
    drug_ionization_type:
        "acid", "base", or "neutral".
    has_primary_amine:
        True if the drug contains a primary amine group.
    has_carboxylic_acid:
        True if the drug contains a carboxylic acid group.

    Returns
    -------
    ExcipientInteractionResult
    """
    if excipient_name not in _VALID_EXCIPIENTS:
        raise ValueError(
            f"excipient_name must be one of {sorted(_VALID_EXCIPIENTS)}, got {excipient_name!r}"
        )
    if drug_ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"drug_ionization_type must be one of {_VALID_IONIZATION_TYPES}, "
            f"got {drug_ionization_type!r}"
        )

    score: float
    interaction_type: str
    severity: str
    mechanism: str

    if excipient_name == "MCC":
        score = 100.0
        interaction_type = "none"
        severity = "none"
        mechanism = (
            "Microcrystalline cellulose is chemically inert and neutral. "
            "No significant interactions expected."
        )

    elif excipient_name == "lactose":
        if has_primary_amine:
            score = 20.0
            interaction_type = "chemical"
            severity = "major"
            mechanism = (
                "Maillard reaction between the primary amine group of the drug and "
                "the reducing sugar (lactose), causing discoloration and potency loss."
            )
        else:
            score = 90.0
            interaction_type = "none"
            severity = "none"
            mechanism = (
                "Lactose is generally compatible with drugs lacking primary amines. "
                "Minor risk of Maillard reaction at elevated temperature/humidity."
            )

    elif excipient_name == "HPMC":
        if drug_logp > 4.0:
            score = 80.0
            interaction_type = "physical"
            severity = "minor"
            mechanism = (
                "HPMC can form a viscous gel layer that may retard dissolution of "
                "highly lipophilic drugs (logP > 4), potentially reducing absorption rate."
            )
        else:
            score = 95.0
            interaction_type = "none"
            severity = "none"
            mechanism = (
                "HPMC is chemically compatible and provides good film-forming "
                "properties. No significant interaction expected."
            )

    elif excipient_name == "PVP":
        if has_carboxylic_acid or drug_ionization_type == "acid":
            score = 65.0
            interaction_type = "electrostatic"
            severity = "moderate"
            mechanism = (
                "PVP can form ion-pair complexes with acidic drugs via electrostatic "
                "interactions, potentially altering dissolution and bioavailability."
            )
        else:
            score = 90.0
            interaction_type = "none"
            severity = "none"
            mechanism = (
                "PVP acts as a binder and solubilizer. Compatible with most non-acidic drugs."
            )

    elif excipient_name == "magnesium_stearate":
        if drug_ionization_type == "acid" or has_carboxylic_acid:
            score = 50.0
            interaction_type = "chemical"
            severity = "moderate"
            mechanism = (
                "Magnesium stearate is basic (pH ~8–9 in suspension) and can react "
                "with acidic drugs, forming insoluble salts that reduce dissolution "
                "and bioavailability. Use at minimum effective concentration."
            )
        else:
            score = 85.0
            interaction_type = "none"
            severity = "none"
            mechanism = (
                "Magnesium stearate is a standard lubricant. Compatible with "
                "non-acidic drugs at typical concentrations (≤1%)."
            )

    elif excipient_name == "croscarmellose":
        if drug_ionization_type == "base" and drug_pka > 8.0:
            score = 70.0
            interaction_type = "physical"
            severity = "minor"
            mechanism = (
                "Croscarmellose sodium is slightly acidic (carboxylate groups). "
                "Strongly basic drugs (pKa > 8) may experience reduced local pH, "
                "potentially affecting solubility and dissolution."
            )
        else:
            score = 88.0
            interaction_type = "none"
            severity = "none"
            mechanism = (
                "Croscarmellose sodium is an effective superdisintegrant with "
                "good compatibility across most drug classes."
            )

    elif excipient_name == "starch":
        score = 95.0
        interaction_type = "none"
        severity = "none"
        mechanism = (
            "Native starch is a neutral polysaccharide with excellent "
            "compatibility across drug classes."
        )

    else:  # talc
        score = 98.0
        interaction_type = "none"
        severity = "none"
        mechanism = (
            "Talc is a chemically inert mineral lubricant/glidant. "
            "No significant drug interactions expected."
        )

    recommendation = _score_to_recommendation(score)

    notes = (
        f"Drug: {drug_name}, logP={drug_logp:.2f}, pKa={drug_pka:.2f}, "
        f"ionization={drug_ionization_type}, "
        f"primary_amine={has_primary_amine}, carboxylic_acid={has_carboxylic_acid}."
    )

    return ExcipientInteractionResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        drug_ionization_type=drug_ionization_type,
        interaction_type=interaction_type,
        interaction_severity=severity,
        compatibility_score=score,
        mechanism=mechanism,
        recommendation=recommendation,
        notes=notes,
    )


def screen_excipient_compatibility(
    drug_name: str,
    excipients: list,
    **drug_kwargs,
) -> list:
    """Screen compatibility of a drug against multiple excipients.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    excipients:
        List of excipient names to screen.
    **drug_kwargs:
        Keyword arguments passed to ``predict_excipient_interaction``
        (drug_logp, drug_pka, drug_ionization_type, has_primary_amine,
        has_carboxylic_acid).

    Returns
    -------
    list[ExcipientInteractionResult]
        Results sorted by compatibility_score descending.
    """
    results = [predict_excipient_interaction(drug_name, exc, **drug_kwargs) for exc in excipients]
    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
