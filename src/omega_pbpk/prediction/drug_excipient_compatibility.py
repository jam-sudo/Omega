"""Drug-Excipient Compatibility Predictor — Phase 817.

Provides:
- ExcipientCompatibilityResult dataclass
- predict_compatibility() using functional-group / physicochemical risk rules
- screen_excipients() for ranked screening of multiple excipients
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Excipient database
# ---------------------------------------------------------------------------

# name → {reducing_sugar, ph_effect, description}
# ph_effect: "neutral", "acidic", "alkaline"
_EXCIPIENT_DB: dict[str, dict] = {
    "Lactose": {
        "reducing_sugar": True,
        "ph_effect": "neutral",
        "description": "Common filler/diluent; reducing sugar; Maillard risk with amines",
    },
    "MCC": {
        "reducing_sugar": False,
        "ph_effect": "neutral",
        "description": "Microcrystalline cellulose; inert filler/binder",
    },
    "HPMC": {
        "reducing_sugar": False,
        "ph_effect": "neutral",
        "description": "Hydroxypropyl methylcellulose; binder/film coat",
    },
    "PVP": {
        "reducing_sugar": False,
        "ph_effect": "neutral",
        "description": "Polyvinylpyrrolidone; binder",
    },
    "Starch": {
        "reducing_sugar": True,
        "ph_effect": "neutral",
        "description": "Native starch; disintegrant/filler; mild reducing character",
    },
    "Magnesium Stearate": {
        "reducing_sugar": False,
        "ph_effect": "alkaline",
        "description": "Lubricant; alkaline microenvironment; may accelerate hydrolysis",
    },
    "Citric Acid": {
        "reducing_sugar": False,
        "ph_effect": "acidic",
        "description": "Acidulant/buffer; acidic microenvironment",
    },
    "Sodium Bicarbonate": {
        "reducing_sugar": False,
        "ph_effect": "alkaline",
        "description": "Alkalizing agent; CO2 effervescent; alkaline microenvironment",
    },
    "Talc": {
        "reducing_sugar": False,
        "ph_effect": "neutral",
        "description": "Glidant; essentially inert",
    },
    "Mannitol": {
        "reducing_sugar": False,
        "ph_effect": "neutral",
        "description": "Non-reducing sugar alcohol; inert filler for moisture-sensitive drugs",
    },
}

# Penalty weights
_PENALTY_MAILLARD = 30.0  # reducing sugar + basic nitrogen
_PENALTY_ACID_BASE_MAJOR = 20.0  # pH effect strongly opposes ionization
_PENALTY_ACID_BASE_MINOR = 10.0  # pH effect mildly opposes ionization
_PENALTY_WETTABILITY = 15.0  # high logP drug + hydrophilic excipient mismatch
_PENALTY_UNKNOWN = 30.0  # starting penalty for unknown excipients

_HIGH_LOGP_THRESHOLD = 3.0
_BASIC_PKA_THRESHOLD = 8.0
_ACIDIC_PKA_THRESHOLD = 4.0


def _classify_drug(pka: float) -> str:
    """Return 'basic', 'acidic', or 'neutral' based on pKa."""
    if pka > _BASIC_PKA_THRESHOLD:
        return "basic"
    if pka < _ACIDIC_PKA_THRESHOLD:
        return "acidic"
    return "neutral"


def _compute_penalties(
    logp: float,
    pka: float,
    excipient: dict,
) -> tuple[float, list[str]]:
    """Return (total_penalty, list_of_interaction_type_strings)."""
    penalties = 0.0
    interactions: list[str] = []

    drug_class = _classify_drug(pka)
    ph_effect = excipient["ph_effect"]
    reducing = excipient["reducing_sugar"]

    # 1. Maillard reaction risk
    if reducing and drug_class == "basic":
        penalties += _PENALTY_MAILLARD
        interactions.append("Maillard reaction")

    # 2. Acid-base / pH microenvironment conflicts
    if ph_effect == "acidic" and drug_class == "acidic":
        # Acid excipient lowers microenvironmental pH → drug less ionised (OK for stability
        # but can reduce dissolution); treat as minor concern
        penalties += _PENALTY_ACID_BASE_MINOR
        interactions.append("pH microenvironment (acidic-acidic)")
    elif ph_effect == "alkaline" and drug_class == "basic":
        penalties += _PENALTY_ACID_BASE_MINOR
        interactions.append("pH microenvironment (alkaline-basic)")
    elif ph_effect == "acidic" and drug_class == "basic":
        # Strong conflict: acidic excipient deprotonates basic drug
        penalties += _PENALTY_ACID_BASE_MAJOR
        interactions.append("acid-base interaction")
    elif ph_effect == "alkaline" and drug_class == "acidic":
        # Strong conflict: alkaline excipient deprotonates acidic drug
        penalties += _PENALTY_ACID_BASE_MAJOR
        interactions.append("acid-base interaction")

    # 3. Wettability / solubility mismatch
    if logp > _HIGH_LOGP_THRESHOLD:
        # Hydrophilic excipients (neutral, non-reducing) can worsen wettability
        if not reducing and ph_effect == "neutral":
            penalties += _PENALTY_WETTABILITY
            interactions.append("wettability mismatch")

    return penalties, interactions


def _score_to_risk(score: float) -> str:
    if score < 50.0:
        return "high"
    if score < 75.0:
        return "moderate"
    return "low"


def _stability_concern(interactions: list[str], score: float) -> str:
    if not interactions:
        return "none identified"
    if score < 50.0:
        return "significant stability concern: " + ", ".join(interactions)
    return "potential concern: " + ", ".join(interactions)


@dataclass(frozen=True)
class ExcipientCompatibilityResult:
    """Result of a drug-excipient compatibility prediction."""

    drug_name: str
    excipient_name: str
    compatibility_score: float  # 0-100 (100 = fully compatible)
    risk_level: str  # "low", "moderate", "high"
    interaction_types: str  # comma-separated list of predicted interactions
    recommended: bool
    stability_concern: str
    notes: str


def predict_compatibility(
    drug_name: str,
    logp: float,
    pka: float,
    mw_da: float,
    excipient_name: str,
) -> ExcipientCompatibilityResult:
    """Predict drug-excipient compatibility from physicochemical descriptors.

    Parameters
    ----------
    drug_name:
        Identifier for the drug (used in result only).
    logp:
        Octanol-water partition coefficient.
    pka:
        Most relevant pKa (use the ionisation pKa driving formulation behaviour).
    mw_da:
        Molecular weight in Daltons.
    excipient_name:
        Name of excipient (see _EXCIPIENT_DB for recognised entries).

    Returns
    -------
    ExcipientCompatibilityResult
    """
    if excipient_name not in _EXCIPIENT_DB:
        # Unknown excipient — return cautious default
        score = 70.0
        risk = "moderate"
        return ExcipientCompatibilityResult(
            drug_name=drug_name,
            excipient_name=excipient_name,
            compatibility_score=score,
            risk_level=risk,
            interaction_types="unknown",
            recommended=score >= 60.0,
            stability_concern="excipient not in database; experimental data required",
            notes=(
                f"Excipient '{excipient_name}' is not in the compatibility database. "
                "Default moderate compatibility assumed."
            ),
        )

    excipient = _EXCIPIENT_DB[excipient_name]
    penalties, interactions = _compute_penalties(logp, pka, excipient)

    score = max(0.0, min(100.0, 100.0 - penalties))
    risk = _score_to_risk(score)
    interaction_str = ", ".join(interactions) if interactions else "none"
    concern = _stability_concern(interactions, score)
    recommended = score >= 60.0

    note_parts: list[str] = [f"Drug class: {_classify_drug(pka)}; logP={logp:.2f}"]
    if excipient["reducing_sugar"]:
        note_parts.append(f"{excipient_name} is a reducing sugar")
    if excipient["ph_effect"] != "neutral":
        note_parts.append(f"pH effect: {excipient['ph_effect']}")
    notes = "; ".join(note_parts)

    return ExcipientCompatibilityResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        compatibility_score=score,
        risk_level=risk,
        interaction_types=interaction_str,
        recommended=recommended,
        stability_concern=concern,
        notes=notes,
    )


def screen_excipients(
    drug_name: str,
    logp: float,
    pka: float,
    mw_da: float,
    excipient_names: list[str],
) -> list[ExcipientCompatibilityResult]:
    """Screen a list of excipients, returning results sorted by compatibility_score descending.

    Parameters
    ----------
    drug_name, logp, pka, mw_da:
        Same as predict_compatibility.
    excipient_names:
        List of excipient names to evaluate.

    Returns
    -------
    List of ExcipientCompatibilityResult sorted descending by compatibility_score.
    """
    results = [predict_compatibility(drug_name, logp, pka, mw_da, exc) for exc in excipient_names]
    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
