"""Phase 249: Excipient-drug interaction predictor.

Predicts physicochemical interactions between drugs and pharmaceutical
excipients based on functional group analysis and excipient properties.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Excipient lookup table
# ---------------------------------------------------------------------------

_EXCIPIENTS = {
    "lactose": {
        "type": "sugar",
        "reducing_sugar": True,
        "amine_reactivity": 0.8,
        "acid_base_reactive": False,
        "adsorption_factor": 10,
    },
    "MCC": {
        "type": "cellulose",
        "reducing_sugar": False,
        "amine_reactivity": 0.1,
        "acid_base_reactive": False,
        "adsorption_factor": 20,
    },
    "starch": {
        "type": "polysaccharide",
        "reducing_sugar": True,
        "amine_reactivity": 0.3,
        "acid_base_reactive": False,
        "adsorption_factor": 10,
    },
    "magnesium_stearate": {
        "type": "metal_salt",
        "reducing_sugar": False,
        "amine_reactivity": 0.0,
        "acid_base_reactive": True,
        "adsorption_factor": 10,
    },
    "HPMC": {
        "type": "cellulose_ether",
        "reducing_sugar": False,
        "amine_reactivity": 0.0,
        "acid_base_reactive": False,
        "adsorption_factor": 10,
    },
    "PVP": {
        "type": "polymer",
        "reducing_sugar": False,
        "amine_reactivity": 0.05,
        "acid_base_reactive": False,
        "adsorption_factor": 10,
    },
    "silicon_dioxide": {
        "type": "inorganic",
        "reducing_sugar": False,
        "amine_reactivity": 0.0,
        "acid_base_reactive": True,
        "adsorption_factor": 60,  # charged drugs adsorbed
    },
}

_ACID_BASE_RISK_NUMERIC = {
    "none": 0,
    "low": 20,
    "moderate": 50,
    "high": 80,
}

_VALID_CHARGES = {"positive", "negative", "neutral", "zwitterion"}
_VALID_ACID_BASE = {"none", "low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExcipientInteractionResult:
    """Result of drug-excipient interaction prediction."""

    drug_name: str
    excipient: str
    drug_functional_groups: tuple[str, ...]

    maillard_risk_score: float        # 0-100
    acid_base_incompatibility_risk: str  # "none", "low", "moderate", "high"
    adsorption_risk_score: float      # 0-100
    overall_compatibility_score: float  # 0-100
    compatibility_class: str          # "compatible", "marginal", "incompatible"
    recommended_alternative: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def _compute_maillard_risk(drug_functional_groups: tuple[str, ...], excipient_props: dict) -> float:
    """Maillard reaction risk for amine drugs + reducing sugars."""
    if "amine" in drug_functional_groups and excipient_props["reducing_sugar"]:
        return excipient_props["amine_reactivity"] * 100.0
    return 0.0


def _compute_acid_base_risk(
    drug_functional_groups: tuple[str, ...],
    excipient_props: dict,
    drug_charge: str,
) -> str:
    """Acid-base incompatibility risk level."""
    if "carboxylic_acid" in drug_functional_groups and excipient_props["type"] == "metal_salt":
        return "high"
    if "amine" in drug_functional_groups and excipient_props["acid_base_reactive"]:
        return "moderate"
    return "none"


def _compute_adsorption_risk(excipient_props: dict, drug_charge: str) -> float:
    """Adsorption risk score."""
    if excipient_props["type"] == "inorganic":
        # Silicon dioxide: high adsorption for charged drugs
        if drug_charge in ("positive", "negative", "zwitterion"):
            return 60.0
        return 30.0
    return float(excipient_props["adsorption_factor"])


def _compatibility_class(score: float) -> str:
    if score >= 70.0:
        return "compatible"
    if score >= 50.0:
        return "marginal"
    return "incompatible"


def _recommend_alternative(excipient: str) -> str:
    """Suggest an alternative excipient when incompatible."""
    if excipient == "MCC":
        return "HPMC"
    return "MCC"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_excipient_interaction(
    drug_name: str,
    drug_functional_groups: list,
    excipient: str,
    drug_charge: str = "neutral",
) -> ExcipientInteractionResult:
    """Predict drug-excipient physicochemical interaction.

    Parameters
    ----------
    drug_name:
        Identifier for the drug molecule.
    drug_functional_groups:
        List of functional group strings, e.g. ["amine", "hydroxyl"].
    excipient:
        Excipient name; must be one of the seven supported excipients.
    drug_charge:
        Charge state of the drug at physiological pH. One of
        {"positive", "negative", "neutral", "zwitterion"}.

    Returns
    -------
    ExcipientInteractionResult
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if excipient not in _EXCIPIENTS:
        raise ValueError(
            f"Unknown excipient '{excipient}'. Valid options: {sorted(_EXCIPIENTS)}"
        )
    if drug_charge not in _VALID_CHARGES:
        raise ValueError(
            f"drug_charge must be one of {_VALID_CHARGES}, got '{drug_charge}'"
        )
    if not isinstance(drug_functional_groups, (list, tuple)):
        raise ValueError("drug_functional_groups must be a list of strings")

    fg_tuple: tuple[str, ...] = tuple(drug_functional_groups)
    props = _EXCIPIENTS[excipient]

    maillard_risk = _compute_maillard_risk(fg_tuple, props)
    ab_risk_str = _compute_acid_base_risk(fg_tuple, props, drug_charge)
    adsorption_risk = _compute_adsorption_risk(props, drug_charge)

    ab_risk_numeric = float(_ACID_BASE_RISK_NUMERIC[ab_risk_str])
    max_risk = max(maillard_risk, adsorption_risk, ab_risk_numeric)
    overall = max(0.0, min(100.0, 100.0 - max_risk))

    compat_class = _compatibility_class(overall)
    alternative = _recommend_alternative(excipient) if compat_class == "incompatible" else ""

    # Build notes
    note_parts = []
    if maillard_risk > 0:
        note_parts.append(f"Maillard risk {maillard_risk:.0f}/100 (amine + reducing sugar)")
    if ab_risk_str != "none":
        note_parts.append(f"Acid-base incompatibility: {ab_risk_str}")
    if adsorption_risk >= 60:
        note_parts.append(f"High adsorption risk ({adsorption_risk:.0f}/100) for charged drug")
    if not note_parts:
        note_parts.append("No major incompatibilities predicted")
    notes = "; ".join(note_parts)

    return ExcipientInteractionResult(
        drug_name=drug_name,
        excipient=excipient,
        drug_functional_groups=fg_tuple,
        maillard_risk_score=maillard_risk,
        acid_base_incompatibility_risk=ab_risk_str,
        adsorption_risk_score=adsorption_risk,
        overall_compatibility_score=overall,
        compatibility_class=compat_class,
        recommended_alternative=alternative,
        notes=notes,
    )


def screen_excipient_compatibility(
    drug_name: str,
    drug_functional_groups: list,
    drug_charge: str = "neutral",
) -> list:
    """Screen all 7 excipients and return results sorted by compatibility score.

    Parameters
    ----------
    drug_name:
        Identifier for the drug molecule.
    drug_functional_groups:
        List of functional group strings.
    drug_charge:
        Charge state. One of {"positive", "negative", "neutral", "zwitterion"}.

    Returns
    -------
    list[ExcipientInteractionResult] sorted by overall_compatibility_score descending.
    """
    results = [
        predict_excipient_interaction(
            drug_name=drug_name,
            drug_functional_groups=drug_functional_groups,
            excipient=exc,
            drug_charge=drug_charge,
        )
        for exc in _EXCIPIENTS
    ]
    results.sort(key=lambda r: r.overall_compatibility_score, reverse=True)
    return results
