"""Phase 291 — Drug-Excipient Compatibility Predictor.

Predicts physicochemical compatibility between a drug and common
pharmaceutical excipients based on acid-base interactions, moisture
sensitivity, and redox reactivity.
"""

from dataclasses import dataclass

_EXCIPIENTS: dict = {
    "MCC": {"type": "neutral_filler", "ph_effect": 0.0, "moisture_risk": 0.1, "redox_risk": 0.0},
    "lactose": {
        "type": "reducing_sugar",
        "ph_effect": 0.0,
        "moisture_risk": 0.3,
        "redox_risk": 0.4,
    },
    "starch": {"type": "neutral_filler", "ph_effect": 0.0, "moisture_risk": 0.4, "redox_risk": 0.1},
    "povidone": {"type": "binder", "ph_effect": 0.0, "moisture_risk": 0.5, "redox_risk": 0.1},
    "HPMC": {"type": "polymer", "ph_effect": 0.0, "moisture_risk": 0.3, "redox_risk": 0.0},
    "citric_acid": {
        "type": "acidic_excipient",
        "ph_effect": -2.0,
        "moisture_risk": 0.2,
        "redox_risk": 0.1,
    },
    "Na_bicarbonate": {
        "type": "basic_excipient",
        "ph_effect": 2.0,
        "moisture_risk": 0.2,
        "redox_risk": 0.2,
    },
    "Mg_stearate": {"type": "lubricant", "ph_effect": 0.5, "moisture_risk": 0.0, "redox_risk": 0.0},
    "talc": {"type": "glidant", "ph_effect": 0.0, "moisture_risk": 0.0, "redox_risk": 0.0},
    "SDS": {"type": "surfactant", "ph_effect": -0.5, "moisture_risk": 0.1, "redox_risk": 0.2},
}

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}
_VALID_CLASSES = {"excellent", "good", "moderate", "poor"}


@dataclass
class CompatibilityResult:
    """Result of drug-excipient compatibility prediction."""

    drug_name: str
    excipient_name: str
    drug_pka: float
    drug_logp: float
    drug_ionization_type: str
    compatibility_score: float
    compatibility_class: str
    risk_factors: list
    ph_effect: float
    moisture_risk_score: float
    redox_risk_score: float
    notes: str


def _validate_inputs(
    drug_pka: float,
    drug_logp: float,
    drug_ionization_type: str,
    excipient_name: str,
) -> None:
    """Raise ValueError if any input is invalid."""
    if not (0 < drug_pka <= 14):
        raise ValueError(f"drug_pka must be in (0, 14], got {drug_pka}")
    if not (-5 < drug_logp < 10):
        raise ValueError(f"drug_logp must be in (-5, 10), got {drug_logp}")
    if drug_ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"drug_ionization_type must be one of {_VALID_IONIZATION_TYPES}, "
            f"got '{drug_ionization_type}'"
        )
    if excipient_name not in _EXCIPIENTS:
        raise ValueError(
            f"excipient_name '{excipient_name}' not in database. "
            f"Valid options: {sorted(_EXCIPIENTS.keys())}"
        )


def predict_compatibility(
    drug_name: str,
    drug_pka: float,
    drug_ionization_type: str,
    drug_logp: float,
    excipient_name: str,
) -> CompatibilityResult:
    """Predict physicochemical compatibility between a drug and one excipient.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    drug_pka:
        Drug pKa value in (0, 14].
    drug_ionization_type:
        One of "acid", "base", or "neutral".
    drug_logp:
        Drug log P value in (-5, 10).
    excipient_name:
        Name of the excipient — must be a key in the internal database.

    Returns
    -------
    CompatibilityResult
    """
    _validate_inputs(drug_pka, drug_logp, drug_ionization_type, excipient_name)

    exc = _EXCIPIENTS[excipient_name]
    ph_effect: float = float(exc["ph_effect"])
    moisture_risk: float = float(exc["moisture_risk"])
    redox_risk: float = float(exc["redox_risk"])

    risk_factors: list = []

    # --- Acid-base incompatibility ---
    conflict_score = 0.0
    if drug_ionization_type == "acid" and drug_pka < 7 and ph_effect > 1.0:
        conflict_score += 20.0
        risk_factors.append("acid-base conflict")
    elif drug_ionization_type == "base" and drug_pka > 7 and ph_effect < -1.0:
        conflict_score += 20.0
        risk_factors.append("acid-base conflict")

    # --- Moisture sensitivity ---
    if drug_logp < 1.0:
        moisture_score = moisture_risk * 30.0
    else:
        moisture_score = moisture_risk * 10.0
    if moisture_score > 0.0:
        risk_factors.append("moisture sensitivity")

    # --- Redox risk ---
    drug_is_susceptible = drug_pka > 8 or drug_logp > 3
    if drug_is_susceptible:
        redox_score = redox_risk * 25.0
    else:
        redox_score = redox_risk * 10.0
    if redox_score > 0.0:
        risk_factors.append("redox degradation")

    # --- Total risk and compatibility score ---
    total_risk = conflict_score + moisture_score + redox_score
    raw_score = 100.0 - total_risk
    compatibility_score = max(0.0, min(100.0, raw_score))

    if compatibility_score >= 80.0:
        compatibility_class = "excellent"
    elif compatibility_score >= 60.0:
        compatibility_class = "good"
    elif compatibility_score >= 40.0:
        compatibility_class = "moderate"
    else:
        compatibility_class = "poor"

    notes = (
        f"Drug '{drug_name}' ({drug_ionization_type}, pKa={drug_pka:.1f}, "
        f"logP={drug_logp:.2f}) vs excipient '{excipient_name}' "
        f"(type={exc['type']}). Identified risks: "
        f"{risk_factors if risk_factors else ['none']}."
    )

    return CompatibilityResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        drug_pka=drug_pka,
        drug_logp=drug_logp,
        drug_ionization_type=drug_ionization_type,
        compatibility_score=compatibility_score,
        compatibility_class=compatibility_class,
        risk_factors=risk_factors,
        ph_effect=ph_effect,
        moisture_risk_score=moisture_score,
        redox_risk_score=redox_score,
        notes=notes,
    )


def screen_excipient_panel(
    drug_name: str,
    drug_pka: float,
    drug_ionization_type: str,
    drug_logp: float,
) -> list:
    """Screen a drug against all 10 excipients in the database.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    drug_pka:
        Drug pKa value in (0, 14].
    drug_ionization_type:
        One of "acid", "base", or "neutral".
    drug_logp:
        Drug log P value in (-5, 10).

    Returns
    -------
    list of CompatibilityResult sorted by compatibility_score descending.
    """
    results = [
        predict_compatibility(drug_name, drug_pka, drug_ionization_type, drug_logp, exc_name)
        for exc_name in _EXCIPIENTS
    ]
    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
