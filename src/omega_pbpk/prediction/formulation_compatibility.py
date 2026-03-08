"""
Phase 936 — Drug-excipient compatibility prediction for solid oral dosage forms.

Rule-based scoring using physicochemical properties and known incompatibility patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FormulationCompatibilityResult",
    "predict_compatibility",
    "screen_excipient_compatibility",
]

EXCIPIENT_RISKS = {
    "lactose": {"amine_drug": 30, "reducing_sugar_risk": True},
    "MCC": {"hygroscopic_penalty": 10},
    "HPMC": {"poorly_soluble_penalty": 20},
    "PVP": {"compatible_all": True, "base_score": 0},
    "magnesium_stearate": {"acid_drug_risk": 15, "lubricant": True},
    "croscarmellose": {"base_score": 0},
    "talc": {"base_score": 0},
    "colloidal_silica": {"base_score": 0},
    "stearic_acid": {"acid_drug_risk": 10},
    "sodium_starch_glycolate": {"base_score": 0},
}

# Water-containing excipients (potential ester hydrolysis)
_WATER_CONTAINING = {"MCC", "HPMC", "PVP", "lactose", "croscarmellose", "sodium_starch_glycolate"}

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class FormulationCompatibilityResult:
    drug_name: str
    excipient: str
    logp: float
    pka: float
    ionization_type: str
    compatibility_score: float
    incompatibility_risks: tuple  # tuple of risk description strings
    risk_level: str  # "low" | "moderate" | "high"
    recommendation: str  # "acceptable" | "use_with_caution" | "avoid"
    notes: str


def _score_and_risks(
    logp: float,
    pka: float,
    ionization_type: str,
    excipient: str,
    has_ester_bond: bool,
) -> tuple:
    """Return (penalty_total, list_of_risk_strings) for the given combination."""
    risks = []
    penalty = 0

    exc_props = EXCIPIENT_RISKS[excipient]

    # Amine drug (base, pKa > 8) + lactose → Maillard reaction
    if excipient == "lactose":
        if ionization_type == "base" and pka > 8:
            penalty += exc_props.get("amine_drug", 0)
            risks.append(f"Maillard reaction risk: amine drug (pKa={pka:.1f}) + lactose")

    # Acidic drug (pKa < 4) + magnesium_stearate → salt form risk
    if excipient == "magnesium_stearate":
        if ionization_type == "acid" and pka < 4:
            penalty += exc_props.get("acid_drug_risk", 0)
            risks.append(f"Salt form risk: acidic drug (pKa={pka:.1f}) + magnesium stearate")

    # Acidic drug (pKa < 4) + stearic_acid → acid-drug interaction
    if excipient == "stearic_acid":
        if ionization_type == "acid" and pka < 4:
            penalty += exc_props.get("acid_drug_risk", 0)
            risks.append(f"Acid-drug interaction risk: acidic drug (pKa={pka:.1f}) + stearic acid")

    # Hygroscopic drug (logP < 0) + MCC → moisture absorption
    if excipient == "MCC":
        if logp < 0:
            penalty += exc_props.get("hygroscopic_penalty", 0)
            risks.append(f"Moisture absorption risk: hygroscopic drug (logP={logp:.1f}) + MCC")

    # Poorly soluble drug (logP > 3) + HPMC → precipitation
    if excipient == "HPMC":
        if logp > 3:
            penalty += exc_props.get("poorly_soluble_penalty", 0)
            risks.append(f"Precipitation risk: poorly soluble drug (logP={logp:.1f}) + HPMC")

    # Ester bond + water-containing excipient → hydrolysis
    if has_ester_bond and excipient in _WATER_CONTAINING:
        penalty += 25
        risks.append(
            f"Ester hydrolysis risk: ester-containing drug + water-containing excipient ({excipient})"
        )

    return penalty, risks


def predict_compatibility(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    excipient: str,
    has_ester_bond: bool = False,
) -> FormulationCompatibilityResult:
    """Predict drug-excipient compatibility for a solid oral dosage form."""
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got '{ionization_type}'"
        )
    if excipient not in EXCIPIENT_RISKS:
        raise ValueError(f"excipient '{excipient}' not in library")

    penalty, risk_list = _score_and_risks(logp, pka, ionization_type, excipient, has_ester_bond)

    raw_score = 100.0 - penalty
    score = max(0.0, min(100.0, raw_score))

    if score > 70:
        risk_level = "low"
        recommendation = "acceptable"
    elif score >= 40:
        risk_level = "moderate"
        recommendation = "use_with_caution"
    else:
        risk_level = "high"
        recommendation = "avoid"

    if risk_list:
        notes_body = "; ".join(risk_list)
    else:
        notes_body = "No known incompatibilities detected"

    notes = (
        f"drug={drug_name}, logP={logp:.2f}, pKa={pka:.1f}, "
        f"ionization={ionization_type}, ester_bond={has_ester_bond}. "
        f"{notes_body}."
    )

    return FormulationCompatibilityResult(
        drug_name=drug_name,
        excipient=excipient,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        compatibility_score=score,
        incompatibility_risks=tuple(risk_list),
        risk_level=risk_level,
        recommendation=recommendation,
        notes=notes,
    )


def screen_excipient_compatibility(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    excipients: list | None = None,
    has_ester_bond: bool = False,
) -> list:
    """Screen compatibility against all (or a subset of) excipients.

    Returns list[FormulationCompatibilityResult] sorted by compatibility_score descending.
    """
    if excipients is None:
        excipients = list(EXCIPIENT_RISKS.keys())

    results = []
    for exc in excipients:
        res = predict_compatibility(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            excipient=exc,
            has_ester_bond=has_ester_bond,
        )
        results.append(res)

    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
