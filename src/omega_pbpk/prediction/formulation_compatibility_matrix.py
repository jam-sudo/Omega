"""Phase 1066 — Drug Formulation Compatibility Matrix.

Predict pairwise excipient-drug compatibility for formulation development.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ExcipientProps:
    name: str
    excipient_type: str
    pH: float
    reducing_sugar: bool
    acidic: bool
    basic: bool


_EXCIPIENTS: dict[str, _ExcipientProps] = {
    "lactose": _ExcipientProps("lactose", "filler", 6.5, True, False, False),
    "MCC": _ExcipientProps("MCC", "filler", 6.5, False, False, False),
    "starch": _ExcipientProps("starch", "filler", 6.5, False, False, False),
    "HPMC": _ExcipientProps("HPMC", "binder", 6.8, False, False, False),
    "PVP": _ExcipientProps("PVP", "binder", 7.0, False, False, False),
    "magnesium_stearate": _ExcipientProps(
        "magnesium_stearate", "lubricant", 8.5, False, False, True
    ),
    "citric_acid": _ExcipientProps("citric_acid", "acidulant", 2.5, False, True, False),
    "sodium_bicarbonate": _ExcipientProps(
        "sodium_bicarbonate", "basifier", 8.5, False, False, True
    ),
    "talc": _ExcipientProps("talc", "glidant", 7.0, False, False, False),
    "croscarmellose": _ExcipientProps("croscarmellose", "disintegrant", 6.5, False, False, False),
}

_VALID_EXCIPIENTS = frozenset(_EXCIPIENTS.keys())

_ADSORBING_EXCIPIENTS = {"HPMC", "PVP"}


@dataclass(frozen=True)
class CompatibilityResult:
    drug_name: str
    excipient: str
    compatibility_score: float
    interaction_type: str
    degradation_risk: str
    mechanism: str
    recommended_ratio: str
    stability_impact_pct: float
    notes: str


def predict_compatibility(
    drug_name: str,
    mw_Da: float,
    logp: float,
    has_amine: bool = False,
    has_ester: bool = False,
    has_aldehyde: bool = False,
    has_acid: bool = False,
    has_hydroxyl: bool = False,
    excipient: str = "lactose",
) -> CompatibilityResult:
    """Predict drug-excipient compatibility."""
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if excipient not in _VALID_EXCIPIENTS:
        raise ValueError(f"excipient must be one of {sorted(_VALID_EXCIPIENTS)}, got '{excipient}'")

    exc = _EXCIPIENTS[excipient]
    score = 100.0
    mechanism_parts: list[str] = []
    chemical_interaction = False
    physical_interaction = False

    # Maillard reaction
    if has_amine and exc.reducing_sugar:
        score -= 40
        mechanism_parts.append("amine-carbonyl Maillard")
        chemical_interaction = True

    # Acid-catalyzed hydrolysis
    if has_ester and exc.acidic:
        score -= 30
        mechanism_parts.append("acid-ester hydrolysis")
        chemical_interaction = True

    # Aldehyde reaction
    if has_aldehyde:
        if exc.reducing_sugar:
            score -= 25
        else:
            score -= 10
        mechanism_parts.append("aldehyde oxidation")
        chemical_interaction = True

    # Basic degradation
    if exc.basic and has_ester:
        score -= 20
        mechanism_parts.append("base-ester saponification")
        chemical_interaction = True
    if exc.basic and has_acid:
        score -= 15
        mechanism_parts.append("acid-base neutralization")
        chemical_interaction = True

    # pH mismatch
    if exc.pH < 4 and has_ester:
        score -= 10
        chemical_interaction = True
    if exc.pH > 8 and has_amine:
        score -= 10
        chemical_interaction = True

    # Adsorption
    if excipient in _ADSORBING_EXCIPIENTS and mw_Da < 400:
        score -= 5
        physical_interaction = True
        if "adsorption" not in mechanism_parts:
            mechanism_parts.append("adsorption")

    # Clamp
    score = max(0.0, min(100.0, score))

    # Interaction type
    if chemical_interaction and physical_interaction:
        interaction_type = "both"
    elif chemical_interaction:
        interaction_type = "chemical"
    elif physical_interaction:
        interaction_type = "physical"
    else:
        interaction_type = "none"

    # Degradation risk
    if score >= 80:
        degradation_risk = "low"
    elif score >= 60:
        degradation_risk = "moderate"
    else:
        degradation_risk = "high"

    # Stability impact
    stability_impact_pct = (100.0 - score) * 0.5

    # Recommended ratio
    if score >= 80:
        recommended_ratio = "1:10"
    elif score >= 60:
        recommended_ratio = "1:5"
    elif score >= 40:
        recommended_ratio = "1:1"
    else:
        recommended_ratio = "avoid"

    mechanism = ", ".join(mechanism_parts) if mechanism_parts else "none"

    notes = (
        f"{drug_name} with {excipient}: score={score:.1f}, risk={degradation_risk}, "
        f"ratio={recommended_ratio}"
    )

    return CompatibilityResult(
        drug_name=drug_name,
        excipient=excipient,
        compatibility_score=score,
        interaction_type=interaction_type,
        degradation_risk=degradation_risk,
        mechanism=mechanism,
        recommended_ratio=recommended_ratio,
        stability_impact_pct=stability_impact_pct,
        notes=notes,
    )


def compatibility_matrix(
    drug_name: str,
    mw_Da: float,
    logp: float,
    has_amine: bool = False,
    has_ester: bool = False,
    has_aldehyde: bool = False,
    has_acid: bool = False,
    has_hydroxyl: bool = False,
) -> list:
    """Return CompatibilityResult for all 10 excipients, sorted by compatibility_score desc."""
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    results = []
    for exc_name in _EXCIPIENTS:
        result = predict_compatibility(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            has_amine=has_amine,
            has_ester=has_ester,
            has_aldehyde=has_aldehyde,
            has_acid=has_acid,
            has_hydroxyl=has_hydroxyl,
            excipient=exc_name,
        )
        results.append(result)

    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
