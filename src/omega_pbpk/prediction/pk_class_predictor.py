"""PK class predictor — classifies compounds into PK archetypes based on physicochemical properties."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PKClassResult",
    "predict_pk_class",
    "screen_pk_classes",
    "generate_formulation_recommendation",
]

_CLASS_NAMES = {
    "I": "ideal_oral",
    "II": "low_solubility",
    "III": "low_permeability",
    "IV": "problematic",
    "V": "biologic",
    "VI": "cns",
}

_CLASS_BIOAVAILABILITY = {
    "I": "high",
    "II": "moderate",
    "III": "low",
    "IV": "low",
    "V": "parenteral_only",
    "VI": "high",
}

_CLASS_CHALLENGES = {
    "I": "Minimal formulation challenges; standard development path",
    "II": "Poor dissolution-limited absorption; solubilization required",
    "III": "Poor permeability; absorption enhancement or prodrug strategies",
    "IV": "Both poor solubility and poor permeability; major development hurdle",
    "V": "Large molecule; requires specialized delivery (SC/IV) and PBPK modeling",
    "VI": "CNS delivery; BBB penetration confirmed; monitor CNS side effects",
}


@dataclass(frozen=True)
class PKClassResult:
    """Result from PK class prediction."""

    compound_name: str
    logP: float
    mw: float
    psa: float
    pk_class: str
    pk_class_name: str
    oral_bioavailability_prediction: str
    development_challenge: str
    notes: str


def _classify(logP: float, mw: float, psa: float) -> str:
    """Internal classification logic (priority order)."""
    # Priority 1: biologic (large molecule)
    if mw > 1000:
        return "V"

    # BBB score
    bbb_penetrant = 1.0 <= logP <= 5.0 and mw < 450 and psa < 90

    # Priority 2: Class IV — both poor solubility AND poor permeability
    if logP > 4 and psa > 100:
        return "IV"

    # Priority 3: Class II — high logP (poor solubility)
    if logP > 3:
        return "II"

    # Priority 4: Class III — high PSA (poor permeability)
    if psa > 100:
        return "III"

    # Priority 5: Class VI — CNS
    if bbb_penetrant:
        return "VI"

    # Default: Class I
    return "I"


def predict_pk_class(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
) -> PKClassResult:
    """Classify a compound into a PK archetype class.

    Args:
        compound_name: Name of the compound.
        logP: Lipophilicity (octanol-water partition coefficient).
        mw: Molecular weight (Da).
        psa: Polar surface area (Å²).
        pka_acid: Acidic pKa (optional, currently informational).
        pka_base: Basic pKa (optional, currently informational).

    Returns:
        PKClassResult with class, bioavailability prediction, and development notes.
    """
    # Validation
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    pk_class = _classify(logP, mw, psa)
    pk_class_name = _CLASS_NAMES[pk_class]
    oral_ba = _CLASS_BIOAVAILABILITY[pk_class]
    challenge = _CLASS_CHALLENGES[pk_class]

    # Permeability score
    if 1 <= logP <= 5:
        perm_score = 30
    elif 0 <= logP < 1 or 5 < logP <= 7:
        perm_score = 20
    else:
        perm_score = 10

    # Solubility penalty
    if logP > 4:
        sol_penalty = 20
    elif logP > 3:
        sol_penalty = 10
    else:
        sol_penalty = 0

    # Absorption score (PSA-based)
    if psa < 60:
        abs_score = 30
    elif psa < 90:
        abs_score = 20
    else:
        abs_score = 10

    oral_score = perm_score - sol_penalty + abs_score

    note_parts = [
        f"logP={logP:.2f}, MW={mw:.1f}, PSA={psa:.1f}.",
        f"Oral composite score={oral_score}.",
    ]
    if pka_acid is not None:
        note_parts.append(f"pKa_acid={pka_acid:.1f}.")
    if pka_base is not None:
        note_parts.append(f"pKa_base={pka_base:.1f}.")
    note_parts.append(f"Assigned to Class {pk_class} ({pk_class_name}).")

    return PKClassResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        pk_class=pk_class,
        pk_class_name=pk_class_name,
        oral_bioavailability_prediction=oral_ba,
        development_challenge=challenge,
        notes=" ".join(note_parts),
    )


def screen_pk_classes(compounds: list) -> list:
    """Screen a list of compounds and classify each.

    Each dict: {"compound_name": str, "logP": float, "mw": float, "psa": float,
                optionally "pka_acid": float, "pka_base": float}

    Returns sorted by pk_class ascending (Class I first).
    """
    results = []
    for c in compounds:
        result = predict_pk_class(
            compound_name=c.get("compound_name", "unknown"),
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            pka_acid=c.get("pka_acid"),
            pka_base=c.get("pka_base"),
        )
        results.append(result)

    results.sort(key=lambda r: r.pk_class)
    return results


def generate_formulation_recommendation(pk_class_result: PKClassResult) -> str:
    """Generate a formulation recommendation based on PK class.

    Args:
        pk_class_result: A PKClassResult from predict_pk_class.

    Returns:
        Recommendation string.
    """
    recommendations = {
        "I": "Standard oral tablet/capsule",
        "II": "Consider solid dispersion, nano-milling, or lipid-based formulation",
        "III": "Absorption enhancers or prodrug approach",
        "IV": "Major formulation challenge; consider parenteral",
        "V": "Subcutaneous/IV administration; PBPK modeling needed",
        "VI": "Standard CNS delivery; consider prodrug to improve BBB penetration",
    }
    return recommendations.get(pk_class_result.pk_class, "Consult formulation scientist")
