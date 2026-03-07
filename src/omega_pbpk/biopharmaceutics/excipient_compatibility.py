"""Phase 194 — Excipient-drug compatibility screening."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExcipientInteraction",
    "CompatibilityResult",
    "screen_excipient_compatibility",
    "suggest_formulation",
]

DEFAULT_EXCIPIENTS: list[str] = [
    "lactose",
    "microcrystalline_cellulose",
    "magnesium_stearate",
    "povidone",
    "croscarmellose",
    "stearic_acid",
    "talc",
    "hydroxypropyl_cellulose",
    "mannitol",
    "silicon_dioxide",
]

# Interaction knowledge base: (drug_functional_group, excipient) → ExcipientInteraction
INTERACTIONS: dict[tuple[str, str], ExcipientInteraction] = {}


@dataclass(frozen=True)
class ExcipientInteraction:
    """A single excipient-drug interaction record."""

    excipient: str
    interaction_type: str
    severity: str
    mechanism: str
    mitigation: str


# Populate the interaction table after the class is defined
def _build_interaction_table() -> dict[tuple[str, str], ExcipientInteraction]:
    return {
        ("amine", "lactose"): ExcipientInteraction(
            excipient="lactose",
            interaction_type="degradation",
            severity="high",
            mechanism=(
                "Maillard reaction: primary amine + reducing sugar \u2192 browning, degradation"
            ),
            mitigation=("Replace lactose with mannitol or microcrystalline cellulose"),
        ),
        ("ester", "magnesium_stearate"): ExcipientInteraction(
            excipient="magnesium_stearate",
            interaction_type="degradation",
            severity="moderate",
            mechanism=("Alkaline conditions from Mg stearate may hydrolyze esters"),
            mitigation=("Minimize magnesium stearate level or use calcium stearate"),
        ),
        ("nitro", "microcrystalline_cellulose"): ExcipientInteraction(
            excipient="microcrystalline_cellulose",
            interaction_type="pH_shift",
            severity="low",
            mechanism=("Minor acidic pH from MCC may affect nitro compound stability"),
            mitigation="Monitor pH and add buffer if needed",
        ),
    }


INTERACTIONS = _build_interaction_table()


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of excipient compatibility screening."""

    drug_name: str
    drug_functional_groups: list[str]
    excipients_tested: list[str]
    interactions: list[ExcipientInteraction]
    n_high_severity: int
    overall_compatibility: str
    recommended_excipients: list[str]
    excipients_to_avoid: list[str]
    notes: str


def _detect_groups(smiles: str) -> list[str]:
    """Detect functional groups from a SMILES string."""
    groups: list[str] = []
    if "N" in smiles:
        groups.append("amine")
    if "C(=O)O" in smiles or "OC(=O)" in smiles:
        groups.append("ester")
    if "C(=O)N" in smiles:
        groups.append("amide")
    if "[N+](=O)[O-]" in smiles:
        groups.append("nitro")
    if "S" in smiles:
        groups.append("sulfur")
    if "c" in smiles:
        groups.append("aromatic")
    return groups


def screen_excipient_compatibility(
    drug_name: str,
    smiles: str,
    excipients: list[str] | None = None,
) -> CompatibilityResult:
    """Screen a drug against a list of excipients for compatibility issues.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    smiles:
        SMILES string used for functional group detection.
    excipients:
        List of excipient names to screen. Defaults to :data:`DEFAULT_EXCIPIENTS`.

    Returns
    -------
    CompatibilityResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    if excipients is None:
        excipients = DEFAULT_EXCIPIENTS

    drug_groups = _detect_groups(smiles)
    interactions: list[ExcipientInteraction] = []

    for exc in excipients:
        matched = False
        for group in drug_groups:
            key = (group, exc)
            if key in INTERACTIONS:
                interactions.append(INTERACTIONS[key])
                matched = True
                break
        if not matched:
            interactions.append(
                ExcipientInteraction(
                    excipient=exc,
                    interaction_type="none",
                    severity="none",
                    mechanism="No known interaction",
                    mitigation="None required",
                )
            )

    n_high = sum(1 for i in interactions if i.severity == "high")
    n_mod = sum(1 for i in interactions if i.severity == "moderate")

    if n_high > 0:
        overall = "incompatible"
    elif n_mod > 1:
        overall = "caution"
    else:
        overall = "compatible"

    recommended = [
        exc for i, exc in zip(interactions, excipients, strict=True) if i.severity == "none"
    ]
    to_avoid = [
        exc
        for i, exc in zip(interactions, excipients, strict=True)
        if i.severity in ("high", "moderate")
    ]

    group_str = ", ".join(drug_groups) if drug_groups else "none detected"
    notes = (
        f"Functional groups detected: {group_str}; "
        f"{n_high} high, {n_mod} moderate, "
        f"{len(interactions) - n_high - n_mod} low/none severity interactions"
    )

    return CompatibilityResult(
        drug_name=drug_name,
        drug_functional_groups=drug_groups,
        excipients_tested=list(excipients),
        interactions=interactions,
        n_high_severity=n_high,
        overall_compatibility=overall,
        recommended_excipients=recommended,
        excipients_to_avoid=to_avoid,
        notes=notes,
    )


# BCS-class specific excipient suggestions
_BCS_EXCIPIENTS: dict[str, list[str]] = {
    "I": ["microcrystalline_cellulose", "mannitol", "povidone"],
    "II": [
        "hydroxypropyl_cellulose",
        "povidone",
        "polysorbate_80",
        "sodium_lauryl_sulfate",
        "microcrystalline_cellulose",
        "mannitol",
    ],
    "III": ["microcrystalline_cellulose", "mannitol", "croscarmellose"],
    "IV": [
        "hydroxypropyl_cellulose",
        "povidone",
        "polysorbate_80",
        "sodium_lauryl_sulfate",
        "microcrystalline_cellulose",
    ],
}

_ROUTE_EXCIPIENTS: dict[str, list[str]] = {
    "oral": DEFAULT_EXCIPIENTS,
    "topical": [
        "petrolatum",
        "glycerin",
        "cetyl_alcohol",
        "silicon_dioxide",
        "stearic_acid",
    ],
    "parenteral": [
        "sodium_chloride",
        "mannitol",
        "polysorbate_80",
        "benzyl_alcohol",
        "silicon_dioxide",
    ],
}


def suggest_formulation(
    drug_name: str,
    smiles: str,
    bcs_class: str = "II",
    route: str = "oral",
) -> CompatibilityResult:
    """Suggest a compatible excipient set for a given drug and BCS class.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    smiles:
        SMILES string.
    bcs_class:
        BCS classification ("I", "II", "III", or "IV").
    route:
        Route of administration ("oral", "topical", "parenteral").

    Returns
    -------
    CompatibilityResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    bcs_class_upper = bcs_class.upper()
    excipients = list(_BCS_EXCIPIENTS.get(bcs_class_upper, DEFAULT_EXCIPIENTS))

    # Merge with route-specific excipients (unique)
    route_exc = _ROUTE_EXCIPIENTS.get(route, DEFAULT_EXCIPIENTS)
    for exc in route_exc:
        if exc not in excipients:
            excipients.append(exc)

    return screen_excipient_compatibility(
        drug_name=drug_name,
        smiles=smiles,
        excipients=excipients,
    )
