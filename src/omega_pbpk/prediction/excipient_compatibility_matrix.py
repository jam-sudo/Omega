"""
Phase 211 — Excipient Compatibility Matrix
Drug-excipient compatibility based on known chemical incompatibilities.
Pure Python (stdlib only: math, dataclasses).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Excipient database
# ---------------------------------------------------------------------------

_EXCIPIENT_CLASSES: dict[str, str] = {
    "lactose": "reducing_sugar",
    "microcrystalline_cellulose": "cellulose",
    "croscarmellose_sodium": "cellulose_derivative",
    "magnesium_stearate": "metal_salt",
    "povidone": "polymer",
    "HPMC": "polymer",
    "calcium_phosphate": "calcium_salt",
    "starch": "polysaccharide",
    "mannitol": "polyol",
    "talc": "mineral",
}

# Aliases / alternative forms treated as polysaccharide for Maillard
_REDUCING_SUGARS = {"reducing_sugar", "polysaccharide"}

# Score constants
_SCORE_HIGH = 3
_SCORE_MODERATE = 2
_SCORE_LOW = 1
_SCORE_NONE = 0

_RISK_LABELS = {0: "none", 1: "low", 2: "moderate", 3: "high"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompatibilityResult:
    """Frozen result for a single drug vs. excipient set assessment."""

    drug_name: str
    excipients_tested: tuple
    compatibility_scores: tuple  # tuple of int, one per excipient
    overall_risk: str  # "none" | "low" | "moderate" | "high"
    max_score: int
    flagged_combinations: tuple  # tuple of str descriptions
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _excipient_class(excipient: str) -> str:
    """Return the class of *excipient* or raise ValueError if unknown."""
    if excipient not in _EXCIPIENT_CLASSES:
        raise ValueError(
            f"Unknown excipient '{excipient}'. Known excipients: {sorted(_EXCIPIENT_CLASSES)}"
        )
    return _EXCIPIENT_CLASSES[excipient]


def _score_pair(drug_properties: dict[str, bool], exc_class: str) -> tuple[int, str]:
    """
    Return (score, flag_description) for a drug + excipient class pair.

    Rules:
    - Amine + reducing_sugar/polysaccharide → Maillard reaction: 3 (HIGH)
    - Carboxyl + calcium_salt             → insoluble salt:      2 (MODERATE)
    - Amine + metal_salt                  → chelation:           2 (MODERATE)
    - Oxidation-sensitive + mineral       → catalytic oxidation: 1 (LOW)
    - Otherwise                           → 0 (NONE)
    """
    has_amine = bool(drug_properties.get("has_amine", False))
    has_carboxyl = bool(drug_properties.get("has_carboxyl", False))
    oxidation_sensitive = bool(drug_properties.get("oxidation_sensitive", False))

    if has_amine and exc_class in _REDUCING_SUGARS:
        return _SCORE_HIGH, "Maillard reaction (amine + reducing sugar)"
    if has_carboxyl and exc_class == "calcium_salt":
        return _SCORE_MODERATE, "Insoluble salt formation (carboxyl + calcium salt)"
    if has_amine and exc_class == "metal_salt":
        return _SCORE_MODERATE, "Chelation (amine + metal salt)"
    if oxidation_sensitive and exc_class == "mineral":
        return _SCORE_LOW, "Catalytic oxidation (oxidation-sensitive drug + mineral)"
    return _SCORE_NONE, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_compatibility(
    drug_name: str,
    drug_properties: dict[str, bool],
    excipients: list[str],
) -> CompatibilityResult:
    """
    Assess drug-excipient compatibility.

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    drug_properties : dict
        Keys: "has_amine" (bool), "has_carboxyl" (bool), "oxidation_sensitive" (bool).
    excipients : list of str
        Excipient names to evaluate (must exist in _EXCIPIENT_CLASSES).

    Returns
    -------
    CompatibilityResult

    Raises
    ------
    ValueError
        If *excipients* is empty or contains an unknown excipient name.
    """
    if not excipients:
        raise ValueError("excipients list must not be empty")

    scores: list[int] = []
    flagged: list[str] = []

    for exc in excipients:
        exc_class = _excipient_class(exc)  # raises ValueError if unknown
        score, flag_desc = _score_pair(drug_properties, exc_class)
        scores.append(score)
        if score > 0:
            flagged.append(f"{exc} ({flag_desc})")

    max_score = max(scores)
    overall_risk = _RISK_LABELS[max_score]

    note_parts: list[str] = []
    if not flagged:
        note_parts.append("No significant incompatibilities detected.")
    else:
        note_parts.append(f"{len(flagged)} incompatibility flag(s) identified.")

    return CompatibilityResult(
        drug_name=drug_name,
        excipients_tested=tuple(excipients),
        compatibility_scores=tuple(scores),
        overall_risk=overall_risk,
        max_score=max_score,
        flagged_combinations=tuple(flagged),
        notes=" ".join(note_parts),
    )


def screen_excipient_set(
    drug_name: str,
    drug_properties: dict[str, bool],
    excipient_sets: list[list[str]],
) -> list[CompatibilityResult]:
    """
    Screen multiple excipient sets for a drug.

    Returns a list of CompatibilityResult objects sorted by overall_risk
    ascending (safest first).

    Parameters
    ----------
    drug_name : str
    drug_properties : dict
    excipient_sets : list of list of str

    Returns
    -------
    list of CompatibilityResult, sorted by max_score ascending
    """
    results = [
        assess_compatibility(drug_name, drug_properties, exc_set) for exc_set in excipient_sets
    ]
    return sorted(results, key=lambda r: r.max_score)
