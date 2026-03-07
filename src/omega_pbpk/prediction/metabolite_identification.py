"""Phase 408 — Metabolite Identification

Predict major metabolic pathways and metabolites from structural alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MetabolitePrediction",
    "MetaboliteResult",
    "predict_metabolites",
    "rank_metabolic_pathways",
]

# ---------------------------------------------------------------------------
# Valid functional groups
# ---------------------------------------------------------------------------

VALID_FUNCTIONAL_GROUPS: frozenset[str] = frozenset(
    [
        "alcohol",
        "aldehyde",
        "amine",
        "aromatic",
        "ester",
        "amide",
        "sulfide",
        "alkene",
        "nitro",
    ]
)

# Likelihood ordering for sorting
_LIKELIHOOD_ORDER: dict[str, int] = {"high": 0, "moderate": 1, "low": 2}

# ---------------------------------------------------------------------------
# Metabolite prediction rules
# ---------------------------------------------------------------------------

# Each rule: (functional_group, metabolite_type, reaction, enzyme, likelihood)
_METABOLITE_RULES: list[tuple[str, str, str, str, str]] = [
    # Phase I — CYP-mediated
    ("aromatic", "aromatic_hydroxylation", "aromatic hydroxylation", "CYP3A4", "high"),
    ("aromatic", "aromatic_hydroxylation", "aromatic hydroxylation", "CYP2D6", "moderate"),
    ("amine", "N-dealkylation", "N-dealkylation", "CYP3A4", "high"),
    ("amine", "N-oxide", "N-oxidation", "CYP3A4", "moderate"),
    ("amine", "N-hydroxylation", "N-hydroxylation", "CYP2D6", "low"),
    ("alcohol", "aldehyde", "oxidation to aldehyde", "CYP2E1", "moderate"),
    ("alcohol", "carboxylic_acid", "further oxidation to carboxylic acid", "CYP2E1", "low"),
    ("sulfide", "sulfoxide", "S-oxidation to sulfoxide", "CYP3A4", "high"),
    ("sulfide", "sulfone", "S-oxidation to sulfone", "CYP3A4", "moderate"),
    ("alkene", "epoxide", "epoxidation", "CYP3A4", "high"),
    ("alkene", "diol", "trans-dihydrodiol via epoxide", "CYP3A4", "moderate"),
    ("aldehyde", "carboxylic_acid", "oxidation to carboxylic acid", "CYP2E1", "high"),
    ("nitro", "nitroso", "nitroso reduction (Phase I)", "CYP reductase", "moderate"),
    ("nitro", "hydroxylamine", "hydroxylamine formation", "CYP reductase", "high"),
    # Phase II — conjugation
    ("alcohol", "glucuronide", "glucuronidation", "UGT", "high"),
    ("alcohol", "sulfate", "sulfation", "SULT", "moderate"),
    ("amine", "glucuronide", "N-glucuronidation", "UGT", "moderate"),
    ("amine", "acetylation", "N-acetylation", "NAT", "moderate"),
    ("aromatic", "sulfate", "phenol sulfation", "SULT", "moderate"),
    ("aromatic", "glucuronide", "phenol glucuronidation", "UGT", "high"),
    # Hydrolysis
    ("ester", "alcohol_acid", "ester hydrolysis", "esterase", "high"),
    ("amide", "amine_acid", "amide hydrolysis", "amidase", "moderate"),
]

# Reactive metabolite risk sources
_REACTIVE_RISK_GROUPS: dict[str, str] = {
    "aromatic": "moderate",  # aromatic → arene oxide → reactive
    "alkene": "moderate",    # alkene → epoxide → reactive
    "amine": "moderate",     # aromatic amine → hydroxylamine → nitrenium ion
    "nitro": "high",         # nitro → hydroxylamine → very reactive
    "aldehyde": "high",      # aldehyde is inherently electrophilic
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetabolitePrediction:
    """A single predicted metabolite/pathway."""

    metabolite_type: str
    reaction: str
    enzyme: str
    likelihood: str  # "high", "moderate", "low"


@dataclass
class MetaboliteResult:
    """Full metabolite prediction result for a compound."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    predicted_metabolites: list[MetabolitePrediction] = field(default_factory=list)
    reactive_metabolite_risk: str = "low"  # "low", "moderate", "high"
    n_pathways: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    functional_groups: list[str],
) -> None:
    """Validate inputs for metabolite prediction functions."""
    if not compound_name or not compound_name.strip():
        raise ValueError("compound_name must be a non-empty string")
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if not isinstance(functional_groups, list):
        raise ValueError("functional_groups must be a list")
    invalid = [g for g in functional_groups if g not in VALID_FUNCTIONAL_GROUPS]
    if invalid:
        raise ValueError(
            f"Invalid functional group(s): {invalid}. "
            f"Valid options: {sorted(VALID_FUNCTIONAL_GROUPS)}"
        )


def _assess_reactive_risk(functional_groups: list[str]) -> str:
    """Determine overall reactive metabolite risk."""
    risk_rank = {"low": 0, "moderate": 1, "high": 2}
    current_max = 0
    for fg in functional_groups:
        if fg in _REACTIVE_RISK_GROUPS:
            level = _REACTIVE_RISK_GROUPS[fg]
            current_max = max(current_max, risk_rank[level])
    inv = {v: k for k, v in risk_rank.items()}
    return inv[current_max]


def _build_predictions(functional_groups: list[str]) -> list[MetabolitePrediction]:
    """Apply metabolite rules for the given functional groups."""
    seen: set[tuple[str, str, str, str]] = set()
    predictions: list[MetabolitePrediction] = []
    for rule in _METABOLITE_RULES:
        fg, met_type, reaction, enzyme, likelihood = rule
        if fg in functional_groups:
            key = (met_type, reaction, enzyme, likelihood)
            if key not in seen:
                seen.add(key)
                predictions.append(
                    MetabolitePrediction(
                        metabolite_type=met_type,
                        reaction=reaction,
                        enzyme=enzyme,
                        likelihood=likelihood,
                    )
                )
    return predictions


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def predict_metabolites(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    functional_groups: list[str],
) -> MetaboliteResult:
    """Predict major metabolic pathways and metabolites from structural alerts.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    smiles:
        SMILES string for the compound.
    mw:
        Molecular weight in Da.
    logP:
        Calculated logP (octanol-water partition coefficient).
    functional_groups:
        List of functional groups present. Valid values:
        "alcohol", "aldehyde", "amine", "aromatic", "ester", "amide",
        "sulfide", "alkene", "nitro".

    Returns
    -------
    MetaboliteResult
    """
    _validate_inputs(compound_name, smiles, mw, logP, functional_groups)

    predictions = _build_predictions(functional_groups)
    reactive_risk = _assess_reactive_risk(functional_groups)
    n_pathways = len(predictions)

    notes: list[str] = []
    if not functional_groups:
        notes.append("No functional groups provided; no metabolites predicted.")
    if reactive_risk in ("moderate", "high"):
        notes.append(
            f"Reactive metabolite risk is {reactive_risk}. "
            "Covalent binding and toxicity screening recommended."
        )
    if mw > 500:
        notes.append("High MW (>500 Da) may limit oral absorption and CYP access.")
    if logP > 5:
        notes.append(
            "High logP (>5): extensive tissue distribution; CYP3A4 metabolism likely."
        )
    if logP < 0:
        notes.append("Low logP (<0): hydrophilic; renal excretion may dominate over metabolism.")

    return MetaboliteResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        predicted_metabolites=predictions,
        reactive_metabolite_risk=reactive_risk,
        n_pathways=n_pathways,
        notes=notes,
    )


def rank_metabolic_pathways(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    functional_groups: list[str],
) -> list[dict[str, str]]:
    """Return predicted metabolic pathways ranked by likelihood (high first).

    Parameters
    ----------
    compound_name:
        Name of the compound.
    smiles:
        SMILES string for the compound.
    mw:
        Molecular weight in Da.
    logP:
        Calculated logP.
    functional_groups:
        List of functional groups present (same valid values as ``predict_metabolites``).

    Returns
    -------
    list[dict[str, str]]
        List of pathway dicts sorted by likelihood, each with keys:
        "metabolite_type", "reaction", "enzyme", "likelihood".
    """
    _validate_inputs(compound_name, smiles, mw, logP, functional_groups)

    predictions = _build_predictions(functional_groups)
    ranked = sorted(predictions, key=lambda p: _LIKELIHOOD_ORDER.get(p.likelihood, 99))

    return [
        {
            "metabolite_type": p.metabolite_type,
            "reaction": p.reaction,
            "enzyme": p.enzyme,
            "likelihood": p.likelihood,
        }
        for p in ranked
    ]
