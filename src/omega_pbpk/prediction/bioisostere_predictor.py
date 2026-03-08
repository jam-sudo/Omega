"""
Phase 432 — Bioisostere Predictor
Predict bioisosteric replacements and their PK impact.

Bioisosteres are functional group replacements that maintain biological activity
while improving PK properties such as metabolic stability, solubility, and permeability.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BioisostereResult:
    drug_name: str
    original_group: str
    replacement_group: str
    category: str  # "classical" or "non_classical"
    logp_delta: float
    pka_delta: float
    mw_delta_Da: float
    h_bond_donor_delta: int
    h_bond_acceptor_delta: int
    metabolic_stability_improvement: str  # "none", "minor", "moderate", "major"
    solubility_impact: str  # "decreased", "unchanged", "increased"
    permeability_impact: str  # "decreased", "unchanged", "increased"
    predicted_f_oral_change: float  # multiplicative factor on F (1.0 = no change)
    toxicity_flag: str  # "none", "monitor", "avoid"
    notes: str


# Bioisostere database — key: "original→replacement"
BIOISOSTERES = {
    "COOH->tetrazole": {
        "logp_delta": 0.0,
        "pka_delta": 0.5,
        "mw_delta": 26,
        "hbd_delta": -1,
        "hba_delta": 3,
        "metabolic": "major",
        "solubility": "increased",
        "permeability": "increased",
        "f_factor": 1.3,
        "category": "non_classical",
        "toxicity": "none",
    },
    "COOH->sulfonamide": {
        "logp_delta": -0.5,
        "pka_delta": 4.0,
        "mw_delta": 17,
        "hbd_delta": 1,
        "hba_delta": 2,
        "metabolic": "moderate",
        "solubility": "unchanged",
        "permeability": "unchanged",
        "f_factor": 1.1,
        "category": "classical",
        "toxicity": "monitor",
    },
    "OH->F": {
        "logp_delta": 0.8,
        "pka_delta": -3.0,
        "mw_delta": 2,
        "hbd_delta": -1,
        "hba_delta": -1,
        "metabolic": "moderate",
        "solubility": "decreased",
        "permeability": "increased",
        "f_factor": 1.2,
        "category": "classical",
        "toxicity": "none",
    },
    "NH2->F": {
        "logp_delta": 1.0,
        "pka_delta": -5.0,
        "mw_delta": 3,
        "hbd_delta": -2,
        "hba_delta": -1,
        "metabolic": "major",
        "solubility": "decreased",
        "permeability": "increased",
        "f_factor": 1.1,
        "category": "classical",
        "toxicity": "none",
    },
    "CH3->CF3": {
        "logp_delta": 1.5,
        "pka_delta": 0.0,
        "mw_delta": 54,
        "hbd_delta": 0,
        "hba_delta": 3,
        "metabolic": "major",
        "solubility": "decreased",
        "permeability": "increased",
        "f_factor": 1.0,
        "category": "classical",
        "toxicity": "monitor",
    },
    "phenyl->thiophene": {
        "logp_delta": 0.5,
        "pka_delta": 0.0,
        "mw_delta": 2,
        "hbd_delta": 0,
        "hba_delta": 1,
        "metabolic": "moderate",
        "solubility": "unchanged",
        "permeability": "unchanged",
        "f_factor": 1.1,
        "category": "non_classical",
        "toxicity": "monitor",
    },
    "ester->amide": {
        "logp_delta": -0.5,
        "pka_delta": 0.0,
        "mw_delta": -1,
        "hbd_delta": 1,
        "hba_delta": 0,
        "metabolic": "major",
        "solubility": "increased",
        "permeability": "unchanged",
        "f_factor": 1.2,
        "category": "classical",
        "toxicity": "none",
    },
    "ketone->CF2": {
        "logp_delta": 1.2,
        "pka_delta": 0.0,
        "mw_delta": 20,
        "hbd_delta": 0,
        "hba_delta": 2,
        "metabolic": "major",
        "solubility": "unchanged",
        "permeability": "increased",
        "f_factor": 1.1,
        "category": "non_classical",
        "toxicity": "none",
    },
}

# Internal separator used as key
_SEP = "->"

# Also support Unicode arrow as external API for user convenience
_UNICODE_SEP = "\u2192"


def _build_key(original_group: str, replacement_group: str) -> str:
    """Build the internal database key."""
    return f"{original_group}{_SEP}{replacement_group}"


def _make_notes(original_group: str, replacement_group: str, entry: dict) -> str:
    """Generate a human-readable summary note."""
    metabolic_map = {
        "none": "no improvement",
        "minor": "minor improvement",
        "moderate": "moderate improvement",
        "major": "major improvement",
    }
    metabolic_desc = metabolic_map.get(entry["metabolic"], entry["metabolic"])
    f_pct = (entry["f_factor"] - 1.0) * 100.0
    sign = "+" if f_pct >= 0 else ""
    return (
        f"Replacing {original_group} with {replacement_group}: "
        f"metabolic stability {metabolic_desc}; "
        f"solubility {entry['solubility']}; "
        f"permeability {entry['permeability']}; "
        f"predicted F oral change {sign}{f_pct:.0f}%; "
        f"logP delta {entry['logp_delta']:+.1f}; "
        f"toxicity flag: {entry['toxicity']}."
    )


def predict_bioisostere(
    drug_name: str,
    original_group: str,
    replacement_group: str,
) -> BioisostereResult:
    """
    Predict the PK impact of a bioisosteric replacement.

    Parameters
    ----------
    drug_name : str
        Name of the drug molecule.
    original_group : str
        Functional group being replaced (e.g., "COOH", "OH").
    replacement_group : str
        Bioisosteric replacement (e.g., "tetrazole", "F").

    Returns
    -------
    BioisostereResult
        Predicted PK properties of the bioisosteric replacement.

    Raises
    ------
    ValueError
        If the replacement pair is not in the database.
    """
    key = _build_key(original_group, replacement_group)
    if key not in BIOISOSTERES:
        raise ValueError(
            f"Bioisostere pair '{original_group}{_UNICODE_SEP}{replacement_group}' not in database"
        )

    entry = BIOISOSTERES[key]

    return BioisostereResult(
        drug_name=drug_name,
        original_group=original_group,
        replacement_group=replacement_group,
        category=entry["category"],
        logp_delta=entry["logp_delta"],
        pka_delta=entry["pka_delta"],
        mw_delta_Da=float(entry["mw_delta"]),
        h_bond_donor_delta=entry["hbd_delta"],
        h_bond_acceptor_delta=entry["hba_delta"],
        metabolic_stability_improvement=entry["metabolic"],
        solubility_impact=entry["solubility"],
        permeability_impact=entry["permeability"],
        predicted_f_oral_change=entry["f_factor"],
        toxicity_flag=entry["toxicity"],
        notes=_make_notes(original_group, replacement_group, entry),
    )


def suggest_bioisosteres(
    drug_name: str,
    functional_groups_present: list,
) -> list:
    """
    Suggest bioisosteric replacements for all functional groups present.

    Parameters
    ----------
    drug_name : str
        Name of the drug molecule.
    functional_groups_present : list of str
        Functional groups present in the molecule (original groups).

    Returns
    -------
    list of BioisostereResult
        All matching replacements, sorted by predicted_f_oral_change descending.
    """
    if not functional_groups_present:
        return []

    results = []
    for key, entry in BIOISOSTERES.items():
        parts = key.split(_SEP, 1)
        if len(parts) != 2:
            continue
        orig, repl = parts
        if orig in functional_groups_present:
            result = BioisostereResult(
                drug_name=drug_name,
                original_group=orig,
                replacement_group=repl,
                category=entry["category"],
                logp_delta=entry["logp_delta"],
                pka_delta=entry["pka_delta"],
                mw_delta_Da=float(entry["mw_delta"]),
                h_bond_donor_delta=entry["hbd_delta"],
                h_bond_acceptor_delta=entry["hba_delta"],
                metabolic_stability_improvement=entry["metabolic"],
                solubility_impact=entry["solubility"],
                permeability_impact=entry["permeability"],
                predicted_f_oral_change=entry["f_factor"],
                toxicity_flag=entry["toxicity"],
                notes=_make_notes(orig, repl, entry),
            )
            results.append(result)

    results.sort(key=lambda r: r.predicted_f_oral_change, reverse=True)
    return results
