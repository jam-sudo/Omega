"""Phase 1146 — Drug-polymer interaction prediction for solid dosage forms.

Predicts interaction strength between a drug and common pharmaceutical
polymers used in tablets, capsules, and amorphous solid dispersions (ASD).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Polymer database
# ---------------------------------------------------------------------------

_POLYMERS: dict[str, dict] = {
    "HPMC": {
        "charge": "neutral",
        "hbd": 3,
        "hba": 10,
        "logP_optimum": 1.5,
        "ionic": False,
    },
    "PVP": {
        "charge": "neutral",
        "hbd": 0,
        "hba": 1,
        "logP_optimum": 2.0,
        "ionic": False,
    },
    "HPMC-AS": {
        "charge": "anionic",
        "hbd": 3,
        "hba": 12,
        "logP_optimum": 2.5,
        "ionic": True,
    },
    "Eudragit-E": {
        "charge": "cationic",
        "hbd": 0,
        "hba": 2,
        "logP_optimum": 3.0,
        "ionic": True,
    },
    "Eudragit-L": {
        "charge": "anionic",
        "hbd": 2,
        "hba": 6,
        "logP_optimum": 2.0,
        "ionic": True,
    },
    "PEG": {
        "charge": "neutral",
        "hbd": 0,
        "hba": 2,
        "logP_optimum": 1.0,
        "ionic": False,
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugPolymerInteractionResult:
    drug_name: str
    polymer_name: str
    logP: float
    ionization_type: str
    has_h_bond_donor: bool
    has_h_bond_acceptor: bool
    h_bond_score: float
    logp_compatibility_score: float
    ionic_interaction_score: float
    interaction_score: float
    interaction_class: str
    recommended_use: str
    notes: str


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _h_bond_score(
    has_h_bond_donor: bool,
    has_h_bond_acceptor: bool,
    polymer: dict,
) -> float:
    """Compute H-bond score (0-40)."""
    # drug HBD interacts with polymer HBA
    drug_hbd_score = min(40, int(has_h_bond_donor) * polymer["hba"] * 4)
    # drug HBA interacts with polymer HBD
    drug_hba_score = min(10, int(has_h_bond_acceptor) * polymer["hbd"] * 2)
    return float(min(40, drug_hbd_score + drug_hba_score))


def _logp_score(logP: float, polymer: dict) -> float:
    """Compute logP compatibility score (0-30)."""
    diff = abs(logP - polymer["logP_optimum"])
    if diff < 1:
        return 30.0
    elif diff < 2:
        return 20.0
    elif diff < 3:
        return 10.0
    else:
        return 5.0


def _ionic_score(ionization_type: str, polymer: dict) -> float:
    """Compute ionic interaction score (0-30)."""
    if not polymer["ionic"]:
        return 0.0
    charge = polymer["charge"]
    if ionization_type == "base" and charge == "anionic":
        return 30.0
    elif ionization_type == "acid" and charge == "cationic":
        return 25.0
    else:
        return 5.0


def _recommended_use(score: float) -> str:
    if score >= 70:
        return "ASD"
    elif score >= 50:
        return "matrix"
    elif score >= 30:
        return "coating"
    else:
        return "not_recommended"


def _interaction_class(score: float) -> str:
    if score >= 70:
        return "strong"
    elif score >= 40:
        return "moderate"
    else:
        return "weak"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def predict_drug_polymer_interaction(
    drug_name: str,
    logP: float,
    ionization_type: str,
    pka: float,
    has_h_bond_donor: bool,
    has_h_bond_acceptor: bool,
    polymer_name: str,
) -> DrugPolymerInteractionResult:
    """Predict interaction between a drug and a specific polymer.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    ionization_type:
        ``"acid"``, ``"base"``, or ``"neutral"``.
    pka:
        pKa of the drug (0-14).
    has_h_bond_donor:
        Whether the drug has H-bond donor groups.
    has_h_bond_acceptor:
        Whether the drug has H-bond acceptor groups.
    polymer_name:
        Name of the polymer (must be in the polymer database).

    Returns
    -------
    DrugPolymerInteractionResult
    """
    # --- Validation ---
    if not (-5 <= logP <= 10):
        raise ValueError("logP must be in [-5, 10]")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError("ionization_type must be 'acid', 'base', or 'neutral'")
    if not (0 <= pka <= 14):
        raise ValueError("pka must be in [0, 14]")
    if polymer_name not in _POLYMERS:
        raise ValueError(
            f"polymer_name '{polymer_name}' not found. Valid options: {sorted(_POLYMERS)}"
        )

    polymer = _POLYMERS[polymer_name]

    hb_score = _h_bond_score(has_h_bond_donor, has_h_bond_acceptor, polymer)
    lp_score = _logp_score(logP, polymer)
    ion_score = _ionic_score(ionization_type, polymer)

    total = min(100.0, hb_score + lp_score + ion_score)

    notes = (
        f"H-bond={hb_score:.0f}, logP_compat={lp_score:.0f}, "
        f"ionic={ion_score:.0f}; polymer_charge={polymer['charge']}"
    )

    return DrugPolymerInteractionResult(
        drug_name=drug_name,
        polymer_name=polymer_name,
        logP=logP,
        ionization_type=ionization_type,
        has_h_bond_donor=has_h_bond_donor,
        has_h_bond_acceptor=has_h_bond_acceptor,
        h_bond_score=hb_score,
        logp_compatibility_score=lp_score,
        ionic_interaction_score=ion_score,
        interaction_score=total,
        interaction_class=_interaction_class(total),
        recommended_use=_recommended_use(total),
        notes=notes,
    )


def rank_polymers_for_drug(
    drug_name: str,
    logP: float,
    ionization_type: str,
    pka: float,
    has_h_bond_donor: bool,
    has_h_bond_acceptor: bool,
) -> list[DrugPolymerInteractionResult]:
    """Rank all polymers for a given drug by interaction score (descending).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    ionization_type:
        ``"acid"``, ``"base"``, or ``"neutral"``.
    pka:
        pKa of the drug (0-14).
    has_h_bond_donor:
        Whether the drug has H-bond donor groups.
    has_h_bond_acceptor:
        Whether the drug has H-bond acceptor groups.

    Returns
    -------
    list of DrugPolymerInteractionResult sorted by interaction_score descending.
    """
    results = [
        predict_drug_polymer_interaction(
            drug_name=drug_name,
            logP=logP,
            ionization_type=ionization_type,
            pka=pka,
            has_h_bond_donor=has_h_bond_donor,
            has_h_bond_acceptor=has_h_bond_acceptor,
            polymer_name=p_name,
        )
        for p_name in _POLYMERS
    ]
    return sorted(results, key=lambda r: r.interaction_score, reverse=True)
