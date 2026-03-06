"""Multi-drug interaction network — graph-based polypharmacy DDI analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDIEdge:
    drug_a: str
    drug_b: str
    mechanism: str  # inhibition / induction / substrate_competition
    aucr: float  # predicted AUCR (A as victim, B as perpetrator)
    severity: str  # none / mild / moderate / severe


@dataclass(frozen=True)
class DDINetworkResult:
    drugs: list[str]
    edges: list[DDIEdge]
    n_interactions: int
    n_severe: int
    n_moderate: int
    high_risk_drugs: list[str]  # drugs with >=1 severe interaction
    summary: str


# Simplified CYP substrate/inhibitor/inducer database (subset)
_CYP_DATA = {
    "warfarin": {"substrate": ["CYP2C9"], "inhibitor": [], "inducer": [], "narrow_ti": True},
    "simvastatin": {"substrate": ["CYP3A4"], "inhibitor": [], "inducer": [], "narrow_ti": False},
    "atorvastatin": {"substrate": ["CYP3A4"], "inhibitor": [], "inducer": [], "narrow_ti": False},
    "fluconazole": {
        "substrate": ["CYP2C9"],
        "inhibitor": ["CYP2C9", "CYP3A4"],
        "inducer": [],
        "narrow_ti": False,
    },
    "ketoconazole": {
        "substrate": ["CYP3A4"],
        "inhibitor": ["CYP3A4"],
        "inducer": [],
        "narrow_ti": False,
    },
    "rifampicin": {
        "substrate": [],
        "inhibitor": [],
        "inducer": ["CYP3A4", "CYP2C9"],
        "narrow_ti": False,
    },
    "phenytoin": {
        "substrate": ["CYP2C9"],
        "inhibitor": [],
        "inducer": ["CYP3A4"],
        "narrow_ti": True,
    },
    "amiodarone": {
        "substrate": ["CYP2C8"],
        "inhibitor": ["CYP2C9", "CYP2D6"],
        "inducer": [],
        "narrow_ti": True,
    },
    "metoprolol": {"substrate": ["CYP2D6"], "inhibitor": [], "inducer": [], "narrow_ti": False},
    "codeine": {"substrate": ["CYP2D6"], "inhibitor": [], "inducer": [], "narrow_ti": False},
    "fluoxetine": {
        "substrate": ["CYP2D6"],
        "inhibitor": ["CYP2D6"],
        "inducer": [],
        "narrow_ti": False,
    },
    "clarithromycin": {
        "substrate": ["CYP3A4"],
        "inhibitor": ["CYP3A4"],
        "inducer": [],
        "narrow_ti": False,
    },
    "cyclosporine": {
        "substrate": ["CYP3A4"],
        "inhibitor": ["CYP3A4"],
        "inducer": [],
        "narrow_ti": True,
    },
    "midazolam": {"substrate": ["CYP3A4"], "inhibitor": [], "inducer": [], "narrow_ti": False},
    "digoxin": {"substrate": [], "inhibitor": [], "inducer": [], "narrow_ti": True},
}

# AUCR estimates for strong/moderate inhibitors
_AUCR_STRONG = 5.0  # strong inhibitor (CYP3A4: ketoconazole)
_AUCR_MODERATE = 2.5  # moderate inhibitor
_AUCR_INDUCTION = 0.4  # strong inducer (rifampicin)


def _lookup_drug(name: str) -> dict:
    """Look up drug CYP profile; return empty if unknown."""
    return _CYP_DATA.get(
        name.lower(), {"substrate": [], "inhibitor": [], "inducer": [], "narrow_ti": False}
    )


def _aucr_for_pair(
    perpetrator_data: dict, victim_data: dict, victim_name: str, perp_name: str
) -> tuple[float, str]:
    """Estimate AUCR and mechanism for a perpetrator-victim pair."""
    # Check inhibition
    shared_inhibited = set(perpetrator_data["inhibitor"]) & set(victim_data["substrate"])
    if shared_inhibited:
        # Classify inhibition strength
        strong_inh = {
            "CYP3A4": ["ketoconazole", "itraconazole", "clarithromycin", "cyclosporine"],
            "CYP2C9": ["fluconazole", "amiodarone"],
            "CYP2D6": ["fluoxetine", "paroxetine"],
        }
        is_strong = any(perp_name.lower() in strong_inh.get(cyp, []) for cyp in shared_inhibited)
        aucr = _AUCR_STRONG if is_strong else _AUCR_MODERATE
        return aucr, "inhibition"

    # Check induction
    shared_induced = set(perpetrator_data["inducer"]) & set(victim_data["substrate"])
    if shared_induced:
        return _AUCR_INDUCTION, "induction"

    return 1.0, "none"


def _severity(aucr: float, narrow_ti: bool) -> str:
    """Classify interaction severity."""
    if aucr == 1.0:
        return "none"
    if aucr >= 5.0 or (aucr >= 2.0 and narrow_ti) or aucr <= 0.3:
        return "severe"
    elif aucr >= 2.0 or (aucr >= 1.5 and narrow_ti) or aucr <= 0.5:
        return "moderate"
    elif aucr > 1.0 or aucr < 1.0:
        return "mild"
    return "none"


def analyze_drug_network(drugs: list[str]) -> DDINetworkResult:
    """Analyze all pairwise DDIs in a drug list.

    Returns a network of interactions with severity classification.
    """
    if not drugs:
        raise ValueError("drugs must be non-empty")

    edges = []
    for i, drug_a in enumerate(drugs):
        for j, drug_b in enumerate(drugs):
            if i >= j:
                continue
            data_a = _lookup_drug(drug_a)
            data_b = _lookup_drug(drug_b)

            # A perpetrator, B victim
            aucr_ab, mech_ab = _aucr_for_pair(data_a, data_b, drug_b, drug_a)
            if mech_ab != "none":
                sev = _severity(aucr_ab, data_b.get("narrow_ti", False))
                edges.append(DDIEdge(drug_a, drug_b, mech_ab, aucr_ab, sev))

            # B perpetrator, A victim
            aucr_ba, mech_ba = _aucr_for_pair(data_b, data_a, drug_a, drug_b)
            if mech_ba != "none":
                sev = _severity(aucr_ba, data_a.get("narrow_ti", False))
                edges.append(DDIEdge(drug_b, drug_a, mech_ba, aucr_ba, sev))

    n_severe = sum(1 for e in edges if e.severity == "severe")
    n_moderate = sum(1 for e in edges if e.severity == "moderate")

    high_risk = list(set(d for e in edges if e.severity == "severe" for d in [e.drug_a, e.drug_b]))

    if n_severe > 0:
        summary = f"⚠ {n_severe} severe interaction(s) detected — review immediately"
    elif n_moderate > 0:
        summary = f"{n_moderate} moderate interaction(s) — monitor closely"
    elif edges:
        summary = "Mild interactions only — routine monitoring"
    else:
        summary = "No significant DDIs detected"

    return DDINetworkResult(
        drugs=list(drugs),
        edges=edges,
        n_interactions=len(edges),
        n_severe=n_severe,
        n_moderate=n_moderate,
        high_risk_drugs=high_risk,
        summary=summary,
    )


def ddi_matrix(drugs: list[str]) -> dict:
    """Return AUCR matrix for all pairs.

    Returns dict: {drug_a: {drug_b: aucr}} where drug_a=victim, drug_b=perpetrator.
    """
    matrix = {d: {} for d in drugs}
    for drug_a in drugs:
        for drug_b in drugs:
            if drug_a == drug_b:
                matrix[drug_a][drug_b] = 1.0
                continue
            data_b = _lookup_drug(drug_b)
            data_a = _lookup_drug(drug_a)
            aucr, _ = _aucr_for_pair(data_b, data_a, drug_a, drug_b)
            matrix[drug_a][drug_b] = aucr
    return matrix


__all__ = [
    "DDIEdge",
    "DDINetworkResult",
    "analyze_drug_network",
    "ddi_matrix",
]
