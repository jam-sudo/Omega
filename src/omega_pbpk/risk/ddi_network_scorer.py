"""DDI Network Scorer — polypharmacy DDI risk assessment.

Phase 874: Score overall DDI risk of a polypharmacy regimen using
a network-based approach with CYP interaction matrix.
"""

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class DDINetworkResult:
    """Result from DDI network scoring."""

    n_drugs: int
    n_pairs_evaluated: int
    n_interactions_moderate_plus: int
    n_interactions_major_plus: int
    n_contraindicated: int
    max_aucr: float
    most_concerning_pair: str
    weighted_risk_score: float
    polypharmacy_risk: str
    interaction_summary: str
    notes: str


def _compute_pair_aucr(drug_i: dict, drug_j: dict) -> float:
    """Compute maximum AUCR for drug_j as substrate when drug_i is inhibitor."""
    css_i = drug_i.get("css_uM", 1.0)
    inhibitions_i = drug_i.get("cyp_inhibitions", {})
    substrates_j = drug_j.get("cyp_substrates", [])

    max_aucr = 1.0
    for cyp in substrates_j:
        if cyp in inhibitions_i:
            ic50 = inhibitions_i[cyp]
            if ic50 > 0:
                r = 1.0 + css_i / ic50
                if r > max_aucr:
                    max_aucr = r
    return max_aucr


def _aucr_severity(aucr: float) -> str:
    """Classify AUCR into severity category."""
    if aucr >= 5.0:
        return "contraindicated"
    elif aucr >= 2.0:
        return "major"
    elif aucr >= 1.25:
        return "moderate"
    else:
        return "minor"


def score_ddi_network(drugs: list) -> DDINetworkResult:
    """Score the overall DDI risk of a polypharmacy regimen.

    Parameters
    ----------
    drugs : list of dict
        Each dict has:
        - "name" (str)
        - "css_uM" (float, default 1.0)
        - "cyp_inhibitions" (dict: enzyme -> IC50_uM)
        - "cyp_substrates" (list of str)

    Returns
    -------
    DDINetworkResult
    """
    if len(drugs) < 2:
        raise ValueError("At least 2 drugs required")

    n = len(drugs)
    # Total pairs = n*(n-1)/2 evaluated bidirectionally:
    # for each ordered pair (i,j) where i != j, check if i inhibits j
    # then take max AUCR per unordered pair {i,j}

    pair_aucrs = {}  # (i_name, j_name) -> max AUCR considering both directions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            drug_i = drugs[i]
            drug_j = drugs[j]
            name_i = drug_i.get("name", f"Drug{i}")
            name_j = drug_j.get("name", f"Drug{j}")

            aucr_ij = _compute_pair_aucr(drug_i, drug_j)

            # Use sorted pair key to aggregate both directions
            pair_key = tuple(sorted([name_i, name_j]))
            current = pair_aucrs.get(pair_key, 1.0)
            if aucr_ij > current:
                pair_aucrs[pair_key] = aucr_ij

    # Count pairs evaluated (unique unordered pairs)
    n_pairs_evaluated = n * (n - 1) // 2

    # Compute statistics
    n_moderate_plus = 0
    n_major_plus = 0
    n_contraindicated = 0
    max_aucr = 1.0
    most_concerning_pair = ""
    weighted_sum = 0.0

    for pair_key, aucr in pair_aucrs.items():
        _severity = _aucr_severity(aucr)

        if aucr >= 1.25:
            n_moderate_plus += 1
            weighted_sum += log2(aucr)

        if aucr >= 2.0:
            n_major_plus += 1

        if aucr >= 5.0:
            n_contraindicated += 1

        if aucr > max_aucr:
            max_aucr = aucr
            most_concerning_pair = f"{pair_key[0]}-{pair_key[1]}"

    # Weighted risk score: sum(log2(AUCR)) * 10, capped at 100
    weighted_risk_score = min(100.0, weighted_sum * 10.0)

    # Polypharmacy risk classification
    if weighted_risk_score < 20.0:
        polypharmacy_risk = "low"
    elif weighted_risk_score < 50.0:
        polypharmacy_risk = "moderate"
    else:
        polypharmacy_risk = "high"

    # Build interaction summary
    if n_moderate_plus == 0:
        interaction_summary = f"No clinically significant DDIs detected among {n} drugs."
    else:
        parts = []
        if n_contraindicated > 0:
            parts.append(f"{n_contraindicated} contraindicated pair(s)")
        if n_major_plus - n_contraindicated > 0:
            parts.append(f"{n_major_plus - n_contraindicated} major pair(s)")
        if n_moderate_plus - n_major_plus > 0:
            parts.append(f"{n_moderate_plus - n_major_plus} moderate pair(s)")
        interaction_summary = (
            f"{n_moderate_plus} significant DDI(s) among {n} drugs: "
            + ", ".join(parts)
            + f". Max AUCR={max_aucr:.2f}."
        )

    notes = (
        "R-value model: AUCR = 1 + Css/IC50. "
        "Severity: >=5 contraindicated, >=2 major, >=1.25 moderate. "
        "Risk score = min(100, sum(log2(AUCR)) * 10) for moderate+ pairs."
    )

    return DDINetworkResult(
        n_drugs=n,
        n_pairs_evaluated=n_pairs_evaluated,
        n_interactions_moderate_plus=n_moderate_plus,
        n_interactions_major_plus=n_major_plus,
        n_contraindicated=n_contraindicated,
        max_aucr=max_aucr,
        most_concerning_pair=most_concerning_pair,
        weighted_risk_score=weighted_risk_score,
        polypharmacy_risk=polypharmacy_risk,
        interaction_summary=interaction_summary,
        notes=notes,
    )


def add_drug_to_regimen(existing_drugs: list, new_drug: dict) -> dict:
    """Add a new drug to an existing regimen and evaluate new interactions.

    Parameters
    ----------
    existing_drugs : list of dict
        Current regimen drugs.
    new_drug : dict
        New drug to add.

    Returns
    -------
    dict with keys:
    - "new_interactions": list of dicts {"with_drug", "aucr", "severity"}
    - "overall_result": DDINetworkResult
    - "recommendation": str
    """
    new_name = new_drug.get("name", "NewDrug")

    # Compute new drug's interactions with each existing drug
    new_interactions = []
    for existing in existing_drugs:
        ex_name = existing.get("name", "ExistingDrug")

        # Check new_drug inhibiting existing substrate
        aucr_new_inh = _compute_pair_aucr(new_drug, existing)
        # Check existing inhibiting new_drug substrate
        aucr_ex_inh = _compute_pair_aucr(existing, new_drug)

        max_aucr = max(aucr_new_inh, aucr_ex_inh)

        if max_aucr >= 1.25:
            severity = _aucr_severity(max_aucr)
            new_interactions.append(
                {
                    "with_drug": ex_name,
                    "aucr": max_aucr,
                    "severity": severity,
                }
            )

    # Full regimen result
    combined = list(existing_drugs) + [new_drug]
    overall_result = score_ddi_network(combined)

    # Recommendation
    if overall_result.polypharmacy_risk == "high":
        recommendation = (
            f"Adding {new_name} creates HIGH polypharmacy risk. "
            "Consider alternative or dose adjustment."
        )
    elif overall_result.polypharmacy_risk == "moderate":
        recommendation = (
            f"Adding {new_name} creates MODERATE polypharmacy risk. "
            "Monitor for DDI-related adverse effects."
        )
    else:
        recommendation = f"Adding {new_name} appears safe. Low DDI risk detected."

    return {
        "new_interactions": new_interactions,
        "overall_result": overall_result,
        "recommendation": recommendation,
    }
