"""Phase 843 — Drug Interaction Network Analysis.

Analyze multi-drug interaction networks for polypharmacy risk.
Uses pure Python stdlib only (math, dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SEVERITY_ORDER: dict[str, int] = {"minor": 0, "moderate": 1, "major": 2}


def _update_worst(current: str | None, candidate: str) -> str:
    """Return the more severe of current and candidate severity strings."""
    if current is None:
        return candidate
    return candidate if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current] else current


@dataclass(frozen=True)
class DrugInteractionNetworkResult:
    drugs: str  # comma-separated drug names
    n_drugs: int
    n_interactions: int  # number of pairwise interactions detected
    n_major: int  # major severity interactions
    n_moderate: int
    n_minor: int
    polypharmacy_risk_score: float  # 0-100
    risk_level: str  # "low", "moderate", "high", "critical"
    critical_pairs: str  # "DrugA-DrugB; DrugC-DrugD" (major interactions)
    highest_risk_drug: str  # drug involved in most interactions
    cyp_conflicts: str  # "CYP3A4: inhibitor DrugA vs substrate DrugB"
    recommended_monitoring: str
    notes: str


def _classify_risk_level(score: float) -> str:
    if score > 75:
        return "critical"
    if score > 50:
        return "high"
    if score > 25:
        return "moderate"
    return "low"


def _recommended_monitoring(risk_level: str, n_major: int) -> str:
    if risk_level == "critical":
        return "Immediate clinical review required; consider regimen change"
    if risk_level == "high":
        return "Close monitoring recommended; check therapeutic drug levels"
    if n_major > 0:
        return "Monitor for adverse effects; review dosing for narrow TI drugs"
    if risk_level == "moderate":
        return "Routine monitoring; patient counseling advised"
    return "Standard monitoring sufficient"


def analyze_drug_interactions(
    drug_list: list[dict[str, Any]],
) -> DrugInteractionNetworkResult:
    """Analyze pairwise drug interactions in a multi-drug regimen.

    Parameters
    ----------
    drug_list:
        Each element is a dict with keys:
        - "name": str
        - "cyp_inhibitors": list[str]
        - "cyp_substrates": list[str]
        - "cyp_inducers": list[str]
        - "narrow_therapeutic_index": bool
        - "qt_risk": bool

    Returns
    -------
    DrugInteractionNetworkResult
    """
    if not drug_list or len(drug_list) < 2:
        raise ValueError("drug_list must contain at least 2 drugs")

    n_drugs = len(drug_list)
    drug_names = [d["name"] for d in drug_list]

    # Interaction counters per pair (keep only the worst severity per pair)
    pair_severity: dict[tuple[int, int], str] = {}
    cyp_conflict_list: list[str] = []

    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            drug_i = drug_list[i]
            drug_j = drug_list[j]

            worst: str | None = None  # None < "minor" < "moderate" < "major"

            # CYP inhibition: drug_i inhibits → drug_j is substrate
            for enzyme in drug_i.get("cyp_inhibitors", []):
                if enzyme in drug_j.get("cyp_substrates", []):
                    severity = (
                        "major" if drug_j.get("narrow_therapeutic_index", False) else "moderate"
                    )
                    worst = _update_worst(worst, severity)
                    conflict_str = (
                        f"{enzyme}: inhibitor {drug_i['name']} vs substrate {drug_j['name']}"
                    )
                    if conflict_str not in cyp_conflict_list:
                        cyp_conflict_list.append(conflict_str)

            # CYP inhibition: drug_j inhibits → drug_i is substrate
            for enzyme in drug_j.get("cyp_inhibitors", []):
                if enzyme in drug_i.get("cyp_substrates", []):
                    severity = (
                        "major" if drug_i.get("narrow_therapeutic_index", False) else "moderate"
                    )
                    worst = _update_worst(worst, severity)
                    conflict_str = (
                        f"{enzyme}: inhibitor {drug_j['name']} vs substrate {drug_i['name']}"
                    )
                    if conflict_str not in cyp_conflict_list:
                        cyp_conflict_list.append(conflict_str)

            # CYP induction: drug_i induces → drug_j is substrate
            for enzyme in drug_i.get("cyp_inducers", []):
                if enzyme in drug_j.get("cyp_substrates", []):
                    worst = _update_worst(worst, "moderate")

            # CYP induction: drug_j induces → drug_i is substrate
            for enzyme in drug_j.get("cyp_inducers", []):
                if enzyme in drug_i.get("cyp_substrates", []):
                    worst = _update_worst(worst, "moderate")

            # QT additive risk
            if drug_i.get("qt_risk", False) and drug_j.get("qt_risk", False):
                worst = _update_worst(worst, "major")

            if worst is not None:
                pair_severity[(i, j)] = worst

    # Tally totals
    n_interactions = len(pair_severity)
    n_major = sum(1 for s in pair_severity.values() if s == "major")
    n_moderate = sum(1 for s in pair_severity.values() if s == "moderate")
    n_minor = sum(1 for s in pair_severity.values() if s == "minor")

    # Polypharmacy risk score
    raw_score = n_drugs * 5 + n_major * 15 + n_moderate * 8 + n_minor * 3
    polypharmacy_risk_score = float(max(0.0, min(100.0, raw_score)))

    risk_level = _classify_risk_level(polypharmacy_risk_score)

    # Critical pairs (major interactions)
    critical_pair_strs: list[str] = []
    for (i, j), sev in pair_severity.items():
        if sev == "major":
            critical_pair_strs.append(f"{drug_names[i]}-{drug_names[j]}")
    critical_pairs = "; ".join(critical_pair_strs) if critical_pair_strs else "none"

    # Highest-risk drug (involved in most interactions)
    interaction_count: dict[str, int] = {name: 0 for name in drug_names}
    for i, j in pair_severity:
        interaction_count[drug_names[i]] += 1
        interaction_count[drug_names[j]] += 1

    highest_risk_drug = max(interaction_count, key=lambda k: interaction_count[k])

    # CYP conflicts string
    cyp_conflicts = "; ".join(cyp_conflict_list) if cyp_conflict_list else "none"

    recommended_monitoring = _recommended_monitoring(risk_level, n_major)

    notes_parts: list[str] = []
    if n_drugs >= 5:
        notes_parts.append("High polypharmacy burden (>=5 drugs)")
    if n_major > 0:
        notes_parts.append(f"{n_major} major interaction(s) require urgent review")
    if not notes_parts:
        notes_parts.append("No critical flags identified")
    notes = "; ".join(notes_parts)

    return DrugInteractionNetworkResult(
        drugs=", ".join(drug_names),
        n_drugs=n_drugs,
        n_interactions=n_interactions,
        n_major=n_major,
        n_moderate=n_moderate,
        n_minor=n_minor,
        polypharmacy_risk_score=polypharmacy_risk_score,
        risk_level=risk_level,
        critical_pairs=critical_pairs,
        highest_risk_drug=highest_risk_drug,
        cyp_conflicts=cyp_conflicts,
        recommended_monitoring=recommended_monitoring,
        notes=notes,
    )


def add_drug_to_regimen(
    existing_drugs: list[dict[str, Any]],
    new_drug: dict[str, Any],
) -> DrugInteractionNetworkResult:
    """Append new_drug to existing list and re-run analysis.

    Parameters
    ----------
    existing_drugs:
        List of drug dicts already in the regimen (must have >= 1 entry).
    new_drug:
        Drug dict to add.

    Returns
    -------
    DrugInteractionNetworkResult for the combined regimen.
    """
    combined = list(existing_drugs) + [new_drug]
    return analyze_drug_interactions(combined)
