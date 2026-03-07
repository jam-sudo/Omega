"""
Phase 318 — Drug Interaction Network Visualizer Data

Build and analyze drug-drug interaction networks for polypharmacy patients.
Identifies high-risk combinations and computes network-level risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDINetworkResult:
    """Result of a DDI network analysis."""

    n_drugs: int
    n_pairs: int
    n_interactions: int
    interaction_density: float
    n_minor: int
    n_moderate: int
    n_major: int
    max_severity: int
    mean_interactions_per_drug: float
    network_risk_score: float
    risk_tier: str
    high_risk_drug: str
    notes: str


@dataclass(frozen=True)
class PolypharmacyRiskResult:
    """Result of polypharmacy risk calculation."""

    n_drugs: int
    network_risk_score: float
    risk_tier: str
    recommendation: str
    notes: str


def _validate_drug_list_and_matrix(
    drug_list: list,
    interaction_matrix: list,
) -> None:
    """Validate drug_list and interaction_matrix."""
    n = len(drug_list)
    if n < 2:
        raise ValueError(f"drug_list must have at least 2 drugs, got {n}")
    if len(interaction_matrix) != n:
        raise ValueError(f"interaction_matrix must have {n} rows, got {len(interaction_matrix)}")
    valid_severities = {0, 1, 2, 3}
    for i, row in enumerate(interaction_matrix):
        if len(row) != n:
            raise ValueError(f"Row {i} of interaction_matrix must have {n} entries, got {len(row)}")
        for j, val in enumerate(row):
            if val not in valid_severities:
                raise ValueError(f"interaction_matrix[{i}][{j}]={val} is not in {{0,1,2,3}}")
    # Check symmetry
    for i in range(n):
        for j in range(n):
            if interaction_matrix[i][j] != interaction_matrix[j][i]:
                raise ValueError(
                    f"interaction_matrix is not symmetric at [{i}][{j}]: "
                    f"{interaction_matrix[i][j]} != {interaction_matrix[j][i]}"
                )


def _compute_network_metrics(
    drug_list: list,
    interaction_matrix: list,
) -> dict:
    """Compute network metrics from drug_list and interaction_matrix."""
    n = len(drug_list)
    n_pairs = n * (n - 1) // 2

    n_minor = 0
    n_moderate = 0
    n_major = 0
    n_interactions = 0
    max_severity = 0

    # Per-drug major interaction counts for identifying high_risk_drug
    major_counts = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            sev = interaction_matrix[i][j]
            if sev > max_severity:
                max_severity = sev
            if sev == 1:
                n_minor += 1
                n_interactions += 1
            elif sev == 2:
                n_moderate += 1
                n_interactions += 1
            elif sev == 3:
                n_major += 1
                n_interactions += 1
                major_counts[i] += 1
                major_counts[j] += 1

    interaction_density = n_interactions / max(1, n_pairs)
    mean_interactions_per_drug = 2 * n_interactions / max(1, n)

    # network_risk_score capped at 100
    raw_score = (n_minor * 1 + n_moderate * 3 + n_major * 5) / max(1, n_pairs) * 100
    network_risk_score = min(100.0, raw_score)

    # risk_tier
    if n_major > 0 and network_risk_score > 30:
        risk_tier = "critical"
    elif n_major > 0:
        risk_tier = "high"
    elif n_moderate > 0:
        risk_tier = "moderate"
    else:
        risk_tier = "low"

    # high_risk_drug: drug with most major interactions
    if n_major > 0:
        max_major = max(major_counts)
        idx = major_counts.index(max_major)
        high_risk_drug = drug_list[idx]
    else:
        high_risk_drug = "none"

    return {
        "n_drugs": n,
        "n_pairs": n_pairs,
        "n_interactions": n_interactions,
        "interaction_density": interaction_density,
        "n_minor": n_minor,
        "n_moderate": n_moderate,
        "n_major": n_major,
        "max_severity": max_severity,
        "mean_interactions_per_drug": mean_interactions_per_drug,
        "network_risk_score": network_risk_score,
        "risk_tier": risk_tier,
        "high_risk_drug": high_risk_drug,
    }


def analyze_ddi_network(
    drug_list: list,
    interaction_matrix: list,
) -> DDINetworkResult:
    """
    Analyze a drug-drug interaction network.

    Parameters
    ----------
    drug_list : list of str
        List of drug names.
    interaction_matrix : list of lists
        n x n matrix where entry [i][j] is severity:
        0=none, 1=minor, 2=moderate, 3=major.
        Must be symmetric with zero diagonal.

    Returns
    -------
    DDINetworkResult
    """
    _validate_drug_list_and_matrix(drug_list, interaction_matrix)
    metrics = _compute_network_metrics(drug_list, interaction_matrix)

    notes = (
        f"DDI network analysis: {metrics['n_drugs']} drugs, "
        f"{metrics['n_interactions']} interactions, "
        f"risk_tier={metrics['risk_tier']}"
    )

    return DDINetworkResult(
        n_drugs=metrics["n_drugs"],
        n_pairs=metrics["n_pairs"],
        n_interactions=metrics["n_interactions"],
        interaction_density=metrics["interaction_density"],
        n_minor=metrics["n_minor"],
        n_moderate=metrics["n_moderate"],
        n_major=metrics["n_major"],
        max_severity=metrics["max_severity"],
        mean_interactions_per_drug=metrics["mean_interactions_per_drug"],
        network_risk_score=metrics["network_risk_score"],
        risk_tier=metrics["risk_tier"],
        high_risk_drug=metrics["high_risk_drug"],
        notes=notes,
    )


def identify_high_risk_pairs(
    drug_list: list,
    interaction_matrix: list,
    severity_threshold: int,
) -> list:
    """
    Identify drug pairs with interactions at or above the severity threshold.

    Parameters
    ----------
    drug_list : list of str
        List of drug names.
    interaction_matrix : list of lists
        n x n severity matrix.
    severity_threshold : int
        Minimum severity to include (1, 2, or 3).

    Returns
    -------
    list of dicts sorted by severity descending. Each dict has:
        - drug_a (str)
        - drug_b (str)
        - severity (int)
        - severity_label (str)
    """
    if severity_threshold not in {1, 2, 3}:
        raise ValueError(f"severity_threshold must be in {{1,2,3}}, got {severity_threshold}")
    _validate_drug_list_and_matrix(drug_list, interaction_matrix)

    severity_labels = {1: "minor", 2: "moderate", 3: "major"}
    pairs = []
    n = len(drug_list)

    for i in range(n):
        for j in range(i + 1, n):
            sev = interaction_matrix[i][j]
            if sev >= severity_threshold:
                pairs.append(
                    {
                        "drug_a": drug_list[i],
                        "drug_b": drug_list[j],
                        "severity": sev,
                        "severity_label": severity_labels[sev],
                    }
                )

    pairs.sort(key=lambda p: p["severity"], reverse=True)
    return pairs


def calculate_polypharmacy_risk(
    drug_list: list,
    interaction_matrix: list,
) -> PolypharmacyRiskResult:
    """
    Calculate overall polypharmacy risk for a drug regimen.

    Parameters
    ----------
    drug_list : list of str
        List of drug names.
    interaction_matrix : list of lists
        n x n severity matrix.

    Returns
    -------
    PolypharmacyRiskResult
    """
    _validate_drug_list_and_matrix(drug_list, interaction_matrix)
    metrics = _compute_network_metrics(drug_list, interaction_matrix)

    risk_tier = metrics["risk_tier"]
    recommendation_map = {
        "critical": "immediate_review",
        "high": "pharmacist_review",
        "moderate": "monitoring",
        "low": "routine",
    }
    recommendation = recommendation_map[risk_tier]

    notes = (
        f"Polypharmacy risk: {metrics['n_drugs']} drugs, "
        f"risk_score={metrics['network_risk_score']:.1f}, "
        f"tier={risk_tier}, recommendation={recommendation}"
    )

    return PolypharmacyRiskResult(
        n_drugs=metrics["n_drugs"],
        network_risk_score=metrics["network_risk_score"],
        risk_tier=risk_tier,
        recommendation=recommendation,
        notes=notes,
    )
