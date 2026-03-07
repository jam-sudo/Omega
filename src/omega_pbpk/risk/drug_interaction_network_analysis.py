"""Phase 626 — Drug interaction network analysis.

Network analysis of drug interaction graphs: centrality, community detection,
risk propagation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugNode:
    drug_name: str
    degree: int
    betweenness_centrality: float
    is_hub: bool
    interaction_count: int
    avg_aucr: float
    risk_category: str  # "perpetrator","victim","both","neutral"


@dataclass(frozen=True)
class NetworkAnalysisResult:
    n_drugs: int
    n_interactions: int
    drug_nodes: tuple
    hub_drugs: tuple
    most_central_drug: str
    network_density: float
    avg_degree: float
    high_risk_pairs: tuple  # pairs with AUCR > 2.0
    community_assignments: dict
    overall_network_risk: str  # "low","moderate","high"
    notes: str


def compute_degree_centrality(
    drug: str,
    interactions: list,
) -> int:
    """Count interactions involving this drug."""
    count = 0
    for item in interactions:
        drug_a, drug_b = item[0], item[1]
        if drug_a == drug or drug_b == drug:
            count += 1
    return count


def find_hub_drugs(drug_nodes: list, threshold_factor: float = 1.5) -> list:
    """Drugs with degree > threshold_factor * mean_degree."""
    if not drug_nodes:
        return []
    mean_degree = sum(n.degree for n in drug_nodes) / len(drug_nodes)
    threshold = threshold_factor * mean_degree
    return [n.drug_name for n in drug_nodes if n.degree > threshold]


def simple_community_detection(
    drugs: list,
    interactions: list,
) -> dict:
    """Assign drugs to communities using simplified Union-Find approach.

    Start with each drug as its own community, merge if high-AUCR interaction (AUCR > 3).
    Returns dict: {drug_name: community_id}.
    """
    # Build parent mapping (Union-Find)
    parent = {drug: drug for drug in drugs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for item in interactions:
        drug_a, drug_b, aucr = item[0], item[1], item[2]
        if aucr > 3.0:
            union(drug_a, drug_b)

    # Assign community IDs
    root_to_id: dict[str, int] = {}
    community_id_counter = [0]

    result: dict[str, int] = {}
    for drug in drugs:
        root = find(drug)
        if root not in root_to_id:
            root_to_id[root] = community_id_counter[0]
            community_id_counter[0] += 1
        result[drug] = root_to_id[root]

    return result


def build_interaction_network(
    drugs: list,
    interactions: list,
) -> NetworkAnalysisResult:
    """Build and analyze a drug interaction network."""
    # Validation
    if not drugs:
        raise ValueError("drugs list must be non-empty")
    drug_set = set(drugs)
    for item in interactions:
        drug_a, drug_b = item[0], item[1]
        if drug_a not in drug_set:
            raise ValueError(f"Drug '{drug_a}' in interactions not in drugs list")
        if drug_b not in drug_set:
            raise ValueError(f"Drug '{drug_b}' in interactions not in drugs list")

    n_drugs = len(drugs)
    n_interactions = len(interactions)

    # Compute degrees
    degrees = {drug: compute_degree_centrality(drug, interactions) for drug in drugs}

    # Mean degree
    avg_degree = sum(degrees.values()) / n_drugs if n_drugs > 0 else 0.0
    mean_degree = avg_degree

    # Compute betweenness centrality (simplified)
    # For each drug, count pairs (A, C) where this drug is connected to both A and C
    # Normalize by (n-1)*(n-2)/2
    def neighbors_of(drug: str) -> set:
        nbrs = set()
        for item in interactions:
            a, b = item[0], item[1]
            if a == drug:
                nbrs.add(b)
            elif b == drug:
                nbrs.add(a)
        return nbrs

    denom_bc = (n_drugs - 1) * (n_drugs - 2) / 2.0 if n_drugs > 2 else 1.0
    betweenness: dict[str, float] = {}
    for drug in drugs:
        nbrs = neighbors_of(drug)
        count = 0
        other_drugs = [d for d in drugs if d != drug]
        for i, a in enumerate(other_drugs):
            for c in other_drugs[i + 1 :]:
                if a in nbrs and c in nbrs:
                    count += 1
        betweenness[drug] = count / denom_bc if denom_bc > 0 else 0.0

    # Build per-drug AUCR info
    def drug_aucrs(drug: str) -> list:
        vals = []
        for item in interactions:
            if item[0] == drug or item[1] == drug:
                vals.append(item[2])
        return vals

    def risk_category(drug: str) -> str:
        aucrs = drug_aucrs(drug)
        if not aucrs:
            return "neutral"
        avg = sum(aucrs) / len(aucrs)
        # Check if drug appears as drug_b with high AUCR (victim)
        victim_aucrs = [item[2] for item in interactions if item[1] == drug and item[2] > 2.0]
        perp_aucrs = [item[2] for item in interactions if item[0] == drug and item[2] > 2.0]
        if victim_aucrs and perp_aucrs:
            return "both"
        elif avg > 2.0:
            return "perpetrator"
        elif victim_aucrs:
            return "victim"
        else:
            return "neutral"

    # Build DrugNode objects
    drug_nodes_list = []
    for drug in drugs:
        aucrs = drug_aucrs(drug)
        avg_aucr = sum(aucrs) / len(aucrs) if aucrs else 0.0
        deg = degrees[drug]
        is_hub = deg > mean_degree
        node = DrugNode(
            drug_name=drug,
            degree=deg,
            betweenness_centrality=betweenness[drug],
            is_hub=is_hub,
            interaction_count=deg,
            avg_aucr=avg_aucr,
            risk_category=risk_category(drug),
        )
        drug_nodes_list.append(node)

    drug_nodes_tuple = tuple(drug_nodes_list)

    # Hub drugs: degree > mean_degree (consistent with is_hub field)
    hub_drugs_list = find_hub_drugs(drug_nodes_list, threshold_factor=1.0)
    hub_drugs_tuple = tuple(hub_drugs_list)

    # Most central drug by betweenness
    most_central = max(drug_nodes_list, key=lambda n: n.betweenness_centrality).drug_name

    # Network density
    max_possible = n_drugs * (n_drugs - 1) / 2.0 if n_drugs > 1 else 1.0
    density = n_interactions / max_possible if max_possible > 0 else 0.0
    density = max(0.0, min(1.0, density))

    # High risk pairs (AUCR > 2.0)
    high_risk = tuple((item[0], item[1], item[2]) for item in interactions if item[2] > 2.0)

    # Community detection
    community_assignments = simple_community_detection(drugs, interactions)

    # Overall network risk
    all_aucrs = [item[2] for item in interactions]
    if any(a >= 5.0 for a in all_aucrs):
        overall_risk = "high"
    elif any(a > 2.0 for a in all_aucrs):
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    notes_parts = [
        f"{n_drugs} drugs, {n_interactions} interactions",
        f"density={density:.3f}",
        f"avg_degree={avg_degree:.2f}",
        f"overall_risk={overall_risk}",
    ]
    notes = "; ".join(notes_parts)

    return NetworkAnalysisResult(
        n_drugs=n_drugs,
        n_interactions=n_interactions,
        drug_nodes=drug_nodes_tuple,
        hub_drugs=hub_drugs_tuple,
        most_central_drug=most_central,
        network_density=density,
        avg_degree=avg_degree,
        high_risk_pairs=high_risk,
        community_assignments=community_assignments,
        overall_network_risk=overall_risk,
        notes=notes,
    )
