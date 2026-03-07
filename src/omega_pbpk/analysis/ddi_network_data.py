"""DDI network graph data generation for visualization.

Generates nodes and edges representing drug-drug interaction networks based
on CYP substrate/inhibitor/inducer profiles. Suitable for downstream rendering
with graph visualization libraries.

References
----------
- FDA Drug Interaction Studies — Study Design, Data Analysis, and Implications
  for Labeling and Prescribing Information (2012).
- Tornio A, Backman JT. Adv Pharmacol 2018;83:57-82.
"""

from __future__ import annotations

__all__ = [
    "build_ddi_network",
    "identify_hub_perpetrators",
    "identify_vulnerable_victims",
    "compute_network_statistics",
    "filter_clinically_relevant",
]

# Predicted AUCR by interaction strength
_INHIBITION_AUCR: dict[str, float] = {
    "strong": 5.0,
    "moderate": 2.5,
    "weak": 1.5,
}

_INDUCTION_AUCR: dict[str, float] = {
    "strong": 0.2,
    "moderate": 0.4,
    "weak": 0.7,
}


def _drug_role(drug: dict) -> str:
    """Determine the role of a drug in the network."""
    substrates = drug.get("cyp_substrates", [])
    inhibitors = drug.get("cyp_inhibitors", {})
    inducers = drug.get("cyp_inducers", {})
    is_victim = bool(substrates)
    is_perpetrator = bool(inhibitors) or bool(inducers)
    if is_victim and is_perpetrator:
        return "both"
    if is_victim:
        return "victim"
    if is_perpetrator:
        return "perpetrator"
    return "neutral"


def build_ddi_network(drugs: list[dict]) -> dict:
    """Build a DDI network from a list of drug profiles.

    Parameters
    ----------
    drugs:
        List of dicts with keys:
        - name (str)
        - cyp_substrates (list[str]): CYP enzymes this drug is a substrate of
        - cyp_inhibitors (dict[str, str]): enzyme → strength ('weak'/'moderate'/'strong')
        - cyp_inducers (dict[str, str]): enzyme → strength ('weak'/'moderate'/'strong')

    Returns
    -------
    dict: nodes, edges, n_drugs, n_interactions, high_risk_count
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list")

    nodes = []
    for drug in drugs:
        name = drug.get("name", "unknown")
        nodes.append({"id": name, "name": name, "type": _drug_role(drug)})

    edges = []
    n = len(drugs)
    for i in range(n):
        perp = drugs[i]
        perp_name = perp.get("name", "unknown")
        inhibitors = perp.get("cyp_inhibitors", {})
        inducers = perp.get("cyp_inducers", {})

        for j in range(n):
            if i == j:
                continue
            victim = drugs[j]
            victim_name = victim.get("name", "unknown")
            substrates = set(victim.get("cyp_substrates", []))

            # Inhibition edges
            for enzyme, strength in inhibitors.items():
                if enzyme in substrates:
                    predicted_aucr = _INHIBITION_AUCR.get(strength, 1.5)
                    edges.append(
                        {
                            "source": perp_name,
                            "target": victim_name,
                            "mechanism": "inhibition",
                            "enzyme": enzyme,
                            "strength": strength,
                            "predicted_aucr": predicted_aucr,
                        }
                    )

            # Induction edges
            for enzyme, strength in inducers.items():
                if enzyme in substrates:
                    predicted_aucr = _INDUCTION_AUCR.get(strength, 0.7)
                    edges.append(
                        {
                            "source": perp_name,
                            "target": victim_name,
                            "mechanism": "induction",
                            "enzyme": enzyme,
                            "strength": strength,
                            "predicted_aucr": predicted_aucr,
                        }
                    )

    # High-risk: inhibition AUCR >= 5 (strong) or induction AUCR <= 0.2 (strong)
    high_risk_count = sum(
        1
        for e in edges
        if (e["mechanism"] == "inhibition" and e["predicted_aucr"] >= 5.0)
        or (e["mechanism"] == "induction" and e["predicted_aucr"] <= 0.2)
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "n_drugs": len(drugs),
        "n_interactions": len(edges),
        "high_risk_count": high_risk_count,
    }


def identify_hub_perpetrators(network: dict) -> list[dict]:
    """Find drugs that cause the most DDIs (highest out-degree).

    Parameters
    ----------
    network:
        Dict returned by build_ddi_network.

    Returns
    -------
    List of dicts sorted by n_victims descending:
    [{name, n_victims, enzymes_affected}]
    """
    edges = network.get("edges", [])
    nodes = network.get("nodes", [])

    # Count unique victims per source and collect enzymes
    perp_victims: dict[str, set] = {}
    perp_enzymes: dict[str, set] = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        enzyme = edge.get("enzyme", "")
        perp_victims.setdefault(src, set()).add(tgt)
        perp_enzymes.setdefault(src, set()).add(enzyme)

    result = []
    seen = set()
    for node in nodes:
        name = node["name"]
        if name in perp_victims and name not in seen:
            seen.add(name)
            result.append(
                {
                    "name": name,
                    "n_victims": len(perp_victims[name]),
                    "enzymes_affected": sorted(perp_enzymes.get(name, set())),
                }
            )

    result.sort(key=lambda d: d["n_victims"], reverse=True)
    return result


def identify_vulnerable_victims(network: dict) -> list[dict]:
    """Find drugs with the most perpetrators affecting them (highest in-degree).

    Parameters
    ----------
    network:
        Dict returned by build_ddi_network.

    Returns
    -------
    List of dicts sorted by n_perpetrators descending:
    [{name, n_perpetrators, enzymes_affected}]
    """
    edges = network.get("edges", [])
    nodes = network.get("nodes", [])

    victim_perps: dict[str, set] = {}
    victim_enzymes: dict[str, set] = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        enzyme = edge.get("enzyme", "")
        victim_perps.setdefault(tgt, set()).add(src)
        victim_enzymes.setdefault(tgt, set()).add(enzyme)

    result = []
    seen = set()
    for node in nodes:
        name = node["name"]
        if name in victim_perps and name not in seen:
            seen.add(name)
            result.append(
                {
                    "name": name,
                    "n_perpetrators": len(victim_perps[name]),
                    "enzymes_affected": sorted(victim_enzymes.get(name, set())),
                }
            )

    result.sort(key=lambda d: d["n_perpetrators"], reverse=True)
    return result


def compute_network_statistics(network: dict) -> dict:
    """Compute summary statistics for a DDI network.

    Parameters
    ----------
    network:
        Dict returned by build_ddi_network.

    Returns
    -------
    dict: density, avg_degree, n_isolated, max_interactions
    """
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])
    n_nodes = len(nodes)
    n_edges = len(edges)

    if n_nodes <= 1:
        density = 0.0
        avg_degree = 0.0
    else:
        density = n_edges / (n_nodes * (n_nodes - 1))
        avg_degree = 2.0 * n_edges / n_nodes

    # Degree per node (in + out)
    degree: dict[str, int] = {node["name"]: 0 for node in nodes}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src in degree:
            degree[src] += 1
        if tgt in degree:
            degree[tgt] += 1

    n_isolated = sum(1 for d in degree.values() if d == 0)
    max_interactions = max(degree.values()) if degree else 0

    return {
        "density": density,
        "avg_degree": avg_degree,
        "n_isolated": n_isolated,
        "max_interactions": max_interactions,
    }


def filter_clinically_relevant(network: dict, min_aucr: float = 2.0) -> dict:
    """Keep only edges with clinically relevant predicted AUCR.

    For inhibition: keep if predicted_aucr >= min_aucr.
    For induction: keep if predicted_aucr <= 1 / min_aucr.

    Parameters
    ----------
    network:
        Dict returned by build_ddi_network.
    min_aucr:
        Minimum fold-change threshold (default 2.0).

    Returns
    -------
    Filtered network dict.
    """
    if min_aucr <= 0:
        raise ValueError("min_aucr must be positive")

    induction_max = 1.0 / min_aucr
    filtered_edges = []
    for edge in network.get("edges", []):
        aucr = edge["predicted_aucr"]
        mechanism = edge["mechanism"]
        if mechanism == "inhibition" and aucr >= min_aucr:
            filtered_edges.append(edge)
        elif mechanism == "induction" and aucr <= induction_max:
            filtered_edges.append(edge)

    high_risk_count = sum(
        1
        for e in filtered_edges
        if (e["mechanism"] == "inhibition" and e["predicted_aucr"] >= 5.0)
        or (e["mechanism"] == "induction" and e["predicted_aucr"] <= 0.2)
    )

    return {
        "nodes": network.get("nodes", []),
        "edges": filtered_edges,
        "n_drugs": network.get("n_drugs", 0),
        "n_interactions": len(filtered_edges),
        "high_risk_count": high_risk_count,
    }
