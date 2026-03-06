"""DDI Network Visualizer Data — build interaction graphs from CYP metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DDINetworkResult", "build_ddi_network", "find_critical_path"]


@dataclass(frozen=True)
class DDINetworkResult:
    """Result of DDI network construction."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, str], ...]
    n_nodes: int
    n_edges: int
    high_severity_pairs: tuple[tuple[str, str], ...]
    most_connected_drug: str
    network_complexity_score: float
    notes: str


def _validate_drugs(drugs: Any) -> None:
    """Validate the drugs input list."""
    if not isinstance(drugs, list):
        msg = "drugs must be a list"
        raise ValueError(msg)
    for i, drug in enumerate(drugs):
        if not isinstance(drug, dict) or "name" not in drug:
            msg = f"Drug at index {i} must be a dict with a 'name' key"
            raise ValueError(msg)
        name = drug["name"]
        if not isinstance(name, str) or not name.strip():
            msg = f"Drug at index {i} has an empty or non-string name"
            raise ValueError(msg)


def _count_connections(drug_name: str, edges: list[dict[str, str]]) -> int:
    """Count edges where *drug_name* appears as source or target."""
    return sum(
        1 for e in edges if e["source"] == drug_name or e["target"] == drug_name
    )


def build_ddi_network(
    drugs: list[dict[str, Any]],
    perpetrators: list[dict[str, Any]] | None = None,  # noqa: ARG001
) -> DDINetworkResult:
    """Build a DDI network from a list of drug dictionaries.

    Parameters
    ----------
    drugs:
        Each dict must contain at minimum a ``name`` key.  Optional keys:
        ``cyp_substrates``, ``cyp_inhibitors``, ``cyp_inducers``, ``is_strong``.
    perpetrators:
        Reserved for future use.

    Returns
    -------
    DDINetworkResult
    """
    _validate_drugs(drugs)

    # Build nodes
    nodes: list[dict[str, Any]] = []
    for drug in drugs:
        nodes.append(
            {
                "name": drug["name"],
                "type": "drug",
                "cyp_roles": {
                    "substrates": list(drug.get("cyp_substrates", [])),
                    "inhibitors": list(drug.get("cyp_inhibitors", [])),
                    "inducers": list(drug.get("cyp_inducers", [])),
                },
            }
        )

    # Build edges
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for a in drugs:
        for b in drugs:
            if a["name"] == b["name"]:
                continue

            a_inhibitors = a.get("cyp_inhibitors", [])
            b_substrates = b.get("cyp_substrates", [])
            a_inducers = a.get("cyp_inducers", [])

            # Inhibition edges
            for cyp in a_inhibitors:
                if cyp in b_substrates:
                    key = (a["name"], b["name"], "inhibition")
                    if key not in seen:
                        seen.add(key)
                        severity = (
                            "high" if a.get("is_strong", False) else "moderate"
                        )
                        edges.append(
                            {
                                "source": a["name"],
                                "target": b["name"],
                                "interaction_type": "inhibition",
                                "severity": severity,
                            }
                        )

            # Induction edges
            for cyp in a_inducers:
                if cyp in b_substrates:
                    key = (a["name"], b["name"], "induction")
                    if key not in seen:
                        seen.add(key)
                        edges.append(
                            {
                                "source": a["name"],
                                "target": b["name"],
                                "interaction_type": "induction",
                                "severity": "low",
                            }
                        )

    # Derived fields
    high_severity_pairs = tuple(
        (e["source"], e["target"]) for e in edges if e["severity"] == "high"
    )

    if nodes:
        most_connected = max(
            (n["name"] for n in nodes),
            key=lambda name: _count_connections(name, edges),
        )
    else:
        most_connected = ""

    n_nodes = len(nodes)
    n_edges = len(edges)
    complexity = n_edges / max(1, n_nodes)

    notes_parts: list[str] = []
    if high_severity_pairs:
        notes_parts.append(
            f"{len(high_severity_pairs)} high-severity interaction(s) detected"
        )
    notes = "; ".join(notes_parts) if notes_parts else "No high-severity interactions"

    return DDINetworkResult(
        nodes=tuple(nodes),
        edges=tuple(edges),
        n_nodes=n_nodes,
        n_edges=n_edges,
        high_severity_pairs=high_severity_pairs,
        most_connected_drug=most_connected,
        network_complexity_score=complexity,
        notes=notes,
    )


def find_critical_path(network: DDINetworkResult) -> list[str]:
    """Find the critical path through the DDI network.

    Starting from the most-connected drug, greedily follow edges to the
    neighbour with the most connections that has not yet been visited.

    Returns
    -------
    list[str]
        Ordered drug names along the critical path.
    """
    if network.n_nodes == 0 or not network.most_connected_drug:
        return []

    edges = list(network.edges)
    visited: set[str] = set()
    path: list[str] = []

    current = network.most_connected_drug
    while current and current not in visited:
        visited.add(current)
        path.append(current)

        # Find unvisited neighbours (connected via any edge direction)
        neighbours: set[str] = set()
        for e in edges:
            if e["source"] == current and e["target"] not in visited:
                neighbours.add(e["target"])
            elif e["target"] == current and e["source"] not in visited:
                neighbours.add(e["source"])

        if not neighbours:
            break

        # Pick neighbour with most connections
        current = max(neighbours, key=lambda n: _count_connections(n, edges))

    return path
