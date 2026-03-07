"""Tests for risk/drug_interaction_network_analysis.py — Phase 626."""

from __future__ import annotations

from omega_pbpk.risk.drug_interaction_network_analysis import (
    NetworkAnalysisResult,
    build_interaction_network,
    compute_degree_centrality,
    find_hub_drugs,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _net() -> NetworkAnalysisResult:
    drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    interactions = [
        ("DrugA", "DrugB", 5.0, "cyp_inhibition"),
        ("DrugA", "DrugC", 2.5, "pgp"),
        ("DrugB", "DrugD", 1.2, "ppb"),
    ]
    return build_interaction_network(drugs, interactions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_result():
    assert isinstance(_net(), NetworkAnalysisResult)


def test_n_drugs():
    assert _net().n_drugs == 4


def test_n_interactions():
    assert _net().n_interactions == 3


def test_drug_nodes_length():
    assert len(_net().drug_nodes) == 4


def test_drug_a_high_degree():
    net = _net()
    node_a = next(n for n in net.drug_nodes if n.drug_name == "DrugA")
    assert node_a.degree == 2


def test_drug_d_low_degree():
    net = _net()
    node_d = next(n for n in net.drug_nodes if n.drug_name == "DrugD")
    assert node_d.degree == 1


def test_network_density_in_0_1():
    net = _net()
    assert 0.0 <= net.network_density <= 1.0


def test_avg_degree_positive():
    assert _net().avg_degree > 0.0


def test_high_risk_pairs_count():
    # AUCR > 2.0: DrugA-DrugB (5.0), DrugA-DrugC (2.5) → 2 pairs
    net = _net()
    assert len(net.high_risk_pairs) == 2


def test_hub_drugs_nonempty():
    # DrugA has degree 2 which is above average
    net = _net()
    assert len(net.hub_drugs) > 0


def test_most_central_drug_nonempty():
    assert len(_net().most_central_drug) > 0


def test_overall_risk_high():
    # has aucr=5.0 → "high"
    assert _net().overall_network_risk == "high"


def test_community_assignments_dict():
    assert isinstance(_net().community_assignments, dict)


def test_community_all_drugs_assigned():
    net = _net()
    drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    for drug in drugs:
        assert drug in net.community_assignments


def test_community_high_aucr_same_community():
    # DrugA and DrugB have AUCR=5.0 > 3 → same community
    net = _net()
    assert net.community_assignments["DrugA"] == net.community_assignments["DrugB"]


def test_compute_degree_for_drug_a():
    interactions = [
        ("DrugA", "DrugB", 5.0, "cyp_inhibition"),
        ("DrugA", "DrugC", 2.5, "pgp"),
        ("DrugB", "DrugD", 1.2, "ppb"),
    ]
    assert compute_degree_centrality("DrugA", interactions) == 2


def test_find_hub_drugs_returns_list():
    drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    interactions = [
        ("DrugA", "DrugB", 5.0, "cyp_inhibition"),
        ("DrugA", "DrugC", 2.5, "pgp"),
        ("DrugB", "DrugD", 1.2, "ppb"),
    ]
    net = build_interaction_network(drugs, interactions)
    result = find_hub_drugs(list(net.drug_nodes))
    assert isinstance(result, list)


def test_drug_risk_category_valid():
    net = _net()
    valid_categories = {"perpetrator", "victim", "both", "neutral"}
    for node in net.drug_nodes:
        assert node.risk_category in valid_categories


def test_betweenness_nonneg():
    net = _net()
    for node in net.drug_nodes:
        assert node.betweenness_centrality >= 0.0


def test_is_hub_for_high_degree_drug():
    # DrugA has the highest degree; check is_hub field
    net = _net()
    node_a = next(n for n in net.drug_nodes if n.drug_name == "DrugA")
    # DrugA has degree 2, mean is (2+2+1+1)/4=1.5 → is_hub if degree > 1.5
    assert node_a.is_hub is True


def test_notes_nonempty():
    assert len(_net().notes) > 0


def test_empty_interactions_zero_density():
    drugs = ["DrugA", "DrugB"]
    interactions: list = []
    net = build_interaction_network(drugs, interactions)
    assert net.network_density == 0.0
