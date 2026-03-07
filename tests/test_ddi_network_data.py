"""Tests for analysis/ddi_network_data.py — Phase 548."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.ddi_network_data import (
    build_ddi_network,
    compute_network_statistics,
    filter_clinically_relevant,
    identify_hub_perpetrators,
    identify_vulnerable_victims,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_pair():
    """Inhibitor + substrate pair that should create one edge."""
    drugs = [
        {
            "name": "DrugA",
            "cyp_substrates": [],
            "cyp_inhibitors": {"CYP3A4": "strong"},
            "cyp_inducers": {},
        },
        {
            "name": "DrugB",
            "cyp_substrates": ["CYP3A4"],
            "cyp_inhibitors": {},
            "cyp_inducers": {},
        },
    ]
    return drugs


def _make_network():
    drugs = [
        {
            "name": "Inhibitor",
            "cyp_substrates": [],
            "cyp_inhibitors": {"CYP2C9": "strong", "CYP3A4": "moderate"},
            "cyp_inducers": {},
        },
        {
            "name": "Inducer",
            "cyp_substrates": [],
            "cyp_inhibitors": {},
            "cyp_inducers": {"CYP3A4": "strong"},
        },
        {
            "name": "Victim1",
            "cyp_substrates": ["CYP2C9", "CYP3A4"],
            "cyp_inhibitors": {},
            "cyp_inducers": {},
        },
        {
            "name": "Victim2",
            "cyp_substrates": ["CYP3A4"],
            "cyp_inhibitors": {},
            "cyp_inducers": {},
        },
        {
            "name": "Neutral",
            "cyp_substrates": [],
            "cyp_inhibitors": {},
            "cyp_inducers": {},
        },
    ]
    return drugs


# ---------------------------------------------------------------------------
# build_ddi_network
# ---------------------------------------------------------------------------


class TestBuildDdiNetwork:
    def test_edge_created_for_inhibitor_substrate_overlap(self):
        drugs = _make_simple_pair()
        network = build_ddi_network(drugs)
        assert network["n_interactions"] >= 1
        assert any(e["source"] == "DrugA" and e["target"] == "DrugB" for e in network["edges"])

    def test_no_edge_without_overlap(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {"CYP2C9": "strong"},
                "cyp_inducers": {},
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        assert network["n_interactions"] == 0

    def test_returns_required_keys(self):
        network = build_ddi_network(_make_simple_pair())
        for key in ("nodes", "edges", "n_drugs", "n_interactions", "high_risk_count"):
            assert key in network

    def test_n_drugs_correct(self):
        drugs = _make_network()
        network = build_ddi_network(drugs)
        assert network["n_drugs"] == len(drugs)

    def test_node_types_assigned(self):
        network = build_ddi_network(_make_network())
        node_map = {n["name"]: n["type"] for n in network["nodes"]}
        assert node_map["Inhibitor"] == "perpetrator"
        assert node_map["Victim1"] == "victim"
        assert node_map["Neutral"] == "neutral"

    def test_induction_edge_created(self):
        drugs = [
            {
                "name": "Inducer",
                "cyp_substrates": [],
                "cyp_inhibitors": {},
                "cyp_inducers": {"CYP3A4": "strong"},
            },
            {
                "name": "Substrate",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        induction_edges = [e for e in network["edges"] if e["mechanism"] == "induction"]
        assert len(induction_edges) >= 1

    def test_strong_inhibition_aucr(self):
        network = build_ddi_network(_make_simple_pair())
        edge = network["edges"][0]
        assert edge["predicted_aucr"] == pytest.approx(5.0)

    def test_moderate_inhibition_aucr(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {"CYP3A4": "moderate"},
                "cyp_inducers": {},
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        assert network["edges"][0]["predicted_aucr"] == pytest.approx(2.5)

    def test_strong_induction_aucr(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {},
                "cyp_inducers": {"CYP3A4": "strong"},
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        assert network["edges"][0]["predicted_aucr"] == pytest.approx(0.2)

    def test_high_risk_count(self):
        network = build_ddi_network(_make_simple_pair())
        # strong inhibition → AUCR=5 → high risk
        assert network["high_risk_count"] >= 1

    def test_empty_drug_list(self):
        network = build_ddi_network([])
        assert network["n_drugs"] == 0
        assert network["n_interactions"] == 0
        assert network["nodes"] == []
        assert network["edges"] == []

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            build_ddi_network("not a list")


# ---------------------------------------------------------------------------
# identify_hub_perpetrators
# ---------------------------------------------------------------------------


class TestIdentifyHubPerpetrators:
    def test_drug_with_most_victims_is_first(self):
        drugs = _make_network()
        network = build_ddi_network(drugs)
        hubs = identify_hub_perpetrators(network)
        assert len(hubs) > 0
        # Inhibitor hits Victim1 (CYP2C9, CYP3A4) and Victim2 (CYP3A4)
        assert hubs[0]["name"] == "Inhibitor"

    def test_returns_n_victims(self):
        network = build_ddi_network(_make_network())
        hubs = identify_hub_perpetrators(network)
        for h in hubs:
            assert "n_victims" in h
            assert h["n_victims"] >= 1

    def test_returns_enzymes_affected(self):
        network = build_ddi_network(_make_network())
        hubs = identify_hub_perpetrators(network)
        for h in hubs:
            assert "enzymes_affected" in h
            assert isinstance(h["enzymes_affected"], list)

    def test_sorted_descending(self):
        network = build_ddi_network(_make_network())
        hubs = identify_hub_perpetrators(network)
        victims = [h["n_victims"] for h in hubs]
        assert victims == sorted(victims, reverse=True)

    def test_empty_network(self):
        network = build_ddi_network([])
        hubs = identify_hub_perpetrators(network)
        assert hubs == []


# ---------------------------------------------------------------------------
# identify_vulnerable_victims
# ---------------------------------------------------------------------------


class TestIdentifyVulnerableVictims:
    def test_most_affected_drug_is_first(self):
        drugs = _make_network()
        network = build_ddi_network(drugs)
        victims = identify_vulnerable_victims(network)
        assert len(victims) > 0
        # Victim1 is targeted by both Inhibitor (CYP2C9, CYP3A4) and Inducer (CYP3A4)
        assert victims[0]["name"] == "Victim1"

    def test_returns_n_perpetrators(self):
        network = build_ddi_network(_make_network())
        victims = identify_vulnerable_victims(network)
        for v in victims:
            assert "n_perpetrators" in v

    def test_sorted_descending(self):
        network = build_ddi_network(_make_network())
        victims = identify_vulnerable_victims(network)
        perps = [v["n_perpetrators"] for v in victims]
        assert perps == sorted(perps, reverse=True)

    def test_empty_network(self):
        network = build_ddi_network([])
        assert identify_vulnerable_victims(network) == []


# ---------------------------------------------------------------------------
# compute_network_statistics
# ---------------------------------------------------------------------------


class TestComputeNetworkStatistics:
    def test_density_in_valid_range(self):
        network = build_ddi_network(_make_network())
        stats = compute_network_statistics(network)
        assert 0.0 <= stats["density"] <= 1.0

    def test_returns_required_keys(self):
        network = build_ddi_network(_make_network())
        stats = compute_network_statistics(network)
        for key in ("density", "avg_degree", "n_isolated", "max_interactions"):
            assert key in stats

    def test_n_isolated_neutral_node(self):
        network = build_ddi_network(_make_network())
        stats = compute_network_statistics(network)
        # Neutral drug has no edges
        assert stats["n_isolated"] >= 1

    def test_avg_degree_nonnegative(self):
        network = build_ddi_network(_make_network())
        stats = compute_network_statistics(network)
        assert stats["avg_degree"] >= 0.0

    def test_empty_network_zeros(self):
        network = build_ddi_network([])
        stats = compute_network_statistics(network)
        assert stats["density"] == pytest.approx(0.0)
        assert stats["n_isolated"] == 0


# ---------------------------------------------------------------------------
# filter_clinically_relevant
# ---------------------------------------------------------------------------


class TestFilterClinicallyRelevant:
    def test_removes_weak_inhibition(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {"CYP3A4": "weak"},  # AUCR=1.5 < 2.0
                "cyp_inducers": {},
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        filtered = filter_clinically_relevant(network, min_aucr=2.0)
        assert filtered["n_interactions"] == 0

    def test_keeps_strong_inhibition(self):
        network = build_ddi_network(_make_simple_pair())  # strong → AUCR=5
        filtered = filter_clinically_relevant(network, min_aucr=2.0)
        assert filtered["n_interactions"] >= 1

    def test_removes_weak_induction(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {},
                "cyp_inducers": {"CYP3A4": "weak"},  # AUCR=0.7, 1/2=0.5 → 0.7 > 0.5 → removed
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        filtered = filter_clinically_relevant(network, min_aucr=2.0)
        assert filtered["n_interactions"] == 0

    def test_keeps_strong_induction(self):
        drugs = [
            {
                "name": "A",
                "cyp_substrates": [],
                "cyp_inhibitors": {},
                "cyp_inducers": {"CYP3A4": "strong"},  # AUCR=0.2 <= 0.5
            },
            {
                "name": "B",
                "cyp_substrates": ["CYP3A4"],
                "cyp_inhibitors": {},
                "cyp_inducers": {},
            },
        ]
        network = build_ddi_network(drugs)
        filtered = filter_clinically_relevant(network, min_aucr=2.0)
        assert filtered["n_interactions"] >= 1

    def test_invalid_min_aucr_raises(self):
        network = build_ddi_network([])
        with pytest.raises(ValueError):
            filter_clinically_relevant(network, min_aucr=0.0)

    def test_preserves_nodes(self):
        network = build_ddi_network(_make_network())
        filtered = filter_clinically_relevant(network, min_aucr=2.0)
        assert len(filtered["nodes"]) == len(network["nodes"])
