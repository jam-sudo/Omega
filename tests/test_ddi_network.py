"""Tests for the DDI Network Visualizer Data module."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ddi_network import (
    DDINetworkResult,
    build_ddi_network,
    find_critical_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drug(name, substrates=None, inhibitors=None, inducers=None, is_strong=False):
    return {
        "name": name,
        "cyp_substrates": substrates or [],
        "cyp_inhibitors": inhibitors or [],
        "cyp_inducers": inducers or [],
        "is_strong": is_strong,
    }


# ---------------------------------------------------------------------------
# Basic construction tests
# ---------------------------------------------------------------------------

class TestBuildDDINetwork:
    def test_empty_drug_list(self):
        result = build_ddi_network([])
        assert result.n_nodes == 0
        assert result.n_edges == 0
        assert result.most_connected_drug == ""
        assert result.edges == ()

    def test_single_drug_no_edges(self):
        result = build_ddi_network([_drug("DrugA", substrates=["CYP3A4"])])
        assert result.n_nodes == 1
        assert result.n_edges == 0

    def test_two_drugs_no_cyp_overlap(self):
        drugs = [
            _drug("A", substrates=["CYP3A4"]),
            _drug("B", substrates=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        assert result.n_edges == 0

    def test_inhibitor_substrate_creates_edge(self):
        drugs = [
            _drug("Inhibitor", inhibitors=["CYP3A4"]),
            _drug("Substrate", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert result.n_edges == 1
        edge = result.edges[0]
        assert edge["source"] == "Inhibitor"
        assert edge["target"] == "Substrate"
        assert edge["interaction_type"] == "inhibition"
        assert edge["severity"] == "moderate"

    def test_strong_inhibitor_high_severity(self):
        drugs = [
            _drug("StrongInh", inhibitors=["CYP3A4"], is_strong=True),
            _drug("Sub", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert result.n_edges == 1
        assert result.edges[0]["severity"] == "high"

    def test_inducer_substrate_creates_edge(self):
        drugs = [
            _drug("Inducer", inducers=["CYP3A4"]),
            _drug("Sub", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert result.n_edges == 1
        edge = result.edges[0]
        assert edge["interaction_type"] == "induction"
        assert edge["severity"] == "low"

    def test_multiple_cyp_overlaps(self):
        drugs = [
            _drug("A", inhibitors=["CYP3A4", "CYP2D6"]),
            _drug("B", substrates=["CYP3A4"]),
            _drug("C", substrates=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        # A inhibits B (CYP3A4), A inhibits C (CYP2D6)
        assert result.n_edges == 2

    def test_no_duplicate_edges(self):
        # A inhibits both CYP3A4 and CYP2D6, B is substrate of both
        # Should produce only one inhibition edge (A -> B)
        drugs = [
            _drug("A", inhibitors=["CYP3A4", "CYP2D6"]),
            _drug("B", substrates=["CYP3A4", "CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        assert result.n_edges == 1

    def test_high_severity_pairs_populated(self):
        drugs = [
            _drug("Strong", inhibitors=["CYP3A4"], is_strong=True),
            _drug("Victim", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert ("Strong", "Victim") in result.high_severity_pairs

    def test_high_severity_pairs_empty_when_none(self):
        drugs = [
            _drug("Weak", inhibitors=["CYP3A4"]),
            _drug("Victim", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert result.high_severity_pairs == ()

    def test_most_connected_drug(self):
        drugs = [
            _drug("Hub", inhibitors=["CYP3A4", "CYP2D6"]),
            _drug("A", substrates=["CYP3A4"]),
            _drug("B", substrates=["CYP2D6"]),
            _drug("C", substrates=["CYP1A2"]),  # no overlap
        ]
        result = build_ddi_network(drugs)
        assert result.most_connected_drug == "Hub"

    def test_network_complexity_score_non_negative(self):
        result = build_ddi_network([])
        assert result.network_complexity_score >= 0.0

    def test_network_complexity_score_calculation(self):
        drugs = [
            _drug("A", inhibitors=["CYP3A4"]),
            _drug("B", substrates=["CYP3A4"]),
        ]
        result = build_ddi_network(drugs)
        assert result.network_complexity_score == pytest.approx(
            result.n_edges / max(1, result.n_nodes)
        )

    def test_n_nodes_matches_drugs(self):
        drugs = [_drug("X"), _drug("Y"), _drug("Z")]
        result = build_ddi_network(drugs)
        assert result.n_nodes == 3

    def test_n_edges_matches_expected(self):
        drugs = [
            _drug("A", inhibitors=["CYP3A4"]),
            _drug("B", substrates=["CYP3A4"], inhibitors=["CYP2D6"]),
            _drug("C", substrates=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        # A->B (inhibition), B->C (inhibition)
        assert result.n_edges == 2

    def test_bidirectional_interactions(self):
        drugs = [
            _drug("A", substrates=["CYP2D6"], inhibitors=["CYP3A4"]),
            _drug("B", substrates=["CYP3A4"], inhibitors=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        # A inhibits B via CYP3A4, B inhibits A via CYP2D6
        assert result.n_edges == 2
        sources = {e["source"] for e in result.edges}
        assert sources == {"A", "B"}

    def test_mixed_inhibition_and_induction(self):
        drugs = [
            _drug("A", inhibitors=["CYP3A4"], inducers=["CYP2C9"]),
            _drug("B", substrates=["CYP3A4", "CYP2C9"]),
        ]
        result = build_ddi_network(drugs)
        types = {e["interaction_type"] for e in result.edges}
        assert "inhibition" in types
        assert "induction" in types

    def test_notes_is_string(self):
        result = build_ddi_network([_drug("A")])
        assert isinstance(result.notes, str)

    def test_nodes_contain_cyp_roles(self):
        drugs = [_drug("A", substrates=["CYP3A4"], inhibitors=["CYP2D6"])]
        result = build_ddi_network(drugs)
        node = result.nodes[0]
        assert "cyp_roles" in node
        assert node["cyp_roles"]["substrates"] == ["CYP3A4"]
        assert node["cyp_roles"]["inhibitors"] == ["CYP2D6"]

    def test_result_is_frozen_dataclass(self):
        result = build_ddi_network([_drug("A")])
        assert isinstance(result, DDINetworkResult)
        with pytest.raises(AttributeError):
            result.n_nodes = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_input_not_list(self):
        with pytest.raises(ValueError, match="drugs must be a list"):
            build_ddi_network("not a list")  # type: ignore[arg-type]

    def test_drug_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            build_ddi_network([{"cyp_substrates": ["CYP3A4"]}])

    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="empty"):
            build_ddi_network([{"name": ""}])


# ---------------------------------------------------------------------------
# find_critical_path tests
# ---------------------------------------------------------------------------

class TestFindCriticalPath:
    def test_returns_list(self):
        result = build_ddi_network([_drug("A")])
        path = find_critical_path(result)
        assert isinstance(path, list)

    def test_empty_network(self):
        result = build_ddi_network([])
        assert find_critical_path(result) == []

    def test_visits_nodes_once(self):
        drugs = [
            _drug("A", inhibitors=["CYP3A4"]),
            _drug("B", substrates=["CYP3A4"], inhibitors=["CYP2D6"]),
            _drug("C", substrates=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        path = find_critical_path(result)
        assert len(path) == len(set(path))

    def test_critical_path_starts_from_most_connected(self):
        drugs = [
            _drug("Hub", inhibitors=["CYP3A4", "CYP2D6"]),
            _drug("A", substrates=["CYP3A4"]),
            _drug("B", substrates=["CYP2D6"]),
        ]
        result = build_ddi_network(drugs)
        path = find_critical_path(result)
        assert path[0] == result.most_connected_drug
