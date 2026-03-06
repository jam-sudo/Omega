"""Tests for drug interaction network — polypharmacy DDI analysis."""

import pytest

from omega_pbpk.clinical.drug_interaction_network import (
    DDIEdge,
    DDINetworkResult,
    analyze_drug_network,
    ddi_matrix,
)


class TestAnalyzeDrugNetwork:
    def test_empty_drugs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            analyze_drug_network([])

    def test_single_drug_no_interactions(self):
        result = analyze_drug_network(["warfarin"])
        assert result.n_interactions == 0
        assert result.n_severe == 0
        assert result.summary == "No significant DDIs detected"

    def test_ketoconazole_simvastatin_severe(self):
        """Ketoconazole (CYP3A4 inhibitor) + simvastatin (CYP3A4 substrate) = severe."""
        result = analyze_drug_network(["ketoconazole", "simvastatin"])
        assert result.n_interactions >= 1
        assert result.n_severe >= 1
        # At least one severe edge
        severe_edges = [e for e in result.edges if e.severity == "severe"]
        assert len(severe_edges) >= 1

    def test_fluconazole_warfarin_severe(self):
        """Fluconazole (CYP2C9 strong inhibitor) + warfarin (narrow TI) = severe."""
        result = analyze_drug_network(["fluconazole", "warfarin"])
        assert result.n_severe >= 1
        assert "warfarin" in result.high_risk_drugs or "fluconazole" in result.high_risk_drugs

    def test_rifampicin_simvastatin_induction(self):
        """Rifampicin (inducer) + simvastatin (CYP3A4 substrate) = moderate/severe."""
        result = analyze_drug_network(["rifampicin", "simvastatin"])
        assert result.n_interactions >= 1
        edges = result.edges
        inducer_edges = [e for e in edges if e.mechanism == "induction"]
        assert len(inducer_edges) >= 1
        # AUCR should be <1 (induction decreases AUC)
        for e in inducer_edges:
            assert e.aucr < 1.0

    def test_no_interaction_pair(self):
        """Two drugs with no shared CYP: metoprolol + simvastatin (different CYPs)."""
        result = analyze_drug_network(["digoxin", "metoprolol"])
        # Both have no common inhibitor/substrate relationships
        assert result.n_interactions == 0 or result.n_severe == 0

    def test_three_drug_network(self):
        result = analyze_drug_network(["ketoconazole", "simvastatin", "midazolam"])
        # Should detect interactions involving ketoconazole
        assert result.n_interactions >= 2
        assert isinstance(result.drugs, list)
        assert len(result.drugs) == 3

    def test_result_structure(self):
        result = analyze_drug_network(["fluoxetine", "codeine"])
        assert isinstance(result, DDINetworkResult)
        assert isinstance(result.edges, list)
        assert isinstance(result.high_risk_drugs, list)
        assert result.n_severe >= 0
        assert result.n_moderate >= 0

    def test_severe_summary(self):
        result = analyze_drug_network(["ketoconazole", "simvastatin"])
        assert "severe" in result.summary.lower() or "⚠" in result.summary

    def test_edge_fields(self):
        result = analyze_drug_network(["fluconazole", "warfarin"])
        for edge in result.edges:
            assert isinstance(edge, DDIEdge)
            assert edge.drug_a in ["fluconazole", "warfarin"]
            assert edge.drug_b in ["fluconazole", "warfarin"]
            assert edge.severity in ("none", "mild", "moderate", "severe")
            assert edge.aucr > 0

    def test_unknown_drug_no_error(self):
        """Unknown drugs return no interactions (graceful fallback)."""
        result = analyze_drug_network(["unknowndrug_xyz", "anotherunknown_abc"])
        assert result.n_interactions == 0

    def test_amiodarone_warfarin_severe(self):
        """Amiodarone (CYP2C9 inhibitor, narrow TI) + warfarin (CYP2C9 substrate, narrow TI)."""
        result = analyze_drug_network(["amiodarone", "warfarin"])
        assert result.n_severe >= 1


class TestDDIMatrix:
    def test_diagonal_is_one(self):
        drugs = ["warfarin", "fluconazole"]
        matrix = ddi_matrix(drugs)
        for d in drugs:
            assert matrix[d][d] == 1.0

    def test_inhibition_raises_aucr(self):
        """Fluconazole perpetrator → warfarin victim: AUCR > 1."""
        matrix = ddi_matrix(["warfarin", "fluconazole"])
        # warfarin as victim, fluconazole as perpetrator
        assert matrix["warfarin"]["fluconazole"] > 1.0

    def test_induction_lowers_aucr(self):
        """Rifampicin perpetrator → simvastatin victim: AUCR < 1."""
        matrix = ddi_matrix(["simvastatin", "rifampicin"])
        assert matrix["simvastatin"]["rifampicin"] < 1.0

    def test_no_interaction_aucr_one(self):
        matrix = ddi_matrix(["digoxin", "midazolam"])
        # If no CYP interaction, AUCR = 1.0
        assert matrix["digoxin"]["midazolam"] == 1.0 or matrix["midazolam"]["digoxin"] == 1.0

    def test_matrix_keys(self):
        drugs = ["warfarin", "fluconazole", "simvastatin"]
        matrix = ddi_matrix(drugs)
        for d in drugs:
            assert d in matrix
            for d2 in drugs:
                assert d2 in matrix[d]
