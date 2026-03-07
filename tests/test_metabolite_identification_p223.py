"""Tests for Phase 223 — metabolite_identification_p223."""

import pytest

from omega_pbpk.prediction.metabolite_identification_p223 import (
    MetaboliteIDResult,
    identify_metabolites,
    predict_metabolic_soft_spots,
)

# ---------------------------------------------------------------------------
# Helper drugs
# ---------------------------------------------------------------------------


def _aromatic_drug():
    return identify_metabolites(
        "AroDrug",
        mw_Da=300.0,
        logP=2.5,
        has_aromatic=True,
        has_aliphatic_ch=False,
        has_amine=False,
        has_ether=False,
        has_sulfur=False,
    )


def _amine_drug():
    return identify_metabolites(
        "AmineDrug",
        mw_Da=350.0,
        logP=1.8,
        has_aromatic=False,
        has_aliphatic_ch=False,
        has_amine=True,
        has_ether=False,
        has_sulfur=False,
    )


def _full_drug():
    """Drug with all functional groups."""
    return identify_metabolites(
        "FullDrug",
        mw_Da=450.0,
        logP=3.0,
        has_aromatic=True,
        has_aliphatic_ch=True,
        has_amine=True,
        has_ether=True,
        has_sulfur=True,
    )


def _heavy_drug():
    """MW >= 500, no sulfation expected."""
    return identify_metabolites(
        "HeavyDrug",
        mw_Da=520.0,
        logP=1.0,
        has_aromatic=False,
        has_aliphatic_ch=False,
    )


# ---------------------------------------------------------------------------
# Return type / structure tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_metabolite_id_result(self):
        result = _aromatic_drug()
        assert isinstance(result, MetaboliteIDResult)

    def test_result_is_frozen(self):
        result = _aromatic_drug()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "x"  # type: ignore[misc]

    def test_metabolite_names_is_tuple(self):
        result = _aromatic_drug()
        assert isinstance(result.metabolite_names, tuple)

    def test_metabolite_mw_Da_is_tuple(self):
        result = _aromatic_drug()
        assert isinstance(result.metabolite_mw_Da, tuple)

    def test_metabolite_pathways_is_tuple(self):
        result = _aromatic_drug()
        assert isinstance(result.metabolite_pathways, tuple)

    def test_metabolite_enzymes_is_tuple(self):
        result = _aromatic_drug()
        assert isinstance(result.metabolite_enzymes, tuple)

    def test_n_metabolites_is_int(self):
        result = _aromatic_drug()
        assert isinstance(result.n_metabolites, int)


# ---------------------------------------------------------------------------
# Field preservation tests
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = _aromatic_drug()
        assert result.drug_name == "AroDrug"

    def test_parent_mw_preserved(self):
        result = identify_metabolites("X", mw_Da=275.5, logP=1.0)
        assert result.parent_mw_Da == pytest.approx(275.5)

    def test_parent_logP_preserved(self):
        result = identify_metabolites("X", mw_Da=300.0, logP=3.14)
        assert result.parent_logP == pytest.approx(3.14)

    def test_n_metabolites_matches_tuple_length(self):
        result = _full_drug()
        assert result.n_metabolites == len(result.metabolite_names)
        assert result.n_metabolites == len(result.metabolite_mw_Da)
        assert result.n_metabolites == len(result.metabolite_pathways)
        assert result.n_metabolites == len(result.metabolite_enzymes)

    def test_primary_pathway_non_empty(self):
        result = _aromatic_drug()
        assert isinstance(result.primary_pathway, str)
        assert len(result.primary_pathway) > 0

    def test_notes_non_empty(self):
        result = _aromatic_drug()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Functional group pathway tests
# ---------------------------------------------------------------------------


class TestPathwayPrediction:
    def test_aromatic_drug_has_aromatic_hydroxylation(self):
        result = _aromatic_drug()
        assert "Aromatic hydroxylation" in result.metabolite_pathways

    def test_aromatic_drug_has_cyp3a4_1a2_enzyme(self):
        result = _aromatic_drug()
        assert any("CYP3A4" in e for e in result.metabolite_enzymes)

    def test_amine_drug_has_n_dealkylation(self):
        result = _amine_drug()
        assert "N-dealkylation" in result.metabolite_pathways

    def test_amine_drug_has_cyp2d6(self):
        result = _amine_drug()
        assert any("CYP2D6" in e for e in result.metabolite_enzymes)

    def test_ether_drug_has_o_dealkylation(self):
        result = identify_metabolites("EtherDrug", 300.0, 2.0, has_ether=True)
        assert "O-dealkylation" in result.metabolite_pathways

    def test_sulfur_drug_has_sulfoxidation(self):
        result = identify_metabolites("SulfDrug", 300.0, 1.0, has_sulfur=True)
        assert "Sulfoxidation" in result.metabolite_pathways

    def test_aliphatic_drug_has_aliphatic_hydroxylation(self):
        result = identify_metabolites("AliphDrug", 300.0, 2.0, has_aliphatic_ch=True)
        assert "Aliphatic hydroxylation" in result.metabolite_pathways

    def test_all_drugs_get_glucuronidation(self):
        for result in [_aromatic_drug(), _amine_drug(), _full_drug(), _heavy_drug()]:
            assert "Glucuronidation" in result.metabolite_pathways

    def test_glucuronide_mw_plus_176(self):
        parent_mw = 300.0
        result = identify_metabolites("Drug", parent_mw, 2.0)
        idx = result.metabolite_pathways.index("Glucuronidation")
        assert result.metabolite_mw_Da[idx] == pytest.approx(parent_mw + 176.0)

    def test_small_mw_drug_gets_sulfation(self):
        result = identify_metabolites("SmallDrug", mw_Da=300.0, logP=1.0)
        assert "Sulfation" in result.metabolite_pathways

    def test_large_mw_drug_no_sulfation(self):
        result = _heavy_drug()  # mw=520
        assert "Sulfation" not in result.metabolite_pathways


# ---------------------------------------------------------------------------
# MW delta tests
# ---------------------------------------------------------------------------


class TestMolecularWeightDeltas:
    def test_oxidative_metabolites_plus_16(self):
        parent_mw = 350.0
        result = identify_metabolites("OxDrug", parent_mw, 2.0, has_aromatic=True)
        idx = result.metabolite_pathways.index("Aromatic hydroxylation")
        assert result.metabolite_mw_Da[idx] == pytest.approx(parent_mw + 16.0)

    def test_n_dealkylation_minus_14(self):
        parent_mw = 350.0
        result = identify_metabolites("AmineDrug", parent_mw, 1.5, has_amine=True)
        idx = result.metabolite_pathways.index("N-dealkylation")
        assert result.metabolite_mw_Da[idx] == pytest.approx(parent_mw - 14.0)

    def test_sulfate_plus_80(self):
        parent_mw = 300.0
        result = identify_metabolites("Drug", parent_mw, 1.0)
        idx = result.metabolite_pathways.index("Sulfation")
        assert result.metabolite_mw_Da[idx] == pytest.approx(parent_mw + 80.0)


# ---------------------------------------------------------------------------
# n_metabolites tests
# ---------------------------------------------------------------------------


class TestNMetabolites:
    def test_n_metabolites_greater_than_zero(self):
        result = identify_metabolites("BasicDrug", mw_Da=200.0, logP=1.0)
        assert result.n_metabolites > 0

    def test_full_drug_has_many_metabolites(self):
        result = _full_drug()
        # aromatic_OH, aliphatic_OH, N_dealkyl, O_dealkyl, sulfoxide, glucuronide, sulfate
        assert result.n_metabolites >= 7

    def test_no_functional_groups_still_has_metabolites(self):
        result = identify_metabolites("Bare", 300.0, 1.0)
        assert result.n_metabolites >= 2  # at least glucuronidation + sulfation


# ---------------------------------------------------------------------------
# Primary pathway tests
# ---------------------------------------------------------------------------


class TestPrimaryPathway:
    def test_aromatic_primary_pathway(self):
        result = _aromatic_drug()
        assert result.primary_pathway == "Aromatic hydroxylation"

    def test_amine_primary_when_no_aromatic(self):
        result = _amine_drug()
        assert result.primary_pathway == "N-dealkylation"

    def test_glucuronidation_primary_when_no_cyp_groups(self):
        result = identify_metabolites("Clean", 300.0, 1.0)
        assert result.primary_pathway == "Glucuronidation"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            identify_metabolites("Drug", mw_Da=0.0, logP=1.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            identify_metabolites("Drug", mw_Da=-100.0, logP=1.0)


# ---------------------------------------------------------------------------
# predict_metabolic_soft_spots tests
# ---------------------------------------------------------------------------


class TestSoftSpots:
    def test_returns_list(self):
        spots = predict_metabolic_soft_spots(
            "Drug",
            300.0,
            2.0,
            {"has_aromatic": True, "has_amine": True},
        )
        assert isinstance(spots, list)

    def test_sorted_by_vulnerability_descending(self):
        spots = predict_metabolic_soft_spots(
            "Drug",
            300.0,
            2.0,
            {"has_aromatic": True, "has_amine": True, "has_sulfur": True},
        )
        scores = [s["vulnerability_score"] for s in spots]
        assert scores == sorted(scores, reverse=True)

    def test_each_spot_has_required_keys(self):
        spots = predict_metabolic_soft_spots(
            "Drug",
            300.0,
            2.0,
            {"has_aromatic": True},
        )
        for spot in spots:
            assert "site" in spot
            assert "pathway" in spot
            assert "enzyme" in spot
            assert "vulnerability_score" in spot

    def test_glucuronidation_always_present(self):
        spots = predict_metabolic_soft_spots("Drug", 300.0, 1.0, {})
        pathways = [s["pathway"] for s in spots]
        assert "Glucuronidation" in pathways

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_metabolic_soft_spots("Drug", -1.0, 1.0, {})
