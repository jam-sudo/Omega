"""Tests for omega_pbpk.prediction.metabolite_identification module."""

import pytest

from omega_pbpk.prediction.metabolite_identification import (
    MetabolitePrediction,
    MetaboliteResult,
    predict_metabolites,
    rank_metabolic_pathways,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG_NAME = "ibuprofen"
SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
MW = 206.28
LOGP = 3.97


# ---------------------------------------------------------------------------
# predict_metabolites — return type and basic structure
# ---------------------------------------------------------------------------


class TestPredictMetabolitesReturnType:
    def test_returns_metabolite_result(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert isinstance(r, MetaboliteResult)

    def test_compound_name_preserved(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert r.compound_name == DRUG_NAME

    def test_smiles_preserved(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert r.smiles == SMILES

    def test_mw_preserved(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert r.mw == pytest.approx(MW)

    def test_logp_preserved(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert r.logP == pytest.approx(LOGP)

    def test_predicted_metabolites_is_list(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["amine"])
        assert isinstance(r.predicted_metabolites, list)

    def test_n_pathways_matches_len_metabolites(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic", "amine"])
        assert r.n_pathways == len(r.predicted_metabolites)

    def test_notes_is_list(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, [])
        assert isinstance(r.notes, list)

    def test_reactive_risk_is_valid_level(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert r.reactive_metabolite_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Pathway generation for specific functional groups
# ---------------------------------------------------------------------------


class TestPathwayGeneration:
    def test_aromatic_generates_hydroxylation(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "aromatic_hydroxylation" in types

    def test_amine_generates_n_dealkylation(self):
        r = predict_metabolites("amine_drug", "CN", 45.0, -0.5, ["amine"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "N-dealkylation" in types

    def test_amine_generates_n_oxide(self):
        r = predict_metabolites("amine_drug", "CN", 45.0, -0.5, ["amine"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "N-oxide" in types

    def test_alcohol_generates_glucuronide(self):
        r = predict_metabolites("alcohol_drug", "CCO", 46.0, -0.1, ["alcohol"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "glucuronide" in types

    def test_ester_generates_hydrolysis(self):
        r = predict_metabolites("ester_drug", "CC(=O)OC", 74.0, 0.7, ["ester"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "alcohol_acid" in types

    def test_sulfide_generates_sulfoxide(self):
        r = predict_metabolites("sulfide_drug", "CSCC", 90.0, 1.2, ["sulfide"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "sulfoxide" in types

    def test_alkene_generates_epoxide(self):
        r = predict_metabolites("alkene_drug", "C=CC", 42.0, 0.9, ["alkene"])
        types = [m.metabolite_type for m in r.predicted_metabolites]
        assert "epoxide" in types

    def test_no_functional_groups_zero_pathways(self):
        r = predict_metabolites("bare_drug", "C", 16.0, 0.1, [])
        assert r.n_pathways == 0

    def test_multiple_groups_more_pathways(self):
        single = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        multi = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic", "amine", "alcohol"])
        assert multi.n_pathways > single.n_pathways


# ---------------------------------------------------------------------------
# Reactive metabolite risk
# ---------------------------------------------------------------------------


class TestReactiveMetaboliteRisk:
    def test_no_alerts_low_risk(self):
        r = predict_metabolites("amide_drug", "CC(=O)N", 59.0, -1.0, ["amide"])
        assert r.reactive_metabolite_risk == "low"

    def test_aromatic_moderate_risk(self):
        r = predict_metabolites("aromatic_drug", "c1ccccc1", 78.0, 2.1, ["aromatic"])
        assert r.reactive_metabolite_risk in {"moderate", "high"}

    def test_nitro_high_risk(self):
        r = predict_metabolites("nitro_drug", "c1ccc([N+](=O)[O-])cc1", 123.0, 1.8, ["nitro"])
        assert r.reactive_metabolite_risk == "high"

    def test_aldehyde_high_risk(self):
        r = predict_metabolites("aldehyde_drug", "CC=O", 44.0, 0.5, ["aldehyde"])
        assert r.reactive_metabolite_risk == "high"


# ---------------------------------------------------------------------------
# Prediction content
# ---------------------------------------------------------------------------


class TestPredictionContent:
    def test_prediction_has_metabolite_type(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        for m in r.predicted_metabolites:
            assert isinstance(m.metabolite_type, str) and m.metabolite_type

    def test_prediction_has_reaction(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        for m in r.predicted_metabolites:
            assert isinstance(m.reaction, str) and m.reaction

    def test_prediction_has_enzyme(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        for m in r.predicted_metabolites:
            assert isinstance(m.enzyme, str) and m.enzyme

    def test_prediction_likelihood_valid(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["aromatic", "amine"])
        for m in r.predicted_metabolites:
            assert m.likelihood in {"high", "moderate", "low"}

    def test_prediction_objects_are_metabolite_prediction(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["sulfide"])
        for m in r.predicted_metabolites:
            assert isinstance(m, MetabolitePrediction)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_raises_empty_compound_name(self):
        with pytest.raises(ValueError):
            predict_metabolites("", SMILES, MW, LOGP, [])

    def test_raises_empty_smiles(self):
        with pytest.raises(ValueError):
            predict_metabolites(DRUG_NAME, "", MW, LOGP, [])

    def test_raises_zero_mw(self):
        with pytest.raises(ValueError):
            predict_metabolites(DRUG_NAME, SMILES, 0.0, LOGP, [])

    def test_raises_negative_mw(self):
        with pytest.raises(ValueError):
            predict_metabolites(DRUG_NAME, SMILES, -10.0, LOGP, [])

    def test_raises_invalid_functional_group(self):
        with pytest.raises(ValueError, match="functional group"):
            predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, ["quaternary_amine"])

    def test_raises_functional_groups_not_list(self):
        with pytest.raises((ValueError, TypeError)):
            predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, "aromatic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rank_metabolic_pathways
# ---------------------------------------------------------------------------


class TestRankMetabolicPathways:
    def test_returns_list(self):
        result = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        assert isinstance(result, list)

    def test_each_entry_is_dict(self):
        result = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        for entry in result:
            assert isinstance(entry, dict)

    def test_dict_has_required_keys(self):
        result = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, ["aromatic"])
        required_keys = {"metabolite_type", "reaction", "enzyme", "likelihood"}
        for entry in result:
            assert required_keys.issubset(entry.keys())

    def test_ranked_high_before_moderate(self):
        result = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, ["aromatic", "amine"])
        likelihoods = [e["likelihood"] for e in result]
        order = {"high": 0, "moderate": 1, "low": 2}
        ranks = [order[lk] for lk in likelihoods]
        assert ranks == sorted(ranks)

    def test_no_functional_groups_empty_list(self):
        result = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, [])
        assert result == []

    def test_rank_raises_on_invalid_group(self):
        with pytest.raises(ValueError):
            rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, ["invalid_group"])

    def test_rank_same_count_as_predict(self):
        groups = ["aromatic", "amine", "sulfide"]
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, groups)
        ranked = rank_metabolic_pathways(DRUG_NAME, SMILES, MW, LOGP, groups)
        assert len(ranked) == r.n_pathways


# ---------------------------------------------------------------------------
# Notes content
# ---------------------------------------------------------------------------


class TestNotes:
    def test_empty_groups_note(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, LOGP, [])
        assert any("No functional groups" in n for n in r.notes)

    def test_high_mw_note(self):
        r = predict_metabolites("big_drug", SMILES, 600.0, LOGP, [])
        assert any("MW" in n or "500" in n for n in r.notes)

    def test_high_logp_note(self):
        r = predict_metabolites(DRUG_NAME, SMILES, MW, 6.0, [])
        assert any("logP" in n or "5" in n for n in r.notes)

    def test_reactive_risk_note_for_nitro(self):
        r = predict_metabolites("nitro_drug", "c1ccc([N+](=O)[O-])cc1", 123.0, 1.8, ["nitro"])
        assert any("reactive" in n.lower() for n in r.notes)
