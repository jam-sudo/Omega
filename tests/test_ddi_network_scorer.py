"""Tests for Phase 874: DDI Network Scorer."""

import pytest

from omega_pbpk.risk.ddi_network_scorer import (
    DDINetworkResult,
    add_drug_to_regimen,
    score_ddi_network,
)

# --- Helpers ---


def _drug(name, css_uM=1.0, inhibitions=None, substrates=None):
    return {
        "name": name,
        "css_uM": css_uM,
        "cyp_inhibitions": inhibitions or {},
        "cyp_substrates": substrates or [],
    }


# --- Basic type and fields ---


def test_returns_correct_type():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert isinstance(result, DDINetworkResult)


def test_n_drugs_field():
    drugs = [_drug("A"), _drug("B"), _drug("C")]
    result = score_ddi_network(drugs)
    assert result.n_drugs == 3


def test_n_pairs_evaluated_two_drugs():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert result.n_pairs_evaluated == 1


def test_n_pairs_evaluated_four_drugs():
    drugs = [_drug("A"), _drug("B"), _drug("C"), _drug("D")]
    result = score_ddi_network(drugs)
    assert result.n_pairs_evaluated == 6


# --- No interaction case ---


def test_no_interaction_two_drugs():
    """Two drugs with no shared CYP interactions."""
    drugs = [
        _drug("A", substrates=["CYP3A4"]),
        _drug("B", substrates=["CYP2D6"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_interactions_moderate_plus == 0


def test_no_interaction_max_aucr_is_1():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert result.max_aucr == 1.0


def test_no_interaction_low_risk():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert result.polypharmacy_risk == "low"


# --- Interaction cases ---


def test_inhibitor_substrate_pair_aucr_gt_1():
    """Drug A inhibits CYP3A4, Drug B is a CYP3A4 substrate."""
    drugs = [
        _drug("Inhibitor", css_uM=10.0, inhibitions={"CYP3A4": 1.0}),
        _drug("Substrate", substrates=["CYP3A4"]),
    ]
    result = score_ddi_network(drugs)
    assert result.max_aucr > 1.0


def test_strong_inhibitor_major_or_contraindicated():
    """IC50=0.1 uM, Css=10 uM → AUCR = 101 (contraindicated)."""
    drugs = [
        _drug("StrongInh", css_uM=10.0, inhibitions={"CYP3A4": 0.1}),
        _drug("Substrate", substrates=["CYP3A4"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_contraindicated >= 1


def test_moderate_inhibitor():
    """IC50=10 uM, Css=2.5 uM → AUCR = 1.25 → moderate."""
    drugs = [
        _drug("ModInh", css_uM=2.5, inhibitions={"CYP3A4": 10.0}),
        _drug("Substrate", substrates=["CYP3A4"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_interactions_moderate_plus >= 1


def test_major_inhibitor():
    """IC50=1 uM, Css=5 uM → AUCR = 6 → contraindicated."""
    drugs = [
        _drug("MajorInh", css_uM=5.0, inhibitions={"CYP3A4": 1.0}),
        _drug("Substrate", substrates=["CYP3A4"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_interactions_major_plus >= 1


# --- Severity counting consistency ---


def test_n_contraindicated_le_n_major_plus():
    drugs = [
        _drug("Inh1", css_uM=10.0, inhibitions={"CYP3A4": 0.1, "CYP2D6": 0.5}),
        _drug("Sub1", substrates=["CYP3A4"]),
        _drug("Sub2", substrates=["CYP2D6"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_contraindicated <= result.n_interactions_major_plus


def test_n_major_plus_le_n_moderate_plus():
    drugs = [
        _drug("Inh1", css_uM=5.0, inhibitions={"CYP3A4": 1.0}),
        _drug("Sub1", substrates=["CYP3A4"]),
        _drug("Sub2", substrates=["CYP2D6"]),
    ]
    result = score_ddi_network(drugs)
    assert result.n_interactions_major_plus <= result.n_interactions_moderate_plus


def test_max_aucr_ge_1():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert result.max_aucr >= 1.0


# --- polypharmacy_risk field ---


def test_polypharmacy_risk_values():
    drugs = [_drug("A"), _drug("B")]
    result = score_ddi_network(drugs)
    assert result.polypharmacy_risk in {"low", "moderate", "high"}


def test_high_risk_strong_inhibitors():
    """Multiple strong interactions should push to high risk."""
    drugs = [
        _drug("I1", css_uM=50.0, inhibitions={"CYP3A4": 0.1, "CYP2D6": 0.1}),
        _drug("S1", substrates=["CYP3A4"]),
        _drug("S2", substrates=["CYP2D6"]),
        _drug("S3", substrates=["CYP3A4", "CYP2D6"]),
    ]
    result = score_ddi_network(drugs)
    assert result.polypharmacy_risk in {"moderate", "high"}


# --- add_drug_to_regimen ---


def test_add_drug_returns_dict_with_keys():
    existing = [_drug("A"), _drug("B")]
    new = _drug("C")
    result = add_drug_to_regimen(existing, new)
    assert "new_interactions" in result
    assert "overall_result" in result
    assert "recommendation" in result


def test_add_drug_overall_result_type():
    existing = [_drug("A"), _drug("B")]
    new = _drug("C")
    result = add_drug_to_regimen(existing, new)
    assert isinstance(result["overall_result"], DDINetworkResult)


def test_add_drug_detects_new_interaction():
    existing = [_drug("Substrate", substrates=["CYP3A4"])]
    new = _drug("Inhibitor", css_uM=10.0, inhibitions={"CYP3A4": 1.0})
    result = add_drug_to_regimen(existing, new)
    # Should detect the interaction
    assert len(result["new_interactions"]) >= 1


def test_add_drug_no_interaction():
    existing = [_drug("Sub_3A4", substrates=["CYP3A4"])]
    new = _drug("NeutralDrug", substrates=["CYP2D6"])
    result = add_drug_to_regimen(existing, new)
    assert len(result["new_interactions"]) == 0


def test_add_drug_recommendation_is_string():
    existing = [_drug("A"), _drug("B")]
    new = _drug("C")
    result = add_drug_to_regimen(existing, new)
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0


# --- Validation ---


def test_validation_less_than_2_drugs():
    with pytest.raises(ValueError, match="At least 2 drugs required"):
        score_ddi_network([_drug("A")])


def test_validation_empty_list():
    with pytest.raises(ValueError, match="At least 2 drugs required"):
        score_ddi_network([])
