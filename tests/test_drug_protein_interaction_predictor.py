"""Tests for Phase 275: Drug-protein interaction predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_protein_interaction_predictor import (
    DrugProteinInteractionResult,
    predict_drug_protein_interaction,
    screen_protein_binding,
)

_PROTEINS = [
    "albumin",
    "alpha1_acid_glycoprotein",
    "lipoprotein",
    "cytochrome_p450_3A4",
    "p_glycoprotein",
]


def _pred(target_protein="albumin", **kw):
    defaults = dict(drug_name="TestDrug", mw=350.0, logp=2.5, psa_A2=60.0)
    defaults.update(kw)
    return predict_drug_protein_interaction(**defaults, target_protein=target_protein)


def test_returns_correct_type():
    assert isinstance(_pred(), DrugProteinInteractionResult)


def test_drug_name_preserved():
    r = _pred(drug_name="Warfarin")
    assert r.drug_name == "Warfarin"


def test_target_protein_preserved():
    for protein in _PROTEINS:
        r = _pred(target_protein=protein)
        assert r.target_protein == protein


def test_kd_positive():
    for protein in _PROTEINS:
        assert _pred(target_protein=protein).predicted_kd_nM > 0.0


def test_higher_logp_lower_kd_albumin():
    r_low = _pred(logp=0.0, target_protein="albumin")
    r_high = _pred(logp=5.0, target_protein="albumin")
    assert r_high.predicted_kd_nM < r_low.predicted_kd_nM


def test_higher_psa_higher_kd():
    r_low = _pred(psa_A2=20.0, target_protein="albumin")
    r_high = _pred(psa_A2=200.0, target_protein="albumin")
    assert r_high.predicted_kd_nM > r_low.predicted_kd_nM


def test_higher_mw_higher_kd():
    r_low = _pred(mw=200.0)
    r_high = _pred(mw=800.0)
    assert r_high.predicted_kd_nM > r_low.predicted_kd_nM


def test_bound_fraction_between_0_and_1():
    for protein in _PROTEINS:
        bf = _pred(target_protein=protein).bound_fraction_at_1uM
        assert 0.0 <= bf <= 1.0


def test_fu_protein_plus_bound_fraction_equals_1():
    r = _pred()
    assert abs(r.fu_protein + r.bound_fraction_at_1uM - 1.0) < 1e-9


def test_high_logp_higher_bound_fraction():
    r_low = _pred(logp=0.0)
    r_high = _pred(logp=5.0)
    assert r_high.bound_fraction_at_1uM >= r_low.bound_fraction_at_1uM


def test_interaction_strength_valid_values():
    valid = {"strong", "moderate", "weak"}
    for protein in _PROTEINS:
        assert _pred(target_protein=protein).interaction_strength in valid


def test_interaction_strength_weak_for_hydrophilic():
    r = _pred(logp=-4.9, psa_A2=300.0, mw=900.0, target_protein="lipoprotein")
    assert r.interaction_strength == "weak"


def test_selectivity_index_positive():
    for protein in _PROTEINS:
        assert _pred(target_protein=protein).selectivity_index > 0.0


def test_lower_kd_higher_selectivity():
    r_low = _pred(logp=0.0)
    r_high = _pred(logp=5.0)
    assert r_high.selectivity_index >= r_low.selectivity_index


def test_binding_risk_valid_values():
    valid = {"high", "moderate", "low"}
    for protein in _PROTEINS:
        assert _pred(target_protein=protein).binding_risk in valid


def test_non_albumin_always_low_risk():
    non_albumin = [p for p in _PROTEINS if p != "albumin"]
    for protein in non_albumin:
        assert _pred(target_protein=protein).binding_risk == "low"


def test_notes_nonempty():
    assert len(_pred().notes) > 0


def test_error_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        _pred(mw=0.0)


def test_error_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        _pred(mw=-100.0)


def test_error_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        _pred(logp=-5.0)


def test_error_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        _pred(logp=10.0)


def test_error_psa_negative():
    with pytest.raises(ValueError, match="psa_A2"):
        _pred(psa_A2=-1.0)


def test_error_invalid_protein():
    with pytest.raises(ValueError, match="target_protein"):
        _pred(target_protein="unknown_protein")


def test_screen_returns_list():
    results = screen_protein_binding("Drug", mw=350.0, logp=2.5, psa_A2=60.0)
    assert isinstance(results, list)


def test_screen_returns_all_proteins():
    results = screen_protein_binding("Drug", mw=350.0, logp=2.5, psa_A2=60.0)
    assert len(results) == len(_PROTEINS)


def test_screen_sorted_descending():
    results = screen_protein_binding("Drug", mw=350.0, logp=2.5, psa_A2=60.0)
    bfs = [r.bound_fraction_at_1uM for r in results]
    assert bfs == sorted(bfs, reverse=True)


def test_screen_all_results_are_correct_type():
    results = screen_protein_binding("Drug", mw=350.0, logp=2.5, psa_A2=60.0)
    assert all(isinstance(r, DrugProteinInteractionResult) for r in results)
