"""Tests for Phase 1006 — Molecular Fingerprint ADMET Predictor."""

import pytest

from omega_pbpk.prediction.molecular_fingerprint_admet import (
    FingerprintADMETResult,
    predict_admet_from_features,
    screen_admet_features,
)


# --- Return type ---

def test_returns_fingerprint_admet_result():
    result = predict_admet_from_features("TestDrug", "aromatic_ring")
    assert isinstance(result, FingerprintADMETResult)


# --- MW increases with features ---

def test_mw_greater_than_base_with_features():
    result = predict_admet_from_features("Drug", "aromatic_ring", base_mw=100.0)
    assert result.mw_predicted > 100.0


# --- HBD / HBA non-negative ---

def test_hbd_non_negative():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert result.hbd_count >= 0


def test_hba_non_negative():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert result.hba_count >= 0


# --- Papp positive ---

def test_caco2_papp_positive():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert result.predicted_caco2_papp > 0.0


# --- Bioavailability in [0, 1] ---

def test_bioavailability_in_range():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert 0.0 <= result.predicted_bioavailability <= 1.0


# --- t½ positive ---

def test_t_half_positive():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert result.predicted_t_half_h > 0.0


# --- Vd positive ---

def test_vd_positive():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert result.predicted_vd_L_per_kg > 0.0


# --- BBB in [0, 1] ---

def test_bbb_penetration_in_range():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert 0.0 <= result.bbb_penetration_probability <= 1.0


# --- Drug-likeness in [0, 100] ---

def test_drug_likeness_in_range():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert 0.0 <= result.drug_likeness_score <= 100.0


# --- Lipinski bool ---

def test_lipinski_pass_is_bool():
    result = predict_admet_from_features("Drug", "aromatic_ring")
    assert isinstance(result.lipinski_pass, bool)


# --- Two aromatic rings → 2 rings ---

def test_two_aromatic_rings_counted():
    result = predict_admet_from_features("Drug", "aromatic_ring,aromatic_ring")
    assert result.aromatic_ring_count == 2


# --- Polar features lower logP ---

def test_polar_features_lower_logp():
    lipophilic = predict_admet_from_features("Drug", "aliphatic_chain_C4,aliphatic_chain_C4")
    polar = predict_admet_from_features("Drug", "hydroxyl,carboxylic_acid")
    assert polar.logp_predicted < lipophilic.logp_predicted


# --- screen_admet_features returns sorted list ---

def test_screen_returns_sorted_list():
    compounds = [
        {"drug_name": "A", "feature_string": "aromatic_ring,hydroxyl"},
        {"drug_name": "B", "feature_string": "aliphatic_chain_C4,aliphatic_chain_C4,aliphatic_chain_C4"},
        {"drug_name": "C", "feature_string": "aromatic_ring,ester"},
    ]
    results = screen_admet_features(compounds)
    assert isinstance(results, list)
    assert len(results) == 3
    scores = [r.drug_likeness_score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- Empty feature string ---

def test_empty_feature_string():
    result = predict_admet_from_features("Empty", "", base_mw=150.0)
    assert isinstance(result, FingerprintADMETResult)
    assert result.mw_predicted == pytest.approx(150.0)
    assert result.hbd_count == 0
    assert result.hba_count == 0


# --- Unknown features silently ignored ---

def test_unknown_feature_no_error():
    result = predict_admet_from_features("Drug", "aromatic_ring,unknown_xyz,halogen_F")
    assert isinstance(result, FingerprintADMETResult)
    # Known features contributed; unknown was skipped
    assert result.mw_predicted > 100.0


# --- Fragment count ---

def test_fragment_count_includes_unknown():
    result = predict_admet_from_features("Drug", "aromatic_ring,unknown_xyz")
    assert result.smiles_fragment_count == 2


# --- base_mw respected ---

def test_base_mw_custom():
    result = predict_admet_from_features("Drug", "", base_mw=200.0)
    assert result.mw_predicted == pytest.approx(200.0)
