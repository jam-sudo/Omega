"""Tests for comprehensive ADMET profiler."""

import pytest

from omega_pbpk.prediction.admet_profile import (
    ADMETProfile,
    predict_admet,
    screen_admet,
)

# --- Basic ---


def test_returns_admet_profile():
    r = predict_admet("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", mw=180.16, logP=1.2)
    assert isinstance(r, ADMETProfile)


def test_mw_le_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_admet("X", "C", mw=0, logP=1.0)


# --- Physicochemical ---


def test_fup_in_range():
    r = predict_admet("Drug", "CCCCCC", mw=300, logP=2.5)
    assert 0 < r.fup < 1


def test_bcs_class_valid():
    r = predict_admet("Drug", "CCO", mw=200, logP=1.0)
    assert r.bcs_class in {"I", "II", "III", "IV"}


# --- Lipinski ---


def test_lipinski_pass_typical_drug():
    r = predict_admet("Good", "CCO", mw=350, logP=2.0, tpsa=60)
    assert r.lipinski_pass is True


def test_lipinski_fail_large_mw():
    r = predict_admet("Big", "C" * 20, mw=600, logP=2.0)
    assert r.lipinski_pass is False


# --- BBB ---


def test_bbb_penetrant_cns_drug():
    r = predict_admet("CNS", "c1ccccc1", mw=200, logP=2.5, tpsa=30)
    assert r.bbb_penetrant is True


def test_bbb_not_penetrant_polar():
    r = predict_admet("Polar", "OC(=O)NCCNC(=O)O", mw=300, logP=0.5, tpsa=120)
    assert r.bbb_penetrant is False


# --- hERG ---


def test_herg_risk_valid():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert r.herg_risk in {"low", "moderate", "high"}


# --- Scores ---


def test_overall_risk_in_range():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert 0 <= r.overall_risk_score <= 100


def test_drug_likeness_in_range():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert 0 <= r.drug_likeness_score <= 1


def test_development_recommendation_valid():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert r.development_recommendation in {"promising", "optimize", "concern"}


# --- PK estimates ---


def test_t_half_positive():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert r.t_half_h > 0


def test_vd_positive():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert r.vd_L_per_kg > 0


def test_f_oral_in_range():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0)
    assert 0 < r.f_oral < 1


# --- Screening ---


def test_screen_returns_same_length():
    compounds = [
        {"name": "A", "smiles": "CCO", "mw": 200, "logP": 1.0},
        {"name": "B", "smiles": "CCCCCC", "mw": 400, "logP": 4.0},
    ]
    results = screen_admet(compounds)
    assert len(results) == 2


def test_screen_sorted_by_risk():
    compounds = [
        {"name": "Bad", "smiles": "C" * 20, "mw": 600, "logP": 6.0},
        {"name": "Good", "smiles": "CCO", "mw": 200, "logP": 1.0},
    ]
    results = screen_admet(compounds)
    assert results[0].overall_risk_score <= results[1].overall_risk_score


# --- Notes ---


def test_notes_contain_compound_name():
    r = predict_admet("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", mw=206.28, logP=3.5)
    assert "Ibuprofen" in r.notes


# --- logS estimated ---


def test_logs_estimated_when_none():
    r = predict_admet("Drug", "CCO", mw=300, logP=2.0, logS=None)
    assert r.logS is not None


# --- HBD/HBA ---


def test_hbd_hba_non_negative_int():
    r = predict_admet("Drug", "OC(=O)NCCNC(=O)O", mw=200, logP=0.5)
    assert isinstance(r.hbd, int) and r.hbd >= 0
    assert isinstance(r.hba, int) and r.hba >= 0
