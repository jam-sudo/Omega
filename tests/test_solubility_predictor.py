"""Tests for Phase 665 — prediction/solubility_predictor.py"""

import math

import pytest

from omega_pbpk.prediction.solubility_predictor import (
    SolubilityPrediction,
    predict_solubility,
    screen_solubility,
)

VALID_CLASSES = {"very_low", "low", "moderate", "high"}
VALID_BCS = {"high", "low"}


# --- Basic return type tests ---


def test_returns_solubility_prediction():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert isinstance(result, SolubilityPrediction)


def test_predicted_solubility_mg_mL_positive():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert result.predicted_solubility_mg_mL > 0


def test_predicted_solubility_ug_mL_is_1000x_mg_mL():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert (
        abs(result.predicted_solubility_ug_mL - result.predicted_solubility_mg_mL * 1000.0) < 1e-6
    )


def test_predicted_log_s_is_finite():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert math.isfinite(result.predicted_log_s)


def test_solubility_class_valid():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert result.solubility_class in VALID_CLASSES


def test_bcs_solubility_valid():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert result.bcs_solubility in VALID_BCS


def test_notes_nonempty():
    result = predict_solubility("TestDrug", logP=2.0, mw=300.0, psa=60.0)
    assert result.notes


# --- Scientific behavior tests ---


def test_high_logP_lower_solubility_than_low_logP():
    r_high = predict_solubility("HighLogP", logP=6.0, mw=300.0, psa=30.0)
    r_low = predict_solubility("LowLogP", logP=1.0, mw=300.0, psa=30.0)
    assert r_high.predicted_solubility_mg_mL < r_low.predicted_solubility_mg_mL


def test_high_psa_higher_solubility():
    r_high_psa = predict_solubility("HighPSA", logP=2.0, mw=300.0, psa=120.0)
    r_low_psa = predict_solubility("LowPSA", logP=2.0, mw=300.0, psa=20.0)
    assert r_high_psa.predicted_solubility_mg_mL > r_low_psa.predicted_solubility_mg_mL


def test_acidic_drug_high_pH_ionization_correction_applied():
    result = predict_solubility("AcidicDrug", logP=2.0, mw=250.0, psa=50.0, pka_acid=5.0, pH=7.4)
    assert result.ionization_correction_applied is True


def test_acidic_drug_higher_solubility_at_high_pH():
    r_ionized = predict_solubility("AcidicDrug", logP=2.0, mw=250.0, psa=50.0, pka_acid=5.0, pH=7.4)
    r_neutral = predict_solubility("NeutralDrug", logP=2.0, mw=250.0, psa=50.0, pH=7.4)
    assert r_ionized.predicted_solubility_mg_mL > r_neutral.predicted_solubility_mg_mL


def test_high_melting_point_lower_solubility():
    r_high_mp = predict_solubility("HighMP", logP=2.0, mw=300.0, psa=60.0, melting_point_c=250.0)
    r_low_mp = predict_solubility("LowMP", logP=2.0, mw=300.0, psa=60.0, melting_point_c=80.0)
    assert r_high_mp.predicted_solubility_mg_mL < r_low_mp.predicted_solubility_mg_mL


def test_high_mw_lower_solubility():
    r_large = predict_solubility("LargeMW", logP=2.0, mw=700.0, psa=60.0)
    r_small = predict_solubility("SmallMW", logP=2.0, mw=200.0, psa=60.0)
    assert r_large.predicted_solubility_mg_mL < r_small.predicted_solubility_mg_mL


def test_bcs_high_solubility_drug():
    # Very soluble: logP=0, small MW, high PSA → expect BCS high
    result = predict_solubility(
        "Soluble",
        logP=0.0,
        mw=150.0,
        psa=100.0,
        melting_point_c=100.0,
        dose_mg=10.0,
        dose_volume_mL=250.0,
    )
    assert result.bcs_solubility == "high"


def test_bcs_low_solubility_drug():
    # Very insoluble: logP=7, large MW, low PSA
    result = predict_solubility(
        "Insoluble",
        logP=7.0,
        mw=500.0,
        psa=10.0,
        melting_point_c=250.0,
        dose_mg=100.0,
        dose_volume_mL=250.0,
    )
    assert result.bcs_solubility == "low"


def test_aqueous_solubility_flag_true_for_hydrophobic():
    # logP=7 → very hydrophobic → flag should be True
    result = predict_solubility("Hydrophobic", logP=7.0, mw=400.0, psa=20.0)
    assert result.aqueous_solubility_flag is True


def test_aqueous_solubility_flag_false_for_hydrophilic():
    # logP=0, small MW, high PSA → very soluble → no flag
    result = predict_solubility("Hydrophilic", logP=0.0, mw=150.0, psa=120.0, melting_point_c=80.0)
    assert result.aqueous_solubility_flag is False


# --- screen_solubility tests ---


def test_screen_solubility_sorted_descending():
    compounds = [
        {"compound_name": "A", "logP": 5.0, "mw": 400.0, "psa": 20.0},
        {"compound_name": "B", "logP": 1.0, "mw": 200.0, "psa": 80.0},
        {"compound_name": "C", "logP": 3.0, "mw": 300.0, "psa": 50.0},
    ]
    results = screen_solubility(compounds)
    assert len(results) == 3
    solvs = [r.predicted_solubility_mg_mL for r in results]
    assert solvs == sorted(solvs, reverse=True)


def test_screen_solubility_empty_returns_empty():
    assert screen_solubility([]) == []


# --- Validation tests ---


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_solubility("X", logP=2.0, mw=0.0, psa=60.0)


def test_validation_pH_negative_raises():
    with pytest.raises(ValueError, match="pH"):
        predict_solubility("X", logP=2.0, mw=300.0, psa=60.0, pH=-1.0)


def test_validation_pH_above_14_raises():
    with pytest.raises(ValueError, match="pH"):
        predict_solubility("X", logP=2.0, mw=300.0, psa=60.0, pH=15.0)


def test_ionization_correction_false_for_neutral_drug():
    result = predict_solubility("Neutral", logP=2.0, mw=300.0, psa=60.0)
    assert result.ionization_correction_applied is False
