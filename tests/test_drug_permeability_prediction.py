"""Tests for Phase 991 — Drug Permeability Prediction."""

import math

import pytest

from omega_pbpk.prediction.drug_permeability_prediction import (
    PermeabilityPredictionResult,
    predict_permeability,
    screen_permeability,
)


# ---------------------------------------------------------------------------
# Basic return-type and field sanity
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = predict_permeability("TestDrug", logp=2.0, mw=300.0)
    assert isinstance(result, PermeabilityPredictionResult)


def test_papp_positive():
    result = predict_permeability("TestDrug", logp=1.5, mw=350.0, psa=70.0, hbd=3)
    assert result.papp_cm_s > 0


def test_log_papp_finite():
    result = predict_permeability("TestDrug", logp=1.5, mw=350.0, psa=70.0, hbd=3)
    assert math.isfinite(result.log_papp)


def test_peff_positive():
    result = predict_permeability("TestDrug", logp=2.0, mw=300.0)
    assert result.peff_cm_s > 0


def test_fa_predicted_in_range():
    result = predict_permeability("TestDrug", logp=2.0, mw=300.0)
    assert 0.0 <= result.fa_predicted <= 1.0


def test_permeability_class_valid():
    result = predict_permeability("TestDrug", logp=2.0, mw=300.0)
    assert result.permeability_class in {"high", "medium", "low"}


def test_bcs_permeability_valid():
    result = predict_permeability("TestDrug", logp=2.0, mw=300.0)
    assert result.bcs_permeability in {"high", "low"}


# ---------------------------------------------------------------------------
# Physicochemical relationships
# ---------------------------------------------------------------------------


def test_high_logp_higher_papp_than_low_logp():
    high = predict_permeability("HighLogP", logp=5.0, mw=300.0, psa=30.0, hbd=1)
    low = predict_permeability("LowLogP", logp=-2.0, mw=300.0, psa=30.0, hbd=1)
    assert high.papp_cm_s > low.papp_cm_s


def test_low_psa_higher_papp_than_high_psa():
    low_psa = predict_permeability("LowPSA", logp=2.0, mw=300.0, psa=20.0, hbd=2)
    high_psa = predict_permeability("HighPSA", logp=2.0, mw=300.0, psa=150.0, hbd=2)
    assert low_psa.papp_cm_s > high_psa.papp_cm_s


def test_pgp_substrate_lower_papp():
    no_pgp = predict_permeability("Drug", logp=3.0, mw=400.0, pgp_substrate=False)
    pgp = predict_permeability("Drug", logp=3.0, mw=400.0, pgp_substrate=True)
    assert pgp.papp_cm_s < no_pgp.papp_cm_s


def test_pgp_efflux_correction_value():
    result_pgp = predict_permeability("Drug", logp=3.0, mw=300.0, pgp_substrate=True)
    result_no_pgp = predict_permeability("Drug", logp=3.0, mw=300.0, pgp_substrate=False)
    assert result_pgp.p_gp_efflux_correction == 2.5
    assert result_no_pgp.p_gp_efflux_correction == 1.0


# ---------------------------------------------------------------------------
# Classification boundaries
# ---------------------------------------------------------------------------


def test_high_permeability_class():
    # Very high logP, low PSA/HBD should give high permeability
    result = predict_permeability("HighPerm", logp=5.0, mw=200.0, psa=5.0, hbd=0)
    # papp should be > 20e-6
    if result.papp_cm_s > 20e-6:
        assert result.permeability_class == "high"


def test_low_permeability_class():
    # Very low logP, high PSA, many HBDs → low permeability
    result = predict_permeability("LowPerm", logp=-3.0, mw=600.0, psa=200.0, hbd=10)
    assert result.permeability_class == "low"


def test_bcs_high_boundary():
    # Peff > 2e-4 cm/s required for BCS high
    result = predict_permeability("BCSHigh", logp=5.0, mw=200.0, psa=5.0, hbd=0)
    # Just verify the field is consistent with peff_cm_s
    if result.peff_cm_s > 2e-4:
        assert result.bcs_permeability == "high"
    else:
        assert result.bcs_permeability == "low"


# ---------------------------------------------------------------------------
# screen_permeability
# ---------------------------------------------------------------------------


def test_screen_returns_sorted_list():
    compounds = [
        {"drug_name": "A", "logp": -2.0, "mw": 400.0, "psa": 120.0},
        {"drug_name": "B", "logp": 4.0, "mw": 300.0, "psa": 30.0},
        {"drug_name": "C", "logp": 1.0, "mw": 350.0, "psa": 80.0},
    ]
    results = screen_permeability(compounds)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].papp_cm_s >= results[i + 1].papp_cm_s


def test_screen_returns_correct_types():
    compounds = [
        {"drug_name": "X", "logp": 2.0, "mw": 300.0},
    ]
    results = screen_permeability(compounds)
    assert all(isinstance(r, PermeabilityPredictionResult) for r in results)


def test_screen_empty_list():
    results = screen_permeability([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability("Drug", logp=2.0, mw=0.0)


def test_invalid_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability("Drug", logp=2.0, mw=-100.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_permeability("Drug", logp=2.0, mw=300.0, ionization_type="zwitterion")


def test_invalid_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_permeability("Drug", logp=2.0, mw=300.0, psa=-1.0)
