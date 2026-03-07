"""Tests for prediction/solubility_predictor.py — Phase 265."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.solubility_predictor import (
    SolubilityResult,
    classify_solubility,
    predict_solubility,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_solubility("X", logP=1.0, mw=0, mp_C=100)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_solubility("X", logP=1.0, mw=-50, mp_C=100)


def test_invalid_mp_negative():
    with pytest.raises(ValueError, match="mp_C"):
        predict_solubility("X", logP=1.0, mw=300, mp_C=-5)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_solubility("X", logP=1.0, mw=300, mp_C=100, dose_mg=0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_solubility("X", logP=1.0, mw=300, mp_C=100, dose_mg=-10)


def test_invalid_n_rotatable_bonds_negative():
    with pytest.raises(ValueError, match="n_rotatable_bonds"):
        predict_solubility("X", logP=1.0, mw=300, mp_C=100, n_rotatable_bonds=-1)


def test_invalid_n_aromatic_rings_negative():
    with pytest.raises(ValueError, match="n_aromatic_rings"):
        predict_solubility("X", logP=1.0, mw=300, mp_C=100, n_aromatic_rings=-1)


# ---------------------------------------------------------------------------
# Basic result properties
# ---------------------------------------------------------------------------


def test_returns_solubility_result():
    r = predict_solubility("Aspirin", logP=1.2, mw=180.16, mp_C=135)
    assert isinstance(r, SolubilityResult)


def test_compound_name_stored():
    r = predict_solubility("Warfarin", logP=2.7, mw=308.3, mp_C=161)
    assert r.compound_name == "Warfarin"


def test_s_mg_mL_positive():
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert r.s_mg_mL > 0


def test_log_s_gse_is_finite():
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert math.isfinite(r.log_s_gse)


def test_log_s_esol_is_finite():
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert math.isfinite(r.log_s_esol)


def test_dose_solubility_number_positive():
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert r.dose_solubility_number > 0


def test_solubility_class_valid():
    valid = {"very_low", "low", "moderate", "high"}
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert r.solubility_class in valid


def test_bcs_class_solubility_valid():
    valid = {"high_solubility", "low_solubility"}
    r = predict_solubility("DrugA", logP=2.0, mw=300, mp_C=150)
    assert r.bcs_class_solubility in valid


# ---------------------------------------------------------------------------
# GSE model behaviour
# ---------------------------------------------------------------------------


def test_high_logP_high_mp_very_low_solubility():
    """High logP (5.0) and high mp (200 C) should give very_low solubility."""
    r = predict_solubility("HighLipophilic", logP=5.0, mw=400, mp_C=200)
    assert r.solubility_class == "very_low"


def test_low_logP_low_mp_high_or_moderate_solubility():
    """Low logP (1.0) and low mp (50 C) should give high or moderate solubility."""
    r = predict_solubility("Hydrophilic", logP=1.0, mw=200, mp_C=50)
    assert r.solubility_class in {"high", "moderate"}


def test_log_s_gse_consistent_with_s_mg_mL():
    """log_s_gse should equal log10(s_mg_mL)."""
    r = predict_solubility("Compound", logP=2.0, mw=300, mp_C=100)
    assert abs(r.log_s_gse - math.log10(r.s_mg_mL)) < 1e-6


def test_higher_logP_reduces_solubility():
    r_low = predict_solubility("LowP", logP=1.0, mw=250, mp_C=100)
    r_high = predict_solubility("HighP", logP=5.0, mw=250, mp_C=100)
    assert r_high.s_mg_mL < r_low.s_mg_mL


def test_high_mp_reduces_solubility():
    r_low_mp = predict_solubility("LowMp", logP=2.0, mw=300, mp_C=50)
    r_high_mp = predict_solubility("HighMp", logP=2.0, mw=300, mp_C=250)
    assert r_high_mp.s_mg_mL < r_low_mp.s_mg_mL


# ---------------------------------------------------------------------------
# BCS dose number
# ---------------------------------------------------------------------------


def test_dose_number_below_1_gives_high_bcs():
    """Very soluble compound should have dose_number < 1 and high_solubility."""
    r = predict_solubility("Soluble", logP=0.0, mw=100, mp_C=25, dose_mg=1.0)
    assert r.dose_solubility_number < 1.0
    assert r.bcs_class_solubility == "high_solubility"


def test_dose_number_above_1_gives_low_bcs():
    """Poorly soluble compound with high dose should have dose_number > 1."""
    r = predict_solubility("Insoluble", logP=6.0, mw=500, mp_C=250, dose_mg=500.0)
    assert r.dose_solubility_number > 1.0
    assert r.bcs_class_solubility == "low_solubility"


# ---------------------------------------------------------------------------
# classify_solubility function
# ---------------------------------------------------------------------------


def test_classify_very_low():
    assert classify_solubility(0.05) == "very_low"


def test_classify_low():
    assert classify_solubility(0.5) == "low"


def test_classify_moderate():
    assert classify_solubility(5.0) == "moderate"


def test_classify_high():
    assert classify_solubility(50.0) == "high"


def test_classify_boundary_0_1():
    """Exactly 0.1 should be 'low' not 'very_low'."""
    assert classify_solubility(0.1) == "low"


def test_classify_boundary_1_0():
    """Exactly 1.0 should be 'moderate'."""
    assert classify_solubility(1.0) == "moderate"


def test_classify_boundary_10_0():
    """Exactly 10.0 should be 'high'."""
    assert classify_solubility(10.0) == "high"
