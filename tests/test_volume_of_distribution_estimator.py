"""Tests for src/omega_pbpk/prediction/volume_of_distribution_estimator.py — Phase 686."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.volume_of_distribution_estimator import (
    VdEstimationResult,
    estimate_vd_from_kp,
    estimate_vd_oie,
    estimate_vd_poulin_theil,
    predict_vd,
    screen_vd,
)

# ---------------------------------------------------------------------------
# estimate_vd_oie
# ---------------------------------------------------------------------------


def test_oie_returns_float():
    vd = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.1)
    assert isinstance(vd, float)


def test_oie_positive():
    vd = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.1)
    assert vd > 0.0


def test_oie_high_logP_higher_vd():
    vd_high = estimate_vd_oie(logP=5.0, mw=300.0, fup=0.1)
    vd_low = estimate_vd_oie(logP=0.0, mw=300.0, fup=0.1)
    assert vd_high > vd_low


def test_oie_low_fup_higher_vd():
    vd_low_fup = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.01)
    vd_high_fup = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.5)
    assert vd_low_fup > vd_high_fup


def test_oie_basic_drug_higher_vd():
    """Basic drug with pKa > 7.4 (lysosomal trapping) → higher Vd than neutral."""
    vd_basic = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.1, pka_base=9.5)
    vd_neutral = estimate_vd_oie(logP=2.0, mw=300.0, fup=0.1, pka_base=None)
    assert vd_basic > vd_neutral


def test_oie_invalid_fup_zero_raises():
    with pytest.raises(ValueError):
        estimate_vd_oie(logP=2.0, mw=300.0, fup=0.0)


def test_oie_invalid_fup_over_1_raises():
    with pytest.raises(ValueError):
        estimate_vd_oie(logP=2.0, mw=300.0, fup=1.5)


def test_oie_invalid_mw_raises():
    with pytest.raises(ValueError):
        estimate_vd_oie(logP=2.0, mw=-100.0, fup=0.1)


# ---------------------------------------------------------------------------
# estimate_vd_poulin_theil
# ---------------------------------------------------------------------------


def test_poulin_theil_returns_float():
    vd = estimate_vd_poulin_theil(logP=2.0, fup=0.1)
    assert isinstance(vd, float)


def test_poulin_theil_positive():
    vd = estimate_vd_poulin_theil(logP=2.0, fup=0.1)
    assert vd > 0.0


def test_poulin_theil_high_logP_higher_vd():
    vd_high = estimate_vd_poulin_theil(logP=5.0, fup=0.1)
    vd_low = estimate_vd_poulin_theil(logP=0.0, fup=0.1)
    assert vd_high > vd_low


def test_poulin_theil_clamped_min():
    """Even with unfavorable params, should be >= 5.0 L."""
    vd = estimate_vd_poulin_theil(logP=-5.0, fup=1.0)
    assert vd >= 5.0


def test_poulin_theil_clamped_max():
    """Very high logP should be clamped to 1000.0 L."""
    vd = estimate_vd_poulin_theil(logP=100.0, fup=0.001)
    assert vd <= 1000.0


def test_poulin_theil_invalid_fup_raises():
    with pytest.raises(ValueError):
        estimate_vd_poulin_theil(logP=2.0, fup=0.0)


# ---------------------------------------------------------------------------
# estimate_vd_from_kp
# ---------------------------------------------------------------------------


def test_vd_from_kp_returns_float():
    vd = estimate_vd_from_kp({"adipose": 3.0, "muscle": 2.0}, fup=0.1)
    assert isinstance(vd, float)


def test_vd_from_kp_positive():
    vd = estimate_vd_from_kp({"liver": 5.0, "muscle": 3.0}, fup=0.1)
    assert vd > 0.0


def test_vd_from_kp_high_kp_high_vd():
    vd_high = estimate_vd_from_kp({"muscle": 20.0, "liver": 15.0}, fup=0.1)
    vd_low = estimate_vd_from_kp({"muscle": 1.0, "liver": 1.0}, fup=0.1)
    assert vd_high > vd_low


def test_vd_from_kp_invalid_fup_raises():
    with pytest.raises(ValueError):
        estimate_vd_from_kp({"muscle": 2.0}, fup=1.5)


# ---------------------------------------------------------------------------
# predict_vd
# ---------------------------------------------------------------------------


def test_predict_vd_returns_result():
    result = predict_vd("CompoundA", logP=2.0, mw=300.0, fup=0.1)
    assert isinstance(result, VdEstimationResult)


def test_predict_vd_ensemble_positive():
    result = predict_vd("CompoundA", logP=2.0, mw=300.0, fup=0.1)
    assert result.vd_ensemble_L > 0.0


def test_predict_vd_class_valid():
    result = predict_vd("CompoundA", logP=2.0, mw=300.0, fup=0.1)
    assert result.vd_class in {"low", "moderate", "high"}


def test_predict_vd_high_logP_class_high():
    """logP=5 drug should be classified as 'high' Vd (>100 L)."""
    result = predict_vd("LipophilicDrug", logP=5.0, mw=300.0, fup=0.05)
    assert result.vd_class == "high"


def test_predict_vd_low_logP_class_low():
    """logP=-1, high fup drug should be classified as 'low' Vd (<30 L)."""
    result = predict_vd("HydrophilicDrug", logP=-1.0, mw=300.0, fup=0.9)
    assert result.vd_class == "low"


def test_predict_vd_notes_nonempty():
    result = predict_vd("CompoundA", logP=2.0, mw=300.0, fup=0.1)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_predict_vd_basic_drug_higher_than_neutral():
    """Basic drug with pKa_base=9.5 should have higher Vd than neutral."""
    vd_basic = predict_vd("Basic", logP=2.0, mw=300.0, fup=0.1, pka_base=9.5)
    vd_neutral = predict_vd("Neutral", logP=2.0, mw=300.0, fup=0.1, pka_base=None)
    assert vd_basic.vd_ensemble_L > vd_neutral.vd_ensemble_L


def test_predict_vd_invalid_fup_raises():
    with pytest.raises(ValueError):
        predict_vd("CompoundA", logP=2.0, mw=300.0, fup=1.5)


def test_predict_vd_invalid_mw_raises():
    with pytest.raises(ValueError):
        predict_vd("CompoundA", logP=2.0, mw=0.0, fup=0.1)


# ---------------------------------------------------------------------------
# screen_vd
# ---------------------------------------------------------------------------


def test_screen_vd_returns_list():
    compounds = [
        {"compound_name": "A", "logP": 2.0, "fup": 0.1},
        {"compound_name": "B", "logP": 4.0, "fup": 0.05},
    ]
    results = screen_vd(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_vd_sorted_descending():
    compounds = [
        {"compound_name": "Low", "logP": -1.0, "fup": 0.9},
        {"compound_name": "High", "logP": 5.0, "fup": 0.05},
        {"compound_name": "Mid", "logP": 2.0, "fup": 0.1},
    ]
    results = screen_vd(compounds)
    vds = [r.vd_ensemble_L for r in results]
    assert vds == sorted(vds, reverse=True)


def test_screen_vd_empty_returns_empty():
    results = screen_vd([])
    assert results == []


def test_screen_vd_all_vd_estimation_results():
    compounds = [{"compound_name": "A", "logP": 2.0, "fup": 0.1}]
    results = screen_vd(compounds)
    assert all(isinstance(r, VdEstimationResult) for r in results)
