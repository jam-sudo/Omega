"""Tests for Phase 966 — Biofilm penetration predictor."""

import pytest

from omega_pbpk.prediction.biofilm_penetration import (
    BiofilmPenetrationResult,
    predict_biofilm_penetration,
    screen_biofilm_activity,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="ciprofloxacin",
        logp=0.28,
        mw=331.3,
        pka=6.1,
        ionization_type="neutral",
        c_bulk_mg_L=2.0,
    )
    defaults.update(kwargs)
    return predict_biofilm_penetration(**defaults)


def test_return_type_frozen():
    result = _default_result()
    assert isinstance(result, BiofilmPenetrationResult)


def test_frozen_immutable():
    result = _default_result()
    with pytest.raises(Exception):
        result.f_penetration = 0.99  # type: ignore


def test_f_penetration_bounds():
    result = _default_result()
    assert 0.01 <= result.f_penetration <= 1.0


def test_c_biofilm_positive():
    result = _default_result()
    assert result.c_biofilm_effective_mg_L > 0


def test_c_biofilm_le_bulk():
    result = _default_result()
    assert result.c_biofilm_effective_mg_L <= result.c_bulk_mg_L


def test_mbec_mic_ratio_ge_2():
    result = _default_result()
    assert result.mbec_mic_ratio >= 2.0


def test_penetration_category_valid():
    result = _default_result()
    assert result.penetration_category in {"good", "moderate", "poor"}


def test_biofilm_activity_factor_positive():
    result = _default_result()
    assert result.biofilm_activity_factor > 0


def test_recommended_dose_multiplier_ge_1():
    result = _default_result()
    assert result.recommended_dose_multiplier >= 1.0


def test_notes_nonempty():
    result = _default_result()
    assert result.notes != ""


def test_drug_name_preserved():
    result = _default_result(drug_name="amikacin")
    assert result.drug_name == "amikacin"


def test_optimal_logp_higher_penetration_than_extreme():
    """logP=1.5 (optimal) should give higher f_penetration than logP=5.0."""
    optimal = predict_biofilm_penetration("drug_opt", 1.5, 400.0, 7.0, "neutral")
    extreme = predict_biofilm_penetration("drug_ext", 5.0, 400.0, 7.0, "neutral")
    assert optimal.f_penetration > extreme.f_penetration


def test_low_mw_higher_penetration_than_high_mw():
    """MW=300 should give higher penetration than MW=900 (same logP)."""
    low_mw = predict_biofilm_penetration("low_mw", 1.0, 300.0, 7.0, "neutral")
    high_mw = predict_biofilm_penetration("high_mw", 1.0, 900.0, 7.0, "neutral")
    assert low_mw.f_penetration > high_mw.f_penetration


def test_mbec_mic_ratio_always_ge_2():
    """Regardless of parameters, MBEC/MIC >= 2."""
    # Use optimal penetration (should still be >= 2)
    result = predict_biofilm_penetration("drug", 1.0, 200.0, 7.0, "neutral")
    assert result.mbec_mic_ratio >= 2.0


def test_c_biofilm_equals_bulk_times_f_penetration():
    result = _default_result(c_bulk_mg_L=4.0)
    expected = result.c_bulk_mg_L * result.f_penetration
    assert abs(result.c_biofilm_effective_mg_L - expected) < 1e-9


def test_screen_biofilm_activity_correct_length():
    compounds = [
        {"drug_name": "drug_a", "logp": 1.0, "mw": 300.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "drug_b", "logp": 4.0, "mw": 500.0, "pka": 5.0, "ionization_type": "acid"},
        {"drug_name": "drug_c", "logp": -2.0, "mw": 400.0, "pka": 9.0, "ionization_type": "base"},
    ]
    results = screen_biofilm_activity(compounds)
    assert len(results) == 3


def test_screen_biofilm_activity_sorted_descending():
    compounds = [
        {"drug_name": "drug_a", "logp": 5.0, "mw": 800.0, "pka": 7.0, "ionization_type": "base"},
        {"drug_name": "drug_b", "logp": 1.0, "mw": 300.0, "pka": 7.0, "ionization_type": "neutral"},
        {
            "drug_name": "drug_c",
            "logp": -2.0,
            "mw": 400.0,
            "pka": 9.0,
            "ionization_type": "neutral",
        },
    ]
    results = screen_biofilm_activity(compounds)
    for i in range(len(results) - 1):
        assert results[i].f_penetration >= results[i + 1].f_penetration


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_biofilm_penetration("drug", 1.0, 0.0, 7.0, "neutral")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_biofilm_penetration("drug", 1.0, -100.0, 7.0, "neutral")


def test_validation_c_bulk_zero():
    with pytest.raises(ValueError, match="c_bulk_mg_L"):
        predict_biofilm_penetration("drug", 1.0, 300.0, 7.0, "neutral", c_bulk_mg_L=0.0)


def test_validation_c_bulk_negative():
    with pytest.raises(ValueError, match="c_bulk_mg_L"):
        predict_biofilm_penetration("drug", 1.0, 300.0, 7.0, "neutral", c_bulk_mg_L=-1.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_biofilm_penetration("drug", 1.0, 300.0, 7.0, "zwitterion")


def test_penetration_category_good_when_f_above_0_6():
    """Optimal logP and low MW should yield 'good' category."""
    result = predict_biofilm_penetration("drug", 1.0, 200.0, 7.0, "neutral")
    assert result.f_penetration > 0.6
    assert result.penetration_category == "good"


def test_penetration_category_poor_when_f_le_0_3():
    """High logP + high MW + cationic should yield 'poor' category."""
    result = predict_biofilm_penetration("drug", 5.0, 1000.0, 9.0, "base")
    assert result.f_penetration <= 0.3
    assert result.penetration_category == "poor"
