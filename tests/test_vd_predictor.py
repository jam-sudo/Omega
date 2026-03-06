"""Tests for omega_pbpk.prediction.vd_predictor module."""

import pytest

from omega_pbpk.prediction.vd_predictor import (
    VdPredictionResult,
    predict_vd,
    vd_allometric_scale,
    vd_population_range,
)

VALID_CATEGORIES = {"low", "moderate", "high", "very_high"}


# --- VdPredictionResult dataclass ---


def test_result_is_dataclass_instance():
    result = predict_vd("DrugA", logP=2.0, MW=300.0, fup=0.1)
    assert isinstance(result, VdPredictionResult)


def test_result_fields_present():
    result = predict_vd("DrugA", logP=2.0, MW=300.0, fup=0.1)
    for field in (
        "drug_name",
        "logP",
        "MW",
        "fup",
        "pka",
        "PSA",
        "vd_predicted_L",
        "vd_per_kg_L_per_kg",
        "vd_central_L",
        "vd_peripheral_L",
        "tissue_distribution_category",
        "confidence",
        "allometric_exponent",
    ):
        assert hasattr(result, field), f"Missing field: {field}"


def test_drug_name_stored():
    result = predict_vd("Warfarin", logP=2.7, MW=308.0, fup=0.01)
    assert result.drug_name == "Warfarin"


def test_logP_MW_fup_stored():
    result = predict_vd("DrugB", logP=1.5, MW=250.0, fup=0.3)
    assert result.logP == 1.5
    assert result.MW == 250.0
    assert result.fup == 0.3


def test_pka_default():
    result = predict_vd("DrugC", logP=1.0, MW=200.0, fup=0.5)
    assert result.pka == 7.0


def test_psa_default():
    result = predict_vd("DrugC", logP=1.0, MW=200.0, fup=0.5)
    assert result.PSA == 60.0


def test_allometric_exponent_is_0_9():
    result = predict_vd("DrugD", logP=2.0, MW=350.0, fup=0.2)
    assert result.allometric_exponent == pytest.approx(0.9)


# --- Central + peripheral volumes ---


def test_vd_central_plus_peripheral_equals_predicted():
    result = predict_vd("DrugE", logP=2.5, MW=400.0, fup=0.15)
    assert result.vd_central_L + result.vd_peripheral_L == pytest.approx(
        result.vd_predicted_L, rel=1e-6
    )


def test_vd_central_is_40_pct():
    result = predict_vd("DrugF", logP=3.0, MW=300.0, fup=0.1)
    assert result.vd_central_L == pytest.approx(result.vd_predicted_L * 0.4, rel=1e-6)


def test_vd_peripheral_is_60_pct():
    result = predict_vd("DrugF", logP=3.0, MW=300.0, fup=0.1)
    assert result.vd_peripheral_L == pytest.approx(result.vd_predicted_L * 0.6, rel=1e-6)


def test_vd_per_kg_is_vd_over_70():
    result = predict_vd("DrugG", logP=1.0, MW=200.0, fup=0.4)
    assert result.vd_per_kg_L_per_kg == pytest.approx(result.vd_predicted_L / 70.0, rel=1e-6)


# --- Category ---


def test_category_is_valid_string():
    result = predict_vd("DrugH", logP=2.0, MW=300.0, fup=0.2)
    assert result.tissue_distribution_category in VALID_CATEGORIES


def test_category_low_for_small_vd():
    # High fup, low logP should give low vd
    result = predict_vd("SmallVd", logP=0.0, MW=150.0, fup=0.99, PSA=120.0)
    assert result.tissue_distribution_category == "low"


def test_category_very_high_for_large_vd():
    # Very low fup, high logP -> large vd
    result = predict_vd("LargeVd", logP=5.0, MW=500.0, fup=0.001, PSA=10.0)
    assert result.tissue_distribution_category == "very_high"


# --- Confidence ---


def test_confidence_high_when_logP_and_MW_in_range():
    result = predict_vd("DrugI", logP=2.0, MW=300.0, fup=0.1)
    assert result.confidence == pytest.approx(0.85)


def test_confidence_low_when_logP_out_of_range():
    result = predict_vd("DrugJ", logP=6.0, MW=300.0, fup=0.1)
    assert result.confidence == pytest.approx(0.6)


def test_confidence_low_when_MW_out_of_range_high():
    result = predict_vd("DrugK", logP=2.0, MW=600.0, fup=0.1)
    assert result.confidence == pytest.approx(0.6)


def test_confidence_low_when_MW_out_of_range_low():
    result = predict_vd("DrugL", logP=2.0, MW=80.0, fup=0.1)
    assert result.confidence == pytest.approx(0.6)


# --- Physicochemical relationships ---


def test_high_logP_gives_higher_vd_than_low_logP():
    low = predict_vd("LowLogP", logP=0.0, MW=300.0, fup=0.1, PSA=60.0)
    high = predict_vd("HighLogP", logP=5.0, MW=300.0, fup=0.1, PSA=60.0)
    assert high.vd_predicted_L > low.vd_predicted_L


def test_low_fup_gives_higher_vd_than_high_fup():
    low_fup = predict_vd("LowFup", logP=2.0, MW=300.0, fup=0.01)
    high_fup = predict_vd("HighFup", logP=2.0, MW=300.0, fup=0.9)
    assert low_fup.vd_predicted_L > high_fup.vd_predicted_L


def test_vd_predicted_within_bounds():
    result = predict_vd("BoundCheck", logP=2.0, MW=300.0, fup=0.2)
    assert 5.0 <= result.vd_predicted_L <= 5000.0


# --- vd_allometric_scale ---


def test_allometric_scale_returns_float():
    result = vd_allometric_scale(280.0, reference_weight_kg=70.0, target_weight_kg=20.0)
    assert isinstance(result, float)


def test_allometric_scale_same_weight_returns_ref_vd():
    ref_vd = 200.0
    result = vd_allometric_scale(ref_vd, reference_weight_kg=70.0, target_weight_kg=70.0)
    assert result == pytest.approx(ref_vd, rel=1e-6)


def test_allometric_scale_smaller_subject_gives_lower_vd():
    result_child = vd_allometric_scale(280.0, reference_weight_kg=70.0, target_weight_kg=20.0)
    assert result_child < 280.0


def test_allometric_scale_larger_subject_gives_higher_vd():
    result_large = vd_allometric_scale(280.0, reference_weight_kg=70.0, target_weight_kg=100.0)
    assert result_large > 280.0


def test_allometric_scale_custom_exponent():
    ref_vd, ref_w, target_w = 280.0, 70.0, 35.0
    exponent = 0.75
    expected = ref_vd * (target_w / ref_w) ** exponent
    result = vd_allometric_scale(ref_vd, ref_w, target_w, exponent=exponent)
    assert result == pytest.approx(expected, rel=1e-6)


# --- vd_population_range ---


def test_population_range_has_required_keys():
    result = vd_population_range("DrugM", logP=2.0, MW=300.0, fup=0.1)
    assert "p5" in result
    assert "p50" in result
    assert "p95" in result


def test_population_range_ordering():
    result = vd_population_range("DrugN", logP=2.0, MW=300.0, fup=0.1)
    assert result["p5"] < result["p50"] < result["p95"]


def test_population_range_p50_matches_predicted():
    pred = predict_vd("DrugO", logP=2.0, MW=300.0, fup=0.1)
    pop = vd_population_range("DrugO", logP=2.0, MW=300.0, fup=0.1)
    assert pop["p50"] == pytest.approx(pred.vd_predicted_L, rel=0.05)


def test_population_range_custom_cv():
    narrow = vd_population_range("DrugP", logP=2.0, MW=300.0, fup=0.1, cv_pct=10.0)
    wide = vd_population_range("DrugP", logP=2.0, MW=300.0, fup=0.1, cv_pct=80.0)
    narrow_spread = narrow["p95"] - narrow["p5"]
    wide_spread = wide["p95"] - wide["p5"]
    assert wide_spread > narrow_spread


# --- ValueError cases ---


def test_raises_for_fup_zero():
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=300.0, fup=0.0)


def test_raises_for_fup_negative():
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=300.0, fup=-0.1)


def test_raises_for_fup_greater_than_one():
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=300.0, fup=1.1)


def test_raises_for_fup_exactly_one():
    # fup > 1 raises; fup == 1 is at boundary — test fup > 1
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=300.0, fup=2.0)


def test_raises_for_MW_zero():
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=0.0, fup=0.2)


def test_raises_for_MW_negative():
    with pytest.raises(ValueError):
        predict_vd("Bad", logP=2.0, MW=-100.0, fup=0.2)
