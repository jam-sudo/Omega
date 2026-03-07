"""Tests for hepatic clearance scaling (Phase 533)."""

import pytest

from omega_pbpk.clinical.hepatic_clearance_scaling import (
    child_pugh_hepatic_adjustment,
    compare_hepatic_models,
    dispersion_model,
    parallel_tube_model,
    well_stirred_model,
)


class TestWellStirredModel:
    def test_returns_expected_keys(self):
        result = well_stirred_model(10.0)
        for key in (
            "clint_whole_liver_L_per_h",
            "clh_L_per_h",
            "extraction_ratio",
            "f_available",
            "qh_L_per_h",
        ):
            assert key in result

    def test_high_clint_er_approaches_one(self):
        # Very high CLint → high extraction ratio approaching 1
        result = well_stirred_model(10000.0, qh_L_per_h=90.0, fu_plasma=1.0)
        assert result["extraction_ratio"] > 0.95
        assert result["clh_L_per_h"] < result["qh_L_per_h"] + 0.01

    def test_clh_never_exceeds_qh(self):
        result = well_stirred_model(50000.0, qh_L_per_h=90.0)
        assert result["clh_L_per_h"] <= 90.0 + 1e-9

    def test_low_clint_linear_region(self):
        # Very low CLint → CLh ≈ fu * CLint (linear region)
        clint = 0.001  # very low
        result = well_stirred_model(clint, qh_L_per_h=90.0, fu_plasma=0.5, rbp=1.0)
        # In linear region: CLh ≈ fu_b * CLint_total
        fu_b = 0.5 / 1.0
        clint_total = clint * 1e-6 * 60 * 39.0 * 1000
        expected_clh = fu_b * clint_total
        assert abs(result["clh_L_per_h"] - expected_clh) / (expected_clh + 1e-9) < 0.01

    def test_f_available_plus_er_equals_one(self):
        result = well_stirred_model(50.0)
        assert abs(result["f_available"] + result["extraction_ratio"] - 1.0) < 1e-10

    def test_fu_plasma_reduces_clh(self):
        result_high_fu = well_stirred_model(10.0, fu_plasma=1.0)
        result_low_fu = well_stirred_model(10.0, fu_plasma=0.1)
        assert result_high_fu["clh_L_per_h"] > result_low_fu["clh_L_per_h"]

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            well_stirred_model(10.0, fu_plasma=0.0)

    def test_invalid_fu_plasma_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            well_stirred_model(10.0, fu_plasma=1.5)

    def test_invalid_rbp_zero(self):
        with pytest.raises(ValueError, match="rbp"):
            well_stirred_model(10.0, rbp=0.0)

    def test_qh_returned_correctly(self):
        result = well_stirred_model(10.0, qh_L_per_h=75.0)
        assert result["qh_L_per_h"] == 75.0


class TestParallelTubeModel:
    def test_returns_expected_keys(self):
        result = parallel_tube_model(10.0)
        for key in (
            "clint_whole_liver_L_per_h",
            "clh_L_per_h",
            "extraction_ratio",
            "f_available",
            "qh_L_per_h",
        ):
            assert key in result

    def test_clh_never_exceeds_qh(self):
        result = parallel_tube_model(50000.0, qh_L_per_h=90.0)
        assert result["clh_L_per_h"] <= 90.0 + 1e-9

    def test_high_clint_approaches_qh(self):
        result = parallel_tube_model(10000.0, qh_L_per_h=90.0)
        assert result["extraction_ratio"] > 0.95

    def test_f_available_plus_er_equals_one(self):
        result = parallel_tube_model(30.0)
        assert abs(result["f_available"] + result["extraction_ratio"] - 1.0) < 1e-10

    def test_invalid_rbp(self):
        with pytest.raises(ValueError, match="rbp"):
            parallel_tube_model(10.0, rbp=-1.0)


class TestDispersionModel:
    def test_returns_expected_keys(self):
        result = dispersion_model(10.0)
        for key in (
            "clint_whole_liver_L_per_h",
            "clh_L_per_h",
            "extraction_ratio",
            "f_available",
            "qh_L_per_h",
        ):
            assert key in result

    def test_clh_physically_bounded(self):
        result = dispersion_model(50.0)
        assert 0.0 <= result["clh_L_per_h"] <= result["qh_L_per_h"] + 1e-9

    def test_extraction_ratio_bounded(self):
        result = dispersion_model(100.0, fu_plasma=0.5)
        assert 0.0 <= result["extraction_ratio"] <= 1.0

    def test_f_available_plus_er_equals_one(self):
        result = dispersion_model(25.0)
        assert abs(result["f_available"] + result["extraction_ratio"] - 1.0) < 1e-10

    def test_invalid_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            dispersion_model(10.0, fu_plasma=0.0)

    def test_high_clint_high_er(self):
        result = dispersion_model(10000.0, qh_L_per_h=90.0)
        assert result["extraction_ratio"] > 0.5


class TestCompareHepaticModels:
    def test_all_three_models_present(self):
        result = compare_hepatic_models(20.0)
        assert "well_stirred" in result
        assert "parallel_tube" in result
        assert "dispersion" in result

    def test_clh_range_is_tuple(self):
        result = compare_hepatic_models(20.0)
        assert "clh_range" in result
        lo, hi = result["clh_range"]
        assert lo <= hi

    def test_clh_range_contains_all_models(self):
        result = compare_hepatic_models(20.0)
        lo, hi = result["clh_range"]
        for model_key in ("well_stirred", "parallel_tube", "dispersion"):
            clh = result[model_key]["clh_L_per_h"]
            assert lo <= clh <= hi + 1e-9

    def test_compare_with_low_fu(self):
        result = compare_hepatic_models(10.0, fu_plasma=0.05, rbp=1.0)
        assert result["clh_range"][0] >= 0.0


class TestChildPughHepaticAdjustment:
    def test_class_a_mild_reduction(self):
        result = child_pugh_hepatic_adjustment(50.0, 5)
        assert result["class_label"] == "A"
        assert abs(result["cl_adjusted_L_per_h"] - 40.0) < 1e-9
        assert result["dose_factor"] == 0.8

    def test_class_a_score_6(self):
        result = child_pugh_hepatic_adjustment(50.0, 6)
        assert result["class_label"] == "A"

    def test_class_b_score_7(self):
        result = child_pugh_hepatic_adjustment(50.0, 7)
        assert result["class_label"] == "B"
        assert abs(result["cl_adjusted_L_per_h"] - 25.0) < 1e-9

    def test_class_b_score_9(self):
        result = child_pugh_hepatic_adjustment(50.0, 9)
        assert result["class_label"] == "B"

    def test_class_c_severe_reduction(self):
        result = child_pugh_hepatic_adjustment(100.0, 10)
        assert result["class_label"] == "C"
        assert abs(result["cl_adjusted_L_per_h"] - 25.0) < 1e-9
        assert result["dose_factor"] == 0.25

    def test_class_c_max_score(self):
        result = child_pugh_hepatic_adjustment(100.0, 15)
        assert result["class_label"] == "C"

    def test_recommendation_present(self):
        result = child_pugh_hepatic_adjustment(50.0, 8)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_invalid_score_too_low(self):
        with pytest.raises(ValueError, match="child_pugh_score"):
            child_pugh_hepatic_adjustment(50.0, 4)

    def test_invalid_score_too_high(self):
        with pytest.raises(ValueError, match="child_pugh_score"):
            child_pugh_hepatic_adjustment(50.0, 16)
