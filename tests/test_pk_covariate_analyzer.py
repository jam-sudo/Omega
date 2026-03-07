"""Tests for clinical/pk_covariate_analyzer.py — Phase 478."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.pk_covariate_analyzer import (
    CovariateSensitivityResult,
    analyze_age_effect,
    analyze_renal_effect,
    analyze_weight_effect,
    covariate_sensitivity,
    power_law_covariate,
)

# ---------------------------------------------------------------------------
# power_law_covariate tests
# ---------------------------------------------------------------------------


class TestPowerLawCovariate:
    def test_at_typical_value_returns_param_typical(self):
        """When covariate_value == covariate_typical, result == param_typical."""
        result = power_law_covariate(10.0, 70.0, 70.0, 0.75)
        assert result == pytest.approx(10.0)

    def test_above_typical_positive_exponent_increases(self):
        """Value above typical with positive exponent → param increases."""
        result = power_law_covariate(10.0, 100.0, 70.0, 0.75)
        assert result > 10.0

    def test_below_typical_positive_exponent_decreases(self):
        """Value below typical with positive exponent → param decreases."""
        result = power_law_covariate(10.0, 50.0, 70.0, 0.75)
        assert result < 10.0

    def test_zero_exponent_returns_typical(self):
        """Exponent=0 → always returns param_typical regardless of covariate."""
        result = power_law_covariate(10.0, 150.0, 70.0, 0.0)
        assert result == pytest.approx(10.0)

    def test_negative_exponent_inverts(self):
        """Negative exponent → higher cov → lower param."""
        result_low = power_law_covariate(10.0, 50.0, 70.0, -0.5)
        result_high = power_law_covariate(10.0, 100.0, 70.0, -0.5)
        assert result_low > result_high

    def test_power_law_correctness(self):
        """P = P_typ * (2/1)^0.75 at cov=2, cov_typ=1."""
        result = power_law_covariate(1.0, 2.0, 1.0, 0.75)
        expected = 2.0**0.75
        assert result == pytest.approx(expected, rel=1e-6)

    def test_invalid_param_typical_raises(self):
        with pytest.raises(ValueError, match="param_typical must be > 0"):
            power_law_covariate(0.0, 70.0, 70.0, 0.75)

    def test_invalid_covariate_value_raises(self):
        with pytest.raises(ValueError, match="covariate_value must be > 0"):
            power_law_covariate(10.0, 0.0, 70.0, 0.75)

    def test_invalid_covariate_typical_raises(self):
        with pytest.raises(ValueError, match="covariate_typical must be > 0"):
            power_law_covariate(10.0, 70.0, 0.0, 0.75)


# ---------------------------------------------------------------------------
# analyze_weight_effect tests
# ---------------------------------------------------------------------------


class TestAnalyzeWeightEffect:
    def test_typical_weight_returns_typical_params(self):
        """70 kg patient → CL and Vd equal to typical values."""
        result = analyze_weight_effect(10.0, 50.0, 70.0)
        assert result["cl_individual"] == pytest.approx(10.0, rel=1e-6)
        assert result["vd_individual"] == pytest.approx(50.0, rel=1e-6)

    def test_heavier_patient_higher_cl(self):
        """Weight > 70 kg, positive exponent → CL increases."""
        result = analyze_weight_effect(10.0, 50.0, 100.0)
        assert result["cl_individual"] > 10.0

    def test_lighter_patient_lower_cl(self):
        """Weight < 70 kg → CL decreases."""
        result = analyze_weight_effect(10.0, 50.0, 50.0)
        assert result["cl_individual"] < 10.0

    def test_cl_ratio_at_typical_weight_is_one(self):
        result = analyze_weight_effect(10.0, 50.0, 70.0)
        assert result["cl_ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_vd_ratio_proportional_to_weight(self):
        """Vd_exponent=1.0 → Vd_ratio = weight/70."""
        result = analyze_weight_effect(10.0, 50.0, 140.0)
        assert result["vd_ratio"] == pytest.approx(2.0, rel=1e-6)

    def test_t_half_computed(self):
        result = analyze_weight_effect(10.0, 50.0, 70.0)
        expected_t_half = math.log(2) * 50.0 / 10.0
        assert result["t_half_h"] == pytest.approx(expected_t_half, rel=1e-6)

    def test_invalid_cl_typical_raises(self):
        with pytest.raises(ValueError, match="cl_typical must be > 0"):
            analyze_weight_effect(0.0, 50.0, 70.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg must be > 0"):
            analyze_weight_effect(10.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# analyze_renal_effect tests
# ---------------------------------------------------------------------------


class TestAnalyzeRenalEffect:
    def test_normal_egfr_returns_typical_cl(self):
        """eGFR=100 (normal) → CL_individual == CL_typical."""
        result = analyze_renal_effect(10.0, 100.0, egfr_normal=100.0)
        assert result["cl_individual"] == pytest.approx(10.0, rel=1e-6)

    def test_egfr_50_reduces_cl(self):
        """eGFR=50 (50% of normal), cl_renal_fraction=0.5 → CL reduced by 25%."""
        result = analyze_renal_effect(10.0, 50.0, egfr_normal=100.0, cl_renal_fraction=0.5)
        # CL_nonrenal = 5, CL_renal_comp = 5 * 0.5 = 2.5; total = 7.5
        assert result["cl_individual"] == pytest.approx(7.5, rel=1e-6)

    def test_egfr_zero_removes_renal_cl(self):
        """eGFR=0 → only non-renal clearance remains."""
        result = analyze_renal_effect(10.0, 0.0, cl_renal_fraction=0.5)
        assert result["cl_individual"] == pytest.approx(5.0, rel=1e-6)

    def test_cl_ratio_below_one_with_low_egfr(self):
        result = analyze_renal_effect(10.0, 30.0, cl_renal_fraction=0.8)
        assert result["cl_ratio"] < 1.0

    def test_renal_impairment_category_normal(self):
        result = analyze_renal_effect(10.0, 100.0)
        assert result["renal_impairment_category"] == "normal"

    def test_renal_impairment_category_mild(self):
        result = analyze_renal_effect(10.0, 75.0)
        assert result["renal_impairment_category"] == "mild"

    def test_renal_impairment_category_severe(self):
        result = analyze_renal_effect(10.0, 20.0)
        assert result["renal_impairment_category"] == "severe"

    def test_renal_impairment_category_failure(self):
        result = analyze_renal_effect(10.0, 10.0)
        assert result["renal_impairment_category"] == "kidney_failure"

    def test_invalid_egfr_negative_raises(self):
        with pytest.raises(ValueError, match="egfr_mL_per_min must be >= 0"):
            analyze_renal_effect(10.0, -1.0)

    def test_invalid_cl_renal_fraction_raises(self):
        with pytest.raises(ValueError, match="cl_renal_fraction must be between 0 and 1"):
            analyze_renal_effect(10.0, 80.0, cl_renal_fraction=1.5)

    def test_zero_renal_fraction_no_renal_effect(self):
        """cl_renal_fraction=0 → eGFR has no effect on CL."""
        result_low = analyze_renal_effect(10.0, 10.0, cl_renal_fraction=0.0)
        result_high = analyze_renal_effect(10.0, 100.0, cl_renal_fraction=0.0)
        assert result_low["cl_individual"] == pytest.approx(result_high["cl_individual"])


# ---------------------------------------------------------------------------
# analyze_age_effect tests
# ---------------------------------------------------------------------------


class TestAnalyzeAgeEffect:
    def test_typical_age_returns_typical_params(self):
        """Age=35 (typical) → CL_individual == CL_typical."""
        result = analyze_age_effect(10.0, 50.0, 35.0)
        assert result["cl_individual"] == pytest.approx(10.0, rel=1e-6)

    def test_elderly_lower_cl_than_adult(self):
        """Age=70 → lower CL than age=35."""
        result_adult = analyze_age_effect(10.0, 50.0, 35.0)
        result_elderly = analyze_age_effect(10.0, 50.0, 70.0)
        assert result_elderly["cl_individual"] < result_adult["cl_individual"]

    def test_young_adult_flat_cl(self):
        """Age < 35 → same CL as typical (flat region)."""
        result_young = analyze_age_effect(10.0, 50.0, 25.0)
        result_typical = analyze_age_effect(10.0, 50.0, 35.0)
        assert result_young["cl_individual"] == pytest.approx(result_typical["cl_individual"])

    def test_age_category_pediatric(self):
        result = analyze_age_effect(10.0, 50.0, 10.0)
        assert result["age_category"] == "pediatric"

    def test_age_category_adult(self):
        result = analyze_age_effect(10.0, 50.0, 40.0)
        assert result["age_category"] == "adult"

    def test_age_category_elderly(self):
        result = analyze_age_effect(10.0, 50.0, 70.0)
        assert result["age_category"] == "elderly"

    def test_cl_ratio_less_than_one_for_elderly(self):
        result = analyze_age_effect(10.0, 50.0, 80.0)
        assert result["cl_ratio"] < 1.0

    def test_t_half_h_computed(self):
        result = analyze_age_effect(10.0, 50.0, 35.0)
        expected = math.log(2) * 50.0 / 10.0
        assert result["t_half_h"] == pytest.approx(expected, rel=1e-6)

    def test_invalid_age_raises(self):
        with pytest.raises(ValueError, match="age_years must be > 0"):
            analyze_age_effect(10.0, 50.0, 0.0)

    def test_invalid_cl_typical_raises(self):
        with pytest.raises(ValueError, match="cl_typical must be > 0"):
            analyze_age_effect(0.0, 50.0, 35.0)


# ---------------------------------------------------------------------------
# covariate_sensitivity tests
# ---------------------------------------------------------------------------


class TestCovariateSensitivity:
    def test_returns_covariate_sensitivity_result(self):
        result = covariate_sensitivity(10.0, [50.0, 70.0, 100.0], 70.0, 0.75)
        assert isinstance(result, CovariateSensitivityResult)

    def test_at_typical_value_ratio_is_one(self):
        """When covariate_range contains only the typical value, ratio = 1."""
        result = covariate_sensitivity(10.0, [70.0, 70.0], 70.0, 0.75)
        for r in result.cl_ratio_to_typical:
            assert r == pytest.approx(1.0, rel=1e-6)

    def test_wide_range_large_fold_change(self):
        """Wide covariate range → large max_fold_change."""
        result = covariate_sensitivity(10.0, [10.0, 200.0], 70.0, 0.75)
        assert result.max_fold_change > 2.0

    def test_narrow_range_small_fold_change(self):
        """Narrow range near typical → small fold change."""
        result = covariate_sensitivity(10.0, [68.0, 70.0, 72.0], 70.0, 0.75)
        assert result.max_fold_change < 1.1

    def test_dose_adjustment_needed_true(self):
        """max_fold_change > 2 → dose_adjustment_needed=True."""
        result = covariate_sensitivity(10.0, [10.0, 200.0], 70.0, 0.75)
        assert result.dose_adjustment_needed is True

    def test_dose_adjustment_needed_false(self):
        """Narrow range → dose_adjustment_needed=False."""
        result = covariate_sensitivity(10.0, [60.0, 80.0], 70.0, 0.75)
        assert result.dose_adjustment_needed is False

    def test_cl_values_positive(self):
        result = covariate_sensitivity(10.0, [40.0, 70.0, 120.0], 70.0, 0.75)
        for cl in result.cl_values:
            assert cl > 0

    def test_notes_not_empty(self):
        result = covariate_sensitivity(10.0, [40.0, 70.0, 120.0], 70.0, 0.75)
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_tuples_same_length(self):
        cov_range = [40.0, 55.0, 70.0, 90.0, 120.0]
        result = covariate_sensitivity(10.0, cov_range, 70.0, 0.75)
        n = len(cov_range)
        assert len(result.covariate_values) == n
        assert len(result.cl_values) == n
        assert len(result.cl_ratio_to_typical) == n

    def test_invalid_cl_typical_raises(self):
        with pytest.raises(ValueError, match="cl_typical must be > 0"):
            covariate_sensitivity(0.0, [70.0, 100.0])

    def test_invalid_covariate_typical_raises(self):
        with pytest.raises(ValueError, match="covariate_typical must be > 0"):
            covariate_sensitivity(10.0, [70.0, 100.0], covariate_typical=0.0)

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="covariate_range must not be empty"):
            covariate_sensitivity(10.0, [])

    def test_negative_value_in_range_raises(self):
        with pytest.raises(ValueError, match="all values in covariate_range must be > 0"):
            covariate_sensitivity(10.0, [-10.0, 70.0])

    def test_single_value_raises(self):
        with pytest.raises(ValueError, match="covariate_range must contain at least 2 values"):
            covariate_sensitivity(10.0, [70.0])
