"""Tests for blood-to-plasma ratio calculator — Phase 481."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.blood_plasma_ratio import (
    RbSensitivityResult,
    classify_rb,
    correct_cl_for_rb,
    correct_vd_for_rb,
    predict_rb,
    rb_sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# predict_rb
# ---------------------------------------------------------------------------


class TestPredictRb:
    def test_high_logP_gives_high_rb(self):
        rb_low = predict_rb(logP=0.0, fu_plasma=0.1)
        rb_high = predict_rb(logP=5.0, fu_plasma=0.1)
        assert rb_high > rb_low

    def test_hydrophilic_drug_rb_less_than_one(self):
        # logP < 0 → Kp_rbc = 0.3 < 1 → Rb < 1
        rb = predict_rb(logP=-2.0, fu_plasma=0.5, hematocrit=0.45)
        assert rb < 1.0

    def test_hct_zero_gives_rb_one(self):
        # No RBCs → Rb = 1.0 regardless of logP
        rb = predict_rb(logP=5.0, fu_plasma=0.1, hematocrit=0.0)
        assert math.isclose(rb, 1.0, rel_tol=1e-9)

    def test_hct_one_rb_equals_kp_rbc(self):
        # Hct = 1 → Rb = Kp_rbc
        kp = 1.0 + 2.0 * 0.3  # logP=2 → Kp=1.6
        rb = predict_rb(logP=2.0, fu_plasma=0.5, hematocrit=1.0)
        assert math.isclose(rb, kp, rel_tol=1e-9)

    def test_explicit_rbc_partition_used(self):
        rb = predict_rb(logP=0.0, fu_plasma=0.5, hematocrit=0.45, rbc_partition=2.0)
        expected = (1.0 - 0.45) + 0.45 * 2.0
        assert math.isclose(rb, expected, rel_tol=1e-9)

    def test_explicit_rbc_partition_overrides_logP(self):
        rb_logP = predict_rb(logP=10.0, fu_plasma=0.5, hematocrit=0.45)
        rb_forced = predict_rb(
            logP=10.0, fu_plasma=0.5, hematocrit=0.45, rbc_partition=1.0
        )
        assert rb_forced != rb_logP

    def test_rb_formula_for_known_values(self):
        # Hct=0.45, logP=0 → Kp_rbc=1.0 → Rb = 0.55 + 0.45*1.0 = 1.0
        rb = predict_rb(logP=0.0, fu_plasma=0.5, hematocrit=0.45)
        assert math.isclose(rb, 1.0, rel_tol=1e-9)

    def test_invalid_hematocrit_below_zero(self):
        with pytest.raises(ValueError, match="hematocrit"):
            predict_rb(logP=1.0, fu_plasma=0.5, hematocrit=-0.1)

    def test_invalid_hematocrit_above_one(self):
        with pytest.raises(ValueError, match="hematocrit"):
            predict_rb(logP=1.0, fu_plasma=0.5, hematocrit=1.5)

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_rb(logP=1.0, fu_plasma=0.0)

    def test_invalid_fu_plasma_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_rb(logP=1.0, fu_plasma=1.5)

    def test_negative_rbc_partition_raises(self):
        with pytest.raises(ValueError, match="rbc_partition"):
            predict_rb(logP=1.0, fu_plasma=0.5, rbc_partition=-0.1)

    def test_rb_positive(self):
        for lp in [-3.0, 0.0, 2.0, 5.0]:
            rb = predict_rb(logP=lp, fu_plasma=0.2)
            assert rb > 0


# ---------------------------------------------------------------------------
# correct_cl_for_rb
# ---------------------------------------------------------------------------


class TestCorrectClForRb:
    def test_known_value(self):
        result = correct_cl_for_rb(10.0, 2.0)
        assert math.isclose(result, 5.0, rel_tol=1e-9)

    def test_rb_one_unchanged(self):
        assert math.isclose(correct_cl_for_rb(5.0, 1.0), 5.0, rel_tol=1e-9)

    def test_high_rb_reduces_cl(self):
        assert correct_cl_for_rb(10.0, 2.0) < 10.0

    def test_low_rb_increases_cl(self):
        assert correct_cl_for_rb(10.0, 0.5) > 10.0

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_plasma"):
            correct_cl_for_rb(0.0, 1.0)

    def test_invalid_rb_zero(self):
        with pytest.raises(ValueError, match="rb"):
            correct_cl_for_rb(5.0, 0.0)


# ---------------------------------------------------------------------------
# correct_vd_for_rb
# ---------------------------------------------------------------------------


class TestCorrectVdForRb:
    def test_known_value(self):
        result = correct_vd_for_rb(70.0, 2.0)
        assert math.isclose(result, 35.0, rel_tol=1e-9)

    def test_rb_one_unchanged(self):
        assert math.isclose(correct_vd_for_rb(50.0, 1.0), 50.0, rel_tol=1e-9)

    def test_high_rb_reduces_vd(self):
        assert correct_vd_for_rb(70.0, 2.0) < 70.0

    def test_invalid_vd_negative(self):
        with pytest.raises(ValueError, match="vd_plasma"):
            correct_vd_for_rb(-1.0, 1.0)

    def test_invalid_rb_negative(self):
        with pytest.raises(ValueError, match="rb"):
            correct_vd_for_rb(70.0, -1.0)


# ---------------------------------------------------------------------------
# classify_rb
# ---------------------------------------------------------------------------


class TestClassifyRb:
    def test_low(self):
        assert classify_rb(0.5) == "low"

    def test_low_boundary(self):
        assert classify_rb(0.54) == "low"

    def test_near_unity_lower(self):
        assert classify_rb(0.55) == "near_unity"

    def test_near_unity_at_one(self):
        assert classify_rb(1.0) == "near_unity"

    def test_near_unity_upper(self):
        assert classify_rb(1.3) == "near_unity"

    def test_high(self):
        assert classify_rb(2.0) == "high"

    def test_high_above_boundary(self):
        assert classify_rb(1.31) == "high"

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="rb"):
            classify_rb(0.0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="rb"):
            classify_rb(-1.0)


# ---------------------------------------------------------------------------
# rb_sensitivity_analysis
# ---------------------------------------------------------------------------


class TestRbSensitivityAnalysis:
    def test_returns_dataclass(self):
        result = rb_sensitivity_analysis([-2.0, 0.0, 2.0, 5.0])
        assert isinstance(result, RbSensitivityResult)

    def test_same_length_as_input(self):
        logPs = [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
        result = rb_sensitivity_analysis(logPs)
        assert len(result.rb_values) == len(logPs)
        assert len(result.cl_correction_factors) == len(logPs)
        assert len(result.logP_values) == len(logPs)

    def test_increasing_logP_increases_rb(self):
        logPs = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = rb_sensitivity_analysis(logPs)
        for i in range(len(result.rb_values) - 1):
            assert result.rb_values[i + 1] >= result.rb_values[i]

    def test_cl_correction_is_inverse_rb(self):
        logPs = [0.0, 2.0, 4.0]
        result = rb_sensitivity_analysis(logPs)
        for rb, cf in zip(result.rb_values, result.cl_correction_factors):
            assert math.isclose(cf, 1.0 / rb, rel_tol=1e-9)

    def test_notes_non_empty(self):
        result = rb_sensitivity_analysis([1.0])
        assert len(result.notes) > 0

    def test_custom_hematocrit(self):
        result = rb_sensitivity_analysis([2.0], hematocrit=0.60)
        assert result.rb_values[0] > 0

    def test_empty_logP_raises(self):
        with pytest.raises(ValueError):
            rb_sensitivity_analysis([])

    def test_invalid_hematocrit(self):
        with pytest.raises(ValueError, match="hematocrit"):
            rb_sensitivity_analysis([1.0], hematocrit=2.0)

    def test_invalid_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            rb_sensitivity_analysis([1.0], fu_plasma=0.0)
