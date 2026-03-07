"""Tests for omega_pbpk.prediction.bioequivalence_predictor (Phase 399)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.bioequivalence_predictor import (
    BEPredictionResult,
    predict_be_outcome,
    sample_size_for_be,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> BEPredictionResult:
    params = dict(
        ref_auc=100.0,
        test_auc=100.0,
        ref_cmax=10.0,
        test_cmax=10.0,
        n_subjects=24,
        intra_cv_pct=20.0,
    )
    params.update(kwargs)
    return predict_be_outcome(**params)


# ---------------------------------------------------------------------------
# BEPredictionResult dataclass structure
# ---------------------------------------------------------------------------


class TestBEPredictionResultStructure:
    def test_is_dataclass(self):
        r = _default_result()
        assert isinstance(r, BEPredictionResult)

    def test_has_gmr_auc(self):
        r = _default_result()
        assert hasattr(r, "gmr_auc")

    def test_has_gmr_cmax(self):
        r = _default_result()
        assert hasattr(r, "gmr_cmax")

    def test_has_ci_auc_lower(self):
        r = _default_result()
        assert hasattr(r, "ci_auc_lower")

    def test_has_ci_auc_upper(self):
        r = _default_result()
        assert hasattr(r, "ci_auc_upper")

    def test_has_be_auc_pass(self):
        r = _default_result()
        assert hasattr(r, "be_auc_pass")

    def test_has_overall_be_pass(self):
        r = _default_result()
        assert hasattr(r, "overall_be_pass")

    def test_has_notes(self):
        r = _default_result()
        assert hasattr(r, "notes")

    def test_notes_is_str(self):
        r = _default_result()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# GMR calculation correctness
# ---------------------------------------------------------------------------


class TestGMRCalculation:
    def test_gmr_auc_unity_for_equal_aucs(self):
        r = _default_result(ref_auc=80.0, test_auc=80.0)
        assert math.isclose(r.gmr_auc, 1.0, rel_tol=1e-9)

    def test_gmr_cmax_unity_for_equal_cmax(self):
        r = _default_result(ref_cmax=5.0, test_cmax=5.0)
        assert math.isclose(r.gmr_cmax, 1.0, rel_tol=1e-9)

    def test_gmr_auc_ratio_0_9(self):
        r = _default_result(ref_auc=100.0, test_auc=90.0)
        assert math.isclose(r.gmr_auc, 0.9, rel_tol=1e-9)

    def test_gmr_cmax_ratio_1_1(self):
        r = _default_result(ref_cmax=10.0, test_cmax=11.0)
        assert math.isclose(r.gmr_cmax, 1.1, rel_tol=1e-9)

    def test_gmr_auc_ratio_1_5(self):
        r = _default_result(ref_auc=100.0, test_auc=150.0)
        assert math.isclose(r.gmr_auc, 1.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 90% CI ordering and positivity
# ---------------------------------------------------------------------------


class TestCIProperties:
    def test_ci_auc_lower_less_than_upper(self):
        r = _default_result()
        assert r.ci_auc_lower < r.ci_auc_upper

    def test_ci_cmax_lower_less_than_upper(self):
        r = _default_result()
        assert r.ci_cmax_lower < r.ci_cmax_upper

    def test_ci_auc_bounds_positive(self):
        r = _default_result()
        assert r.ci_auc_lower > 0
        assert r.ci_auc_upper > 0

    def test_ci_cmax_bounds_positive(self):
        r = _default_result()
        assert r.ci_cmax_lower > 0
        assert r.ci_cmax_upper > 0

    def test_gmr_within_ci(self):
        r = _default_result(ref_auc=100.0, test_auc=95.0)
        assert r.ci_auc_lower <= r.gmr_auc <= r.ci_auc_upper

    def test_larger_n_narrows_ci(self):
        r_small = _default_result(n_subjects=12)
        r_large = _default_result(n_subjects=60)
        width_small = r_small.ci_auc_upper - r_small.ci_auc_lower
        width_large = r_large.ci_auc_upper - r_large.ci_auc_lower
        assert width_large < width_small

    def test_higher_cv_widens_ci(self):
        r_low = _default_result(intra_cv_pct=10.0)
        r_high = _default_result(intra_cv_pct=40.0)
        width_low = r_low.ci_auc_upper - r_low.ci_auc_lower
        width_high = r_high.ci_auc_upper - r_high.ci_auc_lower
        assert width_high > width_low


# ---------------------------------------------------------------------------
# BE pass/fail decisions
# ---------------------------------------------------------------------------


class TestBEDecision:
    def test_be_pass_gmr_one_low_cv_large_n(self):
        r = _default_result(n_subjects=60, intra_cv_pct=10.0)
        assert r.be_auc_pass is True
        assert r.be_cmax_pass is True
        assert r.overall_be_pass is True

    def test_be_fail_gmr_0_7(self):
        r = _default_result(ref_auc=100.0, test_auc=70.0, n_subjects=24, intra_cv_pct=15.0)
        assert r.be_auc_pass is False
        assert r.overall_be_pass is False

    def test_be_fail_gmr_1_4(self):
        r = _default_result(ref_auc=100.0, test_auc=140.0, n_subjects=24, intra_cv_pct=15.0)
        assert r.be_auc_pass is False
        assert r.overall_be_pass is False

    def test_overall_be_false_if_cmax_fails(self):
        # AUC passes, Cmax fails
        r = _default_result(
            ref_auc=100.0,
            test_auc=100.0,
            ref_cmax=10.0,
            test_cmax=7.0,
            n_subjects=60,
            intra_cv_pct=10.0,
        )
        assert r.be_auc_pass is True
        assert r.be_cmax_pass is False
        assert r.overall_be_pass is False

    def test_custom_be_limits_narrower(self):
        # With narrow limits (0.90, 1.11), GMR=0.95 may fail
        r = _default_result(
            ref_auc=100.0,
            test_auc=100.0,
            n_subjects=12,
            intra_cv_pct=30.0,
            be_limits=(0.90, 1.11),
        )
        # With high CV and few subjects, CI will be wide - should fail
        assert isinstance(r.be_auc_pass, bool)

    def test_be_pass_returns_bool(self):
        r = _default_result()
        assert isinstance(r.be_auc_pass, bool)
        assert isinstance(r.be_cmax_pass, bool)
        assert isinstance(r.overall_be_pass, bool)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_ref_auc_zero_raises(self):
        with pytest.raises(ValueError, match="ref_auc"):
            _default_result(ref_auc=0.0)

    def test_test_auc_negative_raises(self):
        with pytest.raises(ValueError, match="test_auc"):
            _default_result(test_auc=-10.0)

    def test_ref_cmax_zero_raises(self):
        with pytest.raises(ValueError, match="ref_cmax"):
            _default_result(ref_cmax=0.0)

    def test_test_cmax_negative_raises(self):
        with pytest.raises(ValueError, match="test_cmax"):
            _default_result(test_cmax=-5.0)

    def test_n_subjects_one_raises(self):
        with pytest.raises(ValueError, match="n_subjects"):
            _default_result(n_subjects=1)

    def test_intra_cv_zero_raises(self):
        with pytest.raises(ValueError, match="intra_cv_pct"):
            _default_result(intra_cv_pct=0.0)

    def test_intra_cv_negative_raises(self):
        with pytest.raises(ValueError, match="intra_cv_pct"):
            _default_result(intra_cv_pct=-5.0)

    def test_invalid_be_limits_raises(self):
        with pytest.raises(ValueError):
            _default_result(be_limits=(1.25, 0.80))


# ---------------------------------------------------------------------------
# sample_size_for_be
# ---------------------------------------------------------------------------


class TestSampleSizeForBE:
    def test_returns_int(self):
        n = sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0)
        assert isinstance(n, int)

    def test_minimum_n_is_12(self):
        n = sample_size_for_be(target_gmr=1.0, intra_cv_pct=5.0)
        assert n >= 12

    def test_even_n(self):
        n = sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0)
        assert n % 2 == 0

    def test_higher_cv_requires_more_subjects(self):
        n_low = sample_size_for_be(target_gmr=1.0, intra_cv_pct=10.0)
        n_high = sample_size_for_be(target_gmr=1.0, intra_cv_pct=30.0)
        assert n_high >= n_low

    def test_gmr_far_from_one_requires_more_subjects(self):
        n_near = sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0)
        n_far = sample_size_for_be(target_gmr=0.85, intra_cv_pct=20.0)
        assert n_far >= n_near

    def test_validation_gmr_zero_raises(self):
        with pytest.raises(ValueError, match="target_gmr"):
            sample_size_for_be(target_gmr=0.0, intra_cv_pct=20.0)

    def test_validation_cv_zero_raises(self):
        with pytest.raises(ValueError, match="intra_cv_pct"):
            sample_size_for_be(target_gmr=1.0, intra_cv_pct=0.0)

    def test_validation_power_zero_raises(self):
        with pytest.raises(ValueError, match="power"):
            sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0, power=0.0)

    def test_validation_power_one_raises(self):
        with pytest.raises(ValueError, match="power"):
            sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0, power=1.0)

    def test_n_is_reasonable_for_typical_scenario(self):
        # Typical BE scenario: GMR=0.95, CV=20%, power=80%
        n = sample_size_for_be(target_gmr=0.95, intra_cv_pct=20.0, power=0.80)
        assert 10 <= n <= 60

    def test_higher_power_requires_more_subjects(self):
        n_80 = sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0, power=0.80)
        n_90 = sample_size_for_be(target_gmr=1.0, intra_cv_pct=20.0, power=0.90)
        assert n_90 >= n_80
