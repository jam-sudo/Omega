"""Tests for Phase 401: dose proportionality power model testing."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.dose_proportionality_test import (
    DoseProportionalityResult,
    normalized_auc_test,
    power_model_test,
)

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

DOSES_4 = [10.0, 30.0, 100.0, 300.0]
DOSES_5 = [5.0, 10.0, 50.0, 100.0, 200.0]

# Perfect proportional: AUC = 8 * dose  (b = 1.0)
AUC_PROP_4 = [8.0 * d for d in DOSES_4]
AUC_PROP_5 = [8.0 * d for d in DOSES_5]

# Supra-proportional: AUC = dose^1.5
AUC_SUPRA = [d**1.5 for d in DOSES_4]

# Sub-proportional: AUC = dose^0.5
AUC_SUB = [d**0.5 for d in DOSES_4]

# Mildly supra: AUC = 5 * dose^1.15
AUC_MILD_SUPRA = [5.0 * d**1.15 for d in DOSES_5]


# ---------------------------------------------------------------------------
# DoseProportionalityResult dataclass
# ---------------------------------------------------------------------------


class TestDoseProportionalityResult:
    def test_is_frozen(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert isinstance(r, DoseProportionalityResult)
        with pytest.raises((AttributeError, TypeError)):
            r.b_exponent = 99.0  # type: ignore[misc]

    def test_has_required_fields(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        for attr in [
            "doses_mg",
            "auc_values",
            "b_exponent",
            "b_ci_lower",
            "b_ci_upper",
            "a_coefficient",
            "r_squared",
            "is_proportional",
            "proportionality_verdict",
            "cv_dn_auc_pct",
            "notes",
        ]:
            assert hasattr(r, attr)


# ---------------------------------------------------------------------------
# power_model_test — proportional scenarios
# ---------------------------------------------------------------------------


class TestPowerModelProportional:
    def test_b_close_to_1_for_proportional_data(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert abs(r.b_exponent - 1.0) < 0.001

    def test_r_squared_high_for_proportional_data(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.r_squared > 0.999

    def test_is_proportional_true_for_perfect_data(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.is_proportional is True

    def test_verdict_proportional(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert "proportional" in r.proportionality_verdict.lower()
        # Should NOT say sub or supra
        assert "supra" not in r.proportionality_verdict
        assert "sub_" not in r.proportionality_verdict

    def test_ci_contains_1_for_proportional_data(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.b_ci_lower <= 1.0 <= r.b_ci_upper

    def test_a_coefficient_reasonable(self):
        # AUC = 8*dose, so a ≈ 8
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert abs(r.a_coefficient - 8.0) < 0.1

    def test_doses_stored_as_tuple(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert isinstance(r.doses_mg, tuple)
        assert list(r.doses_mg) == DOSES_4

    def test_auc_values_stored_as_tuple(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert isinstance(r.auc_values, tuple)

    def test_five_doses_proportional(self):
        r = power_model_test(DOSES_5, AUC_PROP_5)
        assert r.is_proportional is True
        assert abs(r.b_exponent - 1.0) < 0.001


# ---------------------------------------------------------------------------
# power_model_test — supra/sub-proportional
# ---------------------------------------------------------------------------


class TestPowerModelNonProportional:
    def test_b_greater_than_1_for_supra(self):
        r = power_model_test(DOSES_4, AUC_SUPRA)
        assert r.b_exponent > 1.3

    def test_supra_verdict(self):
        r = power_model_test(DOSES_4, AUC_SUPRA)
        assert "supra" in r.proportionality_verdict

    def test_not_proportional_for_supra(self):
        r = power_model_test(DOSES_4, AUC_SUPRA)
        assert r.is_proportional is False

    def test_b_less_than_1_for_sub(self):
        r = power_model_test(DOSES_4, AUC_SUB)
        assert r.b_exponent < 0.7

    def test_sub_verdict(self):
        r = power_model_test(DOSES_4, AUC_SUB)
        assert "sub" in r.proportionality_verdict

    def test_not_proportional_for_sub(self):
        r = power_model_test(DOSES_4, AUC_SUB)
        assert r.is_proportional is False

    def test_r_squared_high_for_supra(self):
        r = power_model_test(DOSES_4, AUC_SUPRA)
        assert r.r_squared > 0.99


# ---------------------------------------------------------------------------
# power_model_test — confidence interval and statistics
# ---------------------------------------------------------------------------


class TestPowerModelCI:
    def test_ci_lower_less_than_upper(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.b_ci_lower < r.b_ci_upper

    def test_b_within_ci(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.b_ci_lower <= r.b_exponent <= r.b_ci_upper

    def test_95pct_ci_wider_than_90pct(self):
        # Use noisy supra-proportional data so residuals are non-zero and
        # the CI width is sensitive to the t critical value.
        noisy_aucs = [d**1.3 * (1.0 + 0.05 * (i % 3 - 1)) for i, d in enumerate(DOSES_4)]
        r90 = power_model_test(DOSES_4, noisy_aucs, confidence_level=0.90)
        r95 = power_model_test(DOSES_4, noisy_aucs, confidence_level=0.95)
        width90 = r90.b_ci_upper - r90.b_ci_lower
        width95 = r95.b_ci_upper - r95.b_ci_lower
        assert width95 >= width90

    def test_cv_dn_auc_near_zero_for_proportional(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert r.cv_dn_auc_pct < 1.0

    def test_cv_dn_auc_nonzero_for_supra(self):
        r = power_model_test(DOSES_4, AUC_SUPRA)
        assert r.cv_dn_auc_pct > 10.0

    def test_notes_contains_ci_info(self):
        r = power_model_test(DOSES_4, AUC_PROP_4)
        assert "CI" in r.notes or "ci" in r.notes.lower()


# ---------------------------------------------------------------------------
# power_model_test — validation
# ---------------------------------------------------------------------------


class TestPowerModelValidation:
    def test_raises_for_fewer_than_3_points(self):
        with pytest.raises(ValueError, match="3"):
            power_model_test([10.0, 30.0], [100.0, 300.0])

    def test_raises_for_mismatched_lengths(self):
        with pytest.raises(ValueError):
            power_model_test([10.0, 30.0, 100.0], [100.0, 300.0])

    def test_raises_for_nonpositive_dose(self):
        with pytest.raises(ValueError):
            power_model_test([-10.0, 30.0, 100.0], [100.0, 300.0, 1000.0])

    def test_raises_for_zero_auc(self):
        with pytest.raises(ValueError):
            power_model_test([10.0, 30.0, 100.0], [0.0, 300.0, 1000.0])

    def test_raises_for_invalid_confidence(self):
        with pytest.raises(ValueError):
            power_model_test(DOSES_4, AUC_PROP_4, confidence_level=1.5)


# ---------------------------------------------------------------------------
# normalized_auc_test
# ---------------------------------------------------------------------------


class TestNormalizedAUCTest:
    def test_proportional_gives_low_cv(self):
        result = normalized_auc_test(DOSES_4, AUC_PROP_4)
        assert result["cv_pct"] < 1.0

    def test_proportional_is_proportional_flag(self):
        result = normalized_auc_test(DOSES_4, AUC_PROP_4)
        assert result["is_proportional"] is True

    def test_supra_gives_higher_cv(self):
        result = normalized_auc_test(DOSES_4, AUC_SUPRA)
        assert result["cv_pct"] > 25.0

    def test_supra_not_proportional(self):
        result = normalized_auc_test(DOSES_4, AUC_SUPRA)
        assert result["is_proportional"] is False

    def test_mean_dn_auc_correct_for_proportional(self):
        result = normalized_auc_test(DOSES_4, AUC_PROP_4)
        # AUC = 8 * dose => dose-normalized = 8
        assert abs(result["mean_dn_auc"] - 8.0) < 0.001

    def test_dose_normalized_aucs_length(self):
        result = normalized_auc_test(DOSES_4, AUC_PROP_4)
        assert len(result["dose_normalized_aucs"]) == len(DOSES_4)

    def test_raises_for_single_point(self):
        with pytest.raises(ValueError):
            normalized_auc_test([10.0], [100.0])

    def test_raises_for_mismatched_lengths(self):
        with pytest.raises(ValueError):
            normalized_auc_test([10.0, 30.0], [100.0])

    def test_raises_for_nonpositive_dose(self):
        with pytest.raises(ValueError):
            normalized_auc_test([0.0, 30.0], [100.0, 300.0])

    def test_sub_proportional_high_cv(self):
        result = normalized_auc_test(DOSES_4, AUC_SUB)
        assert result["cv_pct"] > 25.0
