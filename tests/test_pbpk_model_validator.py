"""Tests for Phase 512: PBPK Model Validator."""

from __future__ import annotations

import pytest

from omega_pbpk.validation.pbpk_model_validator import (
    PKValidationResult,
    acceptance_criteria_check,
    calculate_aafe,
    calculate_afe,
    calculate_mfe,
    validate_pk_profile,
    within_fold,
)

# ---------------------------------------------------------------------------
# calculate_aafe
# ---------------------------------------------------------------------------


class TestCalculateAAFE:
    def test_perfect_prediction(self):
        """Perfect predictions → AAFE = 1.0."""
        obs = [1.0, 2.0, 4.0]
        pred = [1.0, 2.0, 4.0]
        assert calculate_aafe(obs, pred) == pytest.approx(1.0)

    def test_two_fold_over(self):
        """All 2× over-prediction → AAFE = 2.0."""
        obs = [1.0, 2.0, 4.0]
        pred = [2.0, 4.0, 8.0]
        assert calculate_aafe(obs, pred) == pytest.approx(2.0)

    def test_two_fold_under(self):
        """All 2× under-prediction → AAFE = 2.0 (absolute)."""
        obs = [2.0, 4.0, 8.0]
        pred = [1.0, 2.0, 4.0]
        assert calculate_aafe(obs, pred) == pytest.approx(2.0)

    def test_geometric_mean(self):
        """AAFE uses geometric mean of fold errors."""
        # 10× and 0.1× → geometric mean of |log10| = (1+1)/2 = 1 → 10^1 = 10
        obs = [1.0, 10.0]
        pred = [10.0, 1.0]
        assert calculate_aafe(obs, pred) == pytest.approx(10.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_aafe([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_aafe([], [])

    def test_zero_observed_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_aafe([0.0, 1.0], [1.0, 1.0])

    def test_negative_predicted_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_aafe([1.0], [-1.0])


# ---------------------------------------------------------------------------
# calculate_afe
# ---------------------------------------------------------------------------


class TestCalculateAFE:
    def test_perfect_prediction(self):
        obs = [1.0, 2.0, 3.0]
        pred = [1.0, 2.0, 3.0]
        assert calculate_afe(obs, pred) == pytest.approx(1.0)

    def test_two_fold_over_prediction(self):
        """All 2× over → AFE = 2.0."""
        obs = [1.0, 2.0]
        pred = [2.0, 4.0]
        assert calculate_afe(obs, pred) == pytest.approx(2.0)

    def test_two_fold_under_prediction(self):
        """All 2× under → AFE = 0.5."""
        obs = [2.0, 4.0]
        pred = [1.0, 2.0]
        assert calculate_afe(obs, pred) == pytest.approx(0.5)

    def test_mixed_bias(self):
        """Opposite fold errors cancel out for AFE."""
        obs = [1.0, 4.0]
        pred = [4.0, 1.0]  # log10 ratios: +0.602, -0.602 → mean = 0 → AFE = 1.0
        assert calculate_afe(obs, pred) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# calculate_mfe
# ---------------------------------------------------------------------------


class TestCalculateMFE:
    def test_perfect(self):
        obs = [1.0, 2.0]
        pred = [1.0, 2.0]
        assert calculate_mfe(obs, pred) == pytest.approx(1.0)

    def test_two_fold_over(self):
        obs = [1.0, 2.0]
        pred = [2.0, 4.0]
        assert calculate_mfe(obs, pred) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# within_fold
# ---------------------------------------------------------------------------


class TestWithinFold:
    def test_perfect_all_within_2fold(self):
        obs = [1.0, 2.0, 4.0]
        pred = [1.0, 2.0, 4.0]
        assert within_fold(obs, pred) == pytest.approx(1.0)

    def test_half_within_2fold(self):
        """50% within 2-fold."""
        obs = [1.0, 1.0]
        pred = [1.5, 3.0]  # 1.5: within 2-fold; 3.0: outside 2-fold
        frac = within_fold(obs, pred)
        assert frac == pytest.approx(0.5)

    def test_none_within_2fold(self):
        obs = [1.0, 1.0]
        pred = [5.0, 10.0]
        assert within_fold(obs, pred) == pytest.approx(0.0)

    def test_custom_fold_3(self):
        """All predictions within 3-fold."""
        obs = [1.0, 1.0, 1.0]
        pred = [2.5, 0.4, 1.0]  # 2.5 < 3, 0.4 > 1/3, 1.0 perfect
        frac = within_fold(obs, pred, fold=3.0)
        assert frac == pytest.approx(1.0)

    def test_fold_must_be_gt_1(self):
        with pytest.raises(ValueError, match="fold must be"):
            within_fold([1.0], [1.0], fold=1.0)


# ---------------------------------------------------------------------------
# validate_pk_profile
# ---------------------------------------------------------------------------


class TestValidatePkProfile:
    def _perfect_profile(self):
        times = [0.5, 1.0, 2.0, 4.0, 8.0]
        concs = [10.0, 8.0, 5.0, 2.0, 0.5]
        return times, concs

    def test_perfect_prediction_aafe(self):
        """Identical observed and predicted → AAFE = 1.0."""
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.aafe == pytest.approx(1.0, rel=1e-6)

    def test_perfect_prediction_within_2fold(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.within_2fold_pct == pytest.approx(100.0)

    def test_perfect_prediction_r2(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_two_fold_over_aafe(self):
        """2× over-prediction → AAFE = 2.0."""
        t = [1.0, 2.0, 4.0]
        obs = [1.0, 0.5, 0.25]
        pred_concs = [2.0, 1.0, 0.5]
        result = validate_pk_profile(t, obs, t, pred_concs)
        assert result.aafe == pytest.approx(2.0, rel=1e-4)

    def test_validation_passed_perfect(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.validation_passed is True

    def test_validation_failed_high_aafe(self):
        """10× over-prediction → AAFE = 10.0 → validation fails."""
        t = [1.0, 2.0, 4.0]
        obs = [1.0, 1.0, 1.0]
        pred = [10.0, 10.0, 10.0]
        result = validate_pk_profile(t, obs, t, pred)
        assert result.validation_passed is False

    def test_model_bias_over_prediction(self):
        """AFE > 1.05 → 'over-prediction'."""
        t = [1.0, 2.0, 3.0]
        obs = [1.0, 1.0, 1.0]
        pred = [2.0, 2.0, 2.0]
        result = validate_pk_profile(t, obs, t, pred)
        assert result.model_bias == "over-prediction"

    def test_model_bias_under_prediction(self):
        t = [1.0, 2.0, 3.0]
        obs = [2.0, 2.0, 2.0]
        pred = [1.0, 1.0, 1.0]
        result = validate_pk_profile(t, obs, t, pred)
        assert result.model_bias == "under-prediction"

    def test_model_bias_balanced(self):
        t = [1.0, 2.0]
        obs = [1.0, 1.0]
        pred = [1.0, 1.0]
        result = validate_pk_profile(t, obs, t, pred)
        assert result.model_bias == "balanced"

    def test_n_points(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.n_points == len(t)

    def test_interpolation_denser_predicted(self):
        """Predicted on dense grid, observed at subset — should interpolate."""
        pred_t = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        pred_c = [0.01, 5.0, 8.0, 6.0, 3.0, 0.8]
        obs_t = [1.0, 2.0, 4.0]
        obs_c = [8.0, 6.0, 3.0]
        result = validate_pk_profile(obs_t, obs_c, pred_t, pred_c)
        assert result.aafe == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_obs_length_raises(self):
        with pytest.raises(ValueError):
            validate_pk_profile([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])

    def test_mismatched_pred_length_raises(self):
        with pytest.raises(ValueError):
            validate_pk_profile([1.0], [1.0], [1.0, 2.0], [1.0])

    def test_zero_observed_raises(self):
        with pytest.raises(ValueError):
            validate_pk_profile([1.0], [0.0], [1.0], [1.0])

    def test_negative_predicted_raises(self):
        with pytest.raises(ValueError):
            validate_pk_profile([1.0], [1.0], [1.0], [-1.0])

    def test_returns_dataclass(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert isinstance(result, PKValidationResult)

    def test_rmse_zero_perfect(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)

    def test_mean_ratio_perfect(self):
        t, c = self._perfect_profile()
        result = validate_pk_profile(t, c, t, c)
        assert result.mean_pred_obs_ratio == pytest.approx(1.0)

    def test_within_3fold_pct(self):
        t = [1.0, 2.0, 4.0]
        obs = [1.0, 1.0, 1.0]
        pred = [2.5, 2.5, 2.5]  # 2.5-fold: within 3-fold but not 2-fold
        result = validate_pk_profile(t, obs, t, pred)
        assert result.within_3fold_pct == pytest.approx(100.0)
        assert result.within_2fold_pct < 100.0


# ---------------------------------------------------------------------------
# acceptance_criteria_check
# ---------------------------------------------------------------------------


class TestAcceptanceCriteriaCheck:
    def test_perfect_passes(self):
        result = acceptance_criteria_check(aafe=1.0, afe=1.0, within_2fold_pct=100.0)
        assert result["passed"] is True

    def test_high_aafe_fails(self):
        result = acceptance_criteria_check(aafe=2.5, afe=1.0, within_2fold_pct=80.0)
        assert result["passed"] is False
        assert result["aafe_passed"] is False

    def test_low_within_2fold_fails(self):
        result = acceptance_criteria_check(aafe=1.5, afe=1.0, within_2fold_pct=40.0)
        assert result["passed"] is False
        assert result["within_2fold_passed"] is False

    def test_both_criteria_pass(self):
        result = acceptance_criteria_check(aafe=1.8, afe=1.2, within_2fold_pct=60.0)
        assert result["passed"] is True
        assert result["aafe_passed"] is True
        assert result["within_2fold_passed"] is True

    def test_bias_acceptable(self):
        result = acceptance_criteria_check(aafe=1.0, afe=1.5, within_2fold_pct=100.0)
        assert result["bias_acceptable"] is True

    def test_bias_not_acceptable(self):
        result = acceptance_criteria_check(aafe=1.0, afe=3.0, within_2fold_pct=100.0)
        assert result["bias_acceptable"] is False

    def test_return_keys(self):
        result = acceptance_criteria_check(aafe=1.5, afe=1.0, within_2fold_pct=70.0)
        for key in ("passed", "aafe_passed", "within_2fold_passed", "bias_acceptable"):
            assert key in result

    def test_aafe_exactly_2_fails(self):
        """AAFE must be strictly < 2.0."""
        result = acceptance_criteria_check(aafe=2.0, afe=1.0, within_2fold_pct=80.0)
        assert result["aafe_passed"] is False

    def test_within_2fold_exactly_50_fails(self):
        """within_2fold_pct must be strictly > 50%."""
        result = acceptance_criteria_check(aafe=1.5, afe=1.0, within_2fold_pct=50.0)
        assert result["within_2fold_passed"] is False
