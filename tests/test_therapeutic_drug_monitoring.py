"""Tests for TDM-guided dose adjustment (Phase 403)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.therapeutic_drug_monitoring import (
    TDMAdjustmentResult,
    bayesian_dose_adjustment,
    trough_based_adjustment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_decay(cl=5.0, vd=50.0, dose=100.0, times=None):
    """Generate synthetic 1-cpt IV decay observations."""
    if times is None:
        times = [1.0, 4.0, 8.0, 12.0]
    ke = cl / vd
    concs = [(dose / vd) * math.exp(-ke * t) for t in times]
    return concs, times


# ---------------------------------------------------------------------------
# trough_based_adjustment
# ---------------------------------------------------------------------------


class TestTroughBasedAdjustment:
    def test_returns_dict_with_expected_keys(self):
        result = trough_based_adjustment(1.0, 2.0, 100.0)
        assert "new_dose_mg" in result
        assert "dose_change_pct" in result

    def test_double_trough_doubles_dose(self):
        result = trough_based_adjustment(1.0, 2.0, 100.0)
        assert abs(result["new_dose_mg"] - 200.0) < 0.01

    def test_half_trough_halves_dose(self):
        result = trough_based_adjustment(2.0, 1.0, 200.0)
        assert abs(result["new_dose_mg"] - 100.0) < 0.01

    def test_equal_trough_no_change(self):
        result = trough_based_adjustment(3.0, 3.0, 150.0)
        assert abs(result["dose_change_pct"]) < 0.01
        assert abs(result["new_dose_mg"] - 150.0) < 0.01

    def test_dose_change_pct_positive_when_increase(self):
        result = trough_based_adjustment(1.0, 2.0, 100.0)
        assert result["dose_change_pct"] > 0

    def test_dose_change_pct_negative_when_decrease(self):
        result = trough_based_adjustment(3.0, 1.5, 300.0)
        assert result["dose_change_pct"] < 0

    def test_zero_current_trough_raises(self):
        with pytest.raises(ValueError, match="current_trough"):
            trough_based_adjustment(0.0, 1.0, 100.0)

    def test_negative_current_trough_raises(self):
        with pytest.raises(ValueError):
            trough_based_adjustment(-1.0, 1.0, 100.0)

    def test_zero_target_trough_raises(self):
        with pytest.raises(ValueError, match="target_trough"):
            trough_based_adjustment(1.0, 0.0, 100.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="current_dose_mg"):
            trough_based_adjustment(1.0, 2.0, 0.0)

    def test_dose_change_pct_formula_correct(self):
        result = trough_based_adjustment(2.0, 3.0, 100.0)
        expected_new_dose = 100.0 * 3.0 / 2.0
        expected_pct = 100.0 * (expected_new_dose - 100.0) / 100.0
        assert abs(result["dose_change_pct"] - expected_pct) < 0.01


# ---------------------------------------------------------------------------
# bayesian_dose_adjustment — input validation
# ---------------------------------------------------------------------------


class TestBayesianDoseAdjustmentValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            bayesian_dose_adjustment(
                "", [1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0], 5.0, 50.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_dose_adjustment(
                "Drug", [-1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [-1.0, 8.0], 5.0, 50.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_zero_prior_cl_raises(self):
        with pytest.raises(ValueError, match="prior_cl_L_per_h"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 0.0, 50.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_zero_prior_vd_raises(self):
        with pytest.raises(ValueError, match="prior_vd_L"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 5.0, 0.0, 100.0, 12.0, target_auc24=50.0
            )

    def test_zero_current_dose_raises(self):
        with pytest.raises(ValueError, match="current_dose_mg"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 0.0, 12.0, target_auc24=50.0
            )

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 100.0, 0.0, target_auc24=50.0
            )

    def test_no_targets_raises(self):
        with pytest.raises(ValueError, match="target"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 100.0, 12.0
            )

    def test_zero_target_auc24_raises(self):
        with pytest.raises(ValueError, match="target_auc24"):
            bayesian_dose_adjustment(
                "Drug", [1.0, 0.5], [4.0, 8.0], 5.0, 50.0, 100.0, 12.0, target_auc24=0.0
            )

    def test_zero_target_trough_raises(self):
        with pytest.raises(ValueError, match="target_trough"):
            bayesian_dose_adjustment(
                "Drug",
                [1.0, 0.5],
                [4.0, 8.0],
                5.0,
                50.0,
                100.0,
                12.0,
                target_trough=0.0,
            )


# ---------------------------------------------------------------------------
# bayesian_dose_adjustment — functional behaviour
# ---------------------------------------------------------------------------


class TestBayesianDoseAdjustmentFunctional:
    def test_returns_tdm_result_type(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "VancomycinTest", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert isinstance(result, TDMAdjustmentResult)

    def test_drug_name_preserved(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Warfarin", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert result.drug_name == "Warfarin"

    def test_current_dose_preserved(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert abs(result.current_dose_mg - 100.0) < 0.01

    def test_interval_preserved(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert abs(result.interval_h - 12.0) < 0.01

    def test_individual_cl_positive(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert result.individual_cl_L_per_h > 0

    def test_individual_vd_equals_prior_when_sparse(self):
        # Only 1 observation → prior Vd used
        result = bayesian_dose_adjustment(
            "Drug", [1.0], [4.0], 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert abs(result.individual_vd_L - 50.0) < 0.01

    def test_recommended_dose_positive(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert result.recommended_dose_mg > 0

    def test_predicted_auc24_positive(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert result.predicted_auc24 > 0

    def test_predicted_trough_nonneg(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert result.predicted_trough >= 0

    def test_dose_change_pct_computed(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        expected = 100.0 * (result.recommended_dose_mg - 100.0) / 100.0
        assert abs(result.dose_change_pct - expected) < 0.1

    def test_notes_is_list(self):
        concs, times = _simple_decay()
        result = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert isinstance(result.notes, list)

    def test_fewer_than_2_obs_uses_prior_cl(self):
        result = bayesian_dose_adjustment(
            "Drug", [1.0], [4.0], 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        # With 1 observation, prior CL (5.0) is used
        assert abs(result.individual_cl_L_per_h - 5.0) < 0.01

    def test_auc_target_overrides_trough_target(self):
        """When both targets given, AUC target takes precedence."""
        concs, times = _simple_decay()
        result_auc = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0,
            target_auc24=400.0, target_trough=0.5
        )
        result_auc_only = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert abs(result_auc.recommended_dose_mg - result_auc_only.recommended_dose_mg) < 0.01

    def test_trough_target_only_adjusts_dose(self):
        result = bayesian_dose_adjustment(
            "Drug", [1.0], [4.0], 5.0, 50.0, 100.0, 12.0, target_trough=2.0
        )
        assert result.recommended_dose_mg > 0

    def test_empty_observations_uses_prior_cl(self):
        result = bayesian_dose_adjustment(
            "Drug", [], [], 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert abs(result.individual_cl_L_per_h - 5.0) < 0.01

    def test_higher_target_auc_gives_higher_dose(self):
        concs, times = _simple_decay()
        r1 = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=200.0
        )
        r2 = bayesian_dose_adjustment(
            "Drug", concs, times, 5.0, 50.0, 100.0, 12.0, target_auc24=400.0
        )
        assert r2.recommended_dose_mg > r1.recommended_dose_mg
