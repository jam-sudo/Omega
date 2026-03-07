"""Tests for Phase 642 drug exposure metrics."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.drug_exposure_metrics import (
    ExposureMetrics,
    compare_exposures,
    compute_exposure_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Simple triangular PK profile: rises then falls
_TIMES = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
_CONCS = [0.0, 2.0, 4.0, 3.0, 2.0, 1.0, 0.2]

# Constant concentration profile
_TIMES_FLAT = [0.0, 6.0, 12.0, 24.0]
_CONCS_FLAT = [2.0, 2.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# Basic return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_exposure_metrics(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert isinstance(result, ExposureMetrics)

    def test_auc_total_positive(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.auc_total > 0.0

    def test_cmax_equals_max_concs(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.cmax == max(_CONCS)

    def test_tmax_h_correctly_identified(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.tmax_h == 2.0

    def test_cmin_excludes_t0(self):
        """cmin should exclude the t=0 value."""
        result = compute_exposure_metrics(_TIMES, _CONCS)
        # t=0 has conc=0; cmin should be the min of concs[1:] = 0.2
        assert result.cmin == min(_CONCS[1:])

    def test_peak_trough_ratio_greater_than_one(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.peak_trough_ratio > 1.0

    def test_peak_trough_ratio_999_when_cmin_zero(self):
        times = [0.0, 1.0, 2.0, 3.0]
        concs = [1.0, 2.0, 1.0, 0.0]
        result = compute_exposure_metrics(times, concs)
        assert result.peak_trough_ratio == 999.0

    def test_notes_nonempty(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.notes != ""


# ---------------------------------------------------------------------------
# Time above threshold
# ---------------------------------------------------------------------------


class TestTimeAboveThreshold:
    def test_positive_when_all_concs_above_threshold(self):
        result = compute_exposure_metrics(_TIMES_FLAT, _CONCS_FLAT, threshold_mg_L=1.0)
        assert result.time_above_threshold_h > 0.0

    def test_zero_when_threshold_above_cmax(self):
        result = compute_exposure_metrics(_TIMES, _CONCS, threshold_mg_L=10.0)
        assert result.time_above_threshold_h == 0.0

    def test_equals_duration_when_all_above(self):
        result = compute_exposure_metrics(_TIMES_FLAT, _CONCS_FLAT, threshold_mg_L=0.0)
        assert abs(result.time_above_threshold_h - 24.0) < 1e-9


# ---------------------------------------------------------------------------
# Time-weighted average
# ---------------------------------------------------------------------------


class TestTimeWeightedAvg:
    def test_twa_approx_auc_over_duration(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        duration = _TIMES[-1] - _TIMES[0]
        expected_twa = result.auc_total / duration
        assert abs(result.time_weighted_avg_mg_L - expected_twa) < 1e-9

    def test_twa_equals_constant_conc_for_flat_profile(self):
        result = compute_exposure_metrics(_TIMES_FLAT, _CONCS_FLAT)
        assert abs(result.time_weighted_avg_mg_L - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# AUC partials
# ---------------------------------------------------------------------------


class TestAUCPartials:
    def test_auc_0_to_tau_less_than_total_when_tau_partial(self):
        result = compute_exposure_metrics(_TIMES, _CONCS, tau_h=8.0)
        assert result.auc_0_to_tau < result.auc_total

    def test_auc_0_to_tau_approx_total_when_tau_exceeds_duration(self):
        result = compute_exposure_metrics(_TIMES, _CONCS, tau_h=100.0)
        assert abs(result.auc_0_to_tau - result.auc_total) < 1e-9

    def test_auc_halves_sum_to_total_within_5pct(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        summed = result.auc_first_half + result.auc_second_half
        assert abs(summed - result.auc_total) / result.auc_total < 0.05

    def test_auc_first_half_positive(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.auc_first_half > 0.0

    def test_auc_second_half_positive(self):
        result = compute_exposure_metrics(_TIMES, _CONCS)
        assert result.auc_second_half > 0.0


# ---------------------------------------------------------------------------
# compare_exposures
# ---------------------------------------------------------------------------


class TestCompareExposures:
    def _make_scenarios(self):
        return [
            {
                "name": "high_dose",
                "times_h": [0.0, 1.0, 2.0, 4.0, 8.0],
                "concs_mg_L": [0.0, 4.0, 8.0, 6.0, 2.0],
            },
            {
                "name": "low_dose",
                "times_h": [0.0, 1.0, 2.0, 4.0, 8.0],
                "concs_mg_L": [0.0, 1.0, 2.0, 1.5, 0.5],
            },
            {
                "name": "mid_dose",
                "times_h": [0.0, 1.0, 2.0, 4.0, 8.0],
                "concs_mg_L": [0.0, 2.0, 4.0, 3.0, 1.0],
            },
        ]

    def test_returns_sorted_descending_by_auc(self):
        scenarios = self._make_scenarios()
        results = compare_exposures(scenarios)
        aucs = [r["auc_total"] for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_each_entry_has_name_key(self):
        scenarios = self._make_scenarios()
        results = compare_exposures(scenarios)
        for r in results:
            assert "name" in r

    def test_high_dose_higher_auc_than_low_dose(self):
        scenarios = self._make_scenarios()
        results = compare_exposures(scenarios)
        name_to_auc = {r["name"]: r["auc_total"] for r in results}
        assert name_to_auc["high_dose"] > name_to_auc["low_dose"]

    def test_returns_list(self):
        results = compare_exposures(self._make_scenarios())
        assert isinstance(results, list)

    def test_result_count_matches_input(self):
        scenarios = self._make_scenarios()
        results = compare_exposures(scenarios)
        assert len(results) == len(scenarios)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0], [1.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 2.0])

    def test_non_monotone_times_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, -0.1, 0.5])

    def test_tau_h_zero_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], tau_h=0.0)

    def test_tau_h_negative_raises(self):
        with pytest.raises(ValueError):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], tau_h=-1.0)
