"""Tests for IVIVC module (Phase 956)."""

import pytest

from omega_pbpk.prediction.in_vitro_in_vivo_correlation import (
    IVIVCResult,
    compare_formulations_ivivc,
    predict_ivivc,
)

# --- Helpers ---


def _fast_dissolution():
    """Fast dissolving: mostly dissolved by 1h."""
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    fracs = [0.0, 0.5, 0.85, 0.95, 0.99, 1.0]
    return times, fracs


def _slow_dissolution():
    """Slow dissolving: gradual over 8h."""
    times = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
    fracs = [0.0, 0.1, 0.25, 0.50, 0.75, 1.0]
    return times, fracs


def _linear_dissolution():
    """Perfectly linear dissolution for R² test."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    fracs = [t / 8.0 for t in times]
    return times, fracs


def _base_result(**kwargs):
    times, fracs = _fast_dissolution()
    defaults = dict(
        drug_name="ibuprofen",
        dose_mg=200.0,
        dissolution_times_h=times,
        dissolution_fractions=fracs,
    )
    defaults.update(kwargs)
    return predict_ivivc(**defaults)


class TestReturnType:
    def test_return_type(self):
        result = _base_result()
        assert isinstance(result, IVIVCResult)


class TestBasicShape:
    def test_lengths_match(self):
        times, fracs = _fast_dissolution()
        result = predict_ivivc("drug", 100.0, times, fracs)
        assert len(result.predicted_c_plasma_mg_L) == len(times)

    def test_predicted_times_match_input(self):
        times, fracs = _fast_dissolution()
        result = predict_ivivc("drug", 100.0, times, fracs)
        assert result.predicted_times_h == times


class TestPKValues:
    def test_cmax_positive(self):
        result = _base_result()
        assert result.predicted_cmax_mg_L > 0

    def test_auc_positive(self):
        result = _base_result()
        assert result.predicted_auc_mg_h_per_L > 0

    def test_r_squared_in_range(self):
        result = _base_result()
        assert 0.0 <= result.r_squared_level_a <= 1.0

    def test_level_a_correlation_is_bool(self):
        result = _base_result()
        assert isinstance(result.level_a_correlation, bool)

    def test_fdiss_at_1h_in_range(self):
        result = _base_result()
        assert 0.0 <= result.fdiss_at_1h <= 1.0

    def test_fdiss_at_4h_in_range(self):
        result = _base_result()
        assert 0.0 <= result.fdiss_at_4h <= 1.0


class TestFastVsSlowDissolution:
    def test_fast_higher_cmax_than_slow(self):
        fast_times, fast_fracs = _fast_dissolution()
        slow_times, slow_fracs = _slow_dissolution()
        fast = predict_ivivc("drug", 100.0, fast_times, fast_fracs)
        slow = predict_ivivc("drug", 100.0, slow_times, slow_fracs)
        assert fast.predicted_cmax_mg_L > slow.predicted_cmax_mg_L

    def test_fast_higher_auc_than_slow(self):
        fast_times, fast_fracs = _fast_dissolution()
        slow_times, slow_fracs = _slow_dissolution()
        fast = predict_ivivc("drug", 100.0, fast_times, fast_fracs)
        slow = predict_ivivc("drug", 100.0, slow_times, slow_fracs)
        assert fast.predicted_auc_mg_h_per_L > slow.predicted_auc_mg_h_per_L


class TestLevelACorrelation:
    def test_r_squared_high_for_linear_dissolution(self):
        times, fracs = _linear_dissolution()
        result = predict_ivivc("drug", 100.0, times, fracs)
        assert result.r_squared_level_a >= 0.9

    def test_level_a_flag_matches_r_squared(self):
        result = _base_result()
        expected = result.r_squared_level_a >= 0.90
        assert result.level_a_correlation == expected


class TestPreservedFields:
    def test_drug_name_preserved(self):
        result = _base_result(drug_name="aspirin")
        assert result.drug_name == "aspirin"

    def test_dissolution_fractions_preserved(self):
        times, fracs = _fast_dissolution()
        result = predict_ivivc("drug", 100.0, times, fracs)
        assert result.dissolution_fractions == fracs


class TestNotes:
    def test_notes_non_empty(self):
        result = _base_result()
        assert len(result.notes) > 0


class TestCompareFormulations:
    def _make_formulations(self):
        fast_times, fast_fracs = _fast_dissolution()
        slow_times, slow_fracs = _slow_dissolution()
        return [
            {
                "name": "fast",
                "dissolution_times_h": fast_times,
                "dissolution_fractions": fast_fracs,
            },
            {
                "name": "slow",
                "dissolution_times_h": slow_times,
                "dissolution_fractions": slow_fracs,
            },
        ]

    def test_compare_returns_correct_length(self):
        forms = self._make_formulations()
        results = compare_formulations_ivivc("drug", 100.0, forms)
        assert len(results) == 2

    def test_compare_sorted_by_auc_descending(self):
        forms = self._make_formulations()
        results = compare_formulations_ivivc("drug", 100.0, forms)
        aucs = [r.predicted_auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_three_formulations(self):
        fast_t, fast_f = _fast_dissolution()
        slow_t, slow_f = _slow_dissolution()
        lin_t, lin_f = _linear_dissolution()
        forms = [
            {"name": "fast", "dissolution_times_h": fast_t, "dissolution_fractions": fast_f},
            {"name": "slow", "dissolution_times_h": slow_t, "dissolution_fractions": slow_f},
            {"name": "linear", "dissolution_times_h": lin_t, "dissolution_fractions": lin_f},
        ]
        results = compare_formulations_ivivc("drug", 100.0, forms)
        assert len(results) == 3
        aucs = [r.predicted_auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)


class TestValidation:
    def test_invalid_dose_raises(self):
        times, fracs = _fast_dissolution()
        with pytest.raises(ValueError, match="dose_mg"):
            predict_ivivc("drug", 0.0, times, fracs)

    def test_negative_dose_raises(self):
        times, fracs = _fast_dissolution()
        with pytest.raises(ValueError, match="dose_mg"):
            predict_ivivc("drug", -100.0, times, fracs)

    def test_mismatched_lists_raises(self):
        with pytest.raises(ValueError, match="match"):
            predict_ivivc("drug", 100.0, [0.0, 1.0, 2.0], [0.0, 0.5])

    def test_fewer_than_2_points_raises(self):
        with pytest.raises(ValueError, match="2"):
            predict_ivivc("drug", 100.0, [0.0], [0.0])

    def test_fraction_above_1_raises(self):
        with pytest.raises(ValueError, match="fractions"):
            predict_ivivc("drug", 100.0, [0.0, 1.0, 2.0], [0.0, 0.5, 1.2])

    def test_invalid_cl_raises(self):
        times, fracs = _fast_dissolution()
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_ivivc("drug", 100.0, times, fracs, cl_L_per_h=0.0)

    def test_invalid_vd_raises(self):
        times, fracs = _fast_dissolution()
        with pytest.raises(ValueError, match="vd_L"):
            predict_ivivc("drug", 100.0, times, fracs, vd_L=-5.0)
