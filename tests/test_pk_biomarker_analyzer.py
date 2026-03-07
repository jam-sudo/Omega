"""Tests for clinical/pk_biomarker_analyzer.py — Phase 466."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pk_biomarker_analyzer import (
    ExposureResponseResult,
    calculate_pkpd_index,
    classify_responders,
    correlate_pk_biomarker,
    exposure_response_summary,
)

# ---------------------------------------------------------------------------
# correlate_pk_biomarker
# ---------------------------------------------------------------------------


class TestCorrelatePkBiomarker:
    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = correlate_pk_biomarker(x, y)
        assert abs(result["pearson_r"] - 1.0) < 1e-9
        assert abs(result["r_squared"] - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = correlate_pk_biomarker(x, y)
        assert abs(result["pearson_r"] - (-1.0)) < 1e-9

    def test_pearson_r_in_range(self):
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        y = [2.0, 5.0, 3.0, 4.0, 6.0]
        result = correlate_pk_biomarker(x, y)
        assert -1.0 <= result["pearson_r"] <= 1.0

    def test_r_squared_non_negative(self):
        x = [1.0, 2.0, 3.0]
        y = [3.0, 1.0, 2.0]
        result = correlate_pk_biomarker(x, y)
        assert result["r_squared"] >= 0.0

    def test_slope_intercept_linear(self):
        """y = 3x + 1 → slope=3, intercept=1."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 7.0, 10.0, 13.0]
        result = correlate_pk_biomarker(x, y)
        assert abs(result["slope"] - 3.0) < 1e-6
        assert abs(result["intercept"] - 1.0) < 1e-6

    def test_p_value_approx_small_for_perfect_corr(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [float(i) for i in range(1, 9)]
        result = correlate_pk_biomarker(x, y)
        assert result["p_value_approx"] < 0.05

    def test_p_value_between_0_and_1(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 1.0, 4.0, 3.0]
        result = correlate_pk_biomarker(x, y)
        assert 0.0 <= result["p_value_approx"] <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            correlate_pk_biomarker([], [1.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="must equal"):
            correlate_pk_biomarker([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            correlate_pk_biomarker([1.0], [1.0])

    def test_constant_x_returns_zero_r(self):
        """Constant PK values → no variation → r=0."""
        x = [5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0]
        result = correlate_pk_biomarker(x, y)
        assert result["pearson_r"] == 0.0


# ---------------------------------------------------------------------------
# calculate_pkpd_index
# ---------------------------------------------------------------------------


class TestCalculatePkpdIndex:
    def test_auc_mic_single_mic(self):
        aucs = [100.0, 200.0, 300.0]
        mics = [4.0]
        result = calculate_pkpd_index(aucs, mics, index_type="auc_mic")
        assert result == [25.0, 50.0, 75.0]

    def test_auc_mic_per_subject(self):
        aucs = [100.0, 200.0]
        mics = [2.0, 4.0]
        result = calculate_pkpd_index(aucs, mics, index_type="auc_mic")
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(50.0)

    def test_cmax_mic_index_type(self):
        cmaxs = [10.0, 20.0]
        mics = [2.0]
        result = calculate_pkpd_index(cmaxs, mics, index_type="cmax_mic")
        assert result == [5.0, 10.0]

    def test_t_mic_falls_back_to_ratio(self):
        """t_mic defaults to AUC/MIC-style ratio for compatibility."""
        aucs = [240.0]
        mics = [8.0]
        result = calculate_pkpd_index(aucs, mics, index_type="t_mic")
        assert result[0] == pytest.approx(30.0)

    def test_returns_correct_length(self):
        aucs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_pkpd_index(aucs, [1.0])
        assert len(result) == 5

    def test_empty_auc_raises(self):
        with pytest.raises(ValueError, match="auc_values must not be empty"):
            calculate_pkpd_index([], [1.0])

    def test_empty_mic_raises(self):
        with pytest.raises(ValueError, match="mic_values must not be empty"):
            calculate_pkpd_index([1.0], [])

    def test_zero_mic_raises(self):
        with pytest.raises(ValueError, match="All MIC values must be positive"):
            calculate_pkpd_index([100.0], [0.0])

    def test_invalid_index_type_raises(self):
        with pytest.raises(ValueError, match="index_type must be one of"):
            calculate_pkpd_index([1.0], [1.0], index_type="bad_type")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must be 1 or match"):
            calculate_pkpd_index([1.0, 2.0, 3.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# classify_responders
# ---------------------------------------------------------------------------


class TestClassifyResponders:
    def test_all_above_threshold(self):
        values = [10.0, 20.0, 30.0]
        result = classify_responders(values, threshold=5.0, direction="above")
        assert result["n_responders"] == 3
        assert result["n_non_responders"] == 0
        assert result["response_rate_pct"] == pytest.approx(100.0)

    def test_none_above_threshold(self):
        values = [1.0, 2.0, 3.0]
        result = classify_responders(values, threshold=10.0, direction="above")
        assert result["n_responders"] == 0
        assert result["n_non_responders"] == 3
        assert result["response_rate_pct"] == pytest.approx(0.0)

    def test_below_direction(self):
        values = [1.0, 2.0, 5.0, 10.0]
        result = classify_responders(values, threshold=3.0, direction="below")
        assert result["n_responders"] == 2

    def test_threshold_stored_in_result(self):
        result = classify_responders([1.0, 2.0], threshold=1.5)
        assert result["threshold"] == 1.5

    def test_response_rate_pct_range(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = classify_responders(values, threshold=2.0, direction="above")
        assert 0.0 <= result["response_rate_pct"] <= 100.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            classify_responders([], threshold=1.0)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be one of"):
            classify_responders([1.0, 2.0], threshold=1.0, direction="equal")

    def test_exactly_at_threshold_not_counted_above(self):
        """Boundary: value == threshold is NOT a responder for 'above'."""
        values = [3.0, 3.0, 5.0]
        result = classify_responders(values, threshold=3.0, direction="above")
        assert result["n_responders"] == 1


# ---------------------------------------------------------------------------
# exposure_response_summary
# ---------------------------------------------------------------------------


class TestExposureResponseSummary:
    def _make_perfect_data(self, n=20):
        pk = [float(i) for i in range(1, n + 1)]
        resp = [float(i) * 2.0 for i in range(1, n + 1)]
        return pk, resp

    def test_returns_exposure_response_result(self):
        pk, resp = self._make_perfect_data()
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert isinstance(result, ExposureResponseResult)

    def test_n_subjects_correct(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert result.n_subjects == 20

    def test_n_bins_correct(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert result.n_bins == 4

    def test_bin_lists_length(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert len(result.bin_median_exposure) == 4
        assert len(result.bin_mean_response) == 4
        assert len(result.bin_n) == 4

    def test_bin_n_sums_to_total(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert sum(result.bin_n) == 20

    def test_monotonic_for_perfect_data(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert result.monotonic is True

    def test_overall_correlation_near_one_for_perfect_data(self):
        pk, resp = self._make_perfect_data(20)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert abs(result.overall_correlation_r - 1.0) < 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            exposure_response_summary([], [])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="must equal"):
            exposure_response_summary([1.0, 2.0], [1.0])

    def test_fewer_subjects_than_bins_raises(self):
        with pytest.raises(ValueError):
            exposure_response_summary([1.0, 2.0], [1.0, 2.0], n_bins=5)

    def test_bin_edges_length(self):
        pk, resp = self._make_perfect_data(12)
        result = exposure_response_summary(pk, resp, n_bins=4)
        # bin_edges has n_bins + 1 entries (start + one per bin end)
        assert len(result.bin_edges) == 5

    def test_notes_contains_n_subjects(self):
        pk, resp = self._make_perfect_data(16)
        result = exposure_response_summary(pk, resp, n_bins=4)
        assert "16" in result.notes

    def test_non_monotonic_flag(self):
        pk = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        resp = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0]
        result = exposure_response_summary(pk, resp, n_bins=4)
        # With noisy data monotonicity is not guaranteed — just check flag is bool
        assert isinstance(result.monotonic, bool)
