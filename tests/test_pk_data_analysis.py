"""Tests for Phase 437 — PK Data Analysis (pk_data_analysis.py).

Covers:
  - summarize_pk_data: descriptive stats, geometric stats, edge cases, validation
  - detect_outliers: IQR and z-score methods, edge cases, validation
  - correlation_analysis: Pearson r, slope/intercept, validation
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_data_analysis import (
    PKDataSummary,
    correlation_analysis,
    detect_outliers,
    summarize_pk_data,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SUBJECTS_5 = [
    {"id": 1, "cl": 5.0, "vd": 50.0, "auc": 100.0, "t_half": 6.9},
    {"id": 2, "cl": 7.0, "vd": 60.0, "auc": 140.0, "t_half": 5.9},
    {"id": 3, "cl": 6.0, "vd": 55.0, "auc": 120.0, "t_half": 6.4},
    {"id": 4, "cl": 8.0, "vd": 70.0, "auc": 160.0, "t_half": 6.1},
    {"id": 5, "cl": 4.0, "vd": 45.0, "auc": 80.0, "t_half": 7.8},
]

SUBJECTS_3 = [
    {"id": 1, "cl": 2.0, "auc": 50.0},
    {"id": 2, "cl": 4.0, "auc": 100.0},
    {"id": 3, "cl": 8.0, "auc": 200.0},
]


# ===========================================================================
# summarize_pk_data
# ===========================================================================


class TestSummarizePKData:
    def test_returns_pk_data_summary(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert isinstance(result, PKDataSummary)

    def test_frozen_dataclass(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        with pytest.raises((AttributeError, TypeError)):
            result.n = 99  # type: ignore[misc]

    def test_n_correct(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.n == 5

    def test_pk_param_stored(self):
        result = summarize_pk_data(SUBJECTS_5, "auc")
        assert result.pk_param == "auc"

    def test_mean_correct(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        # cl values: 5, 7, 6, 8, 4 → mean = 30/5 = 6.0
        assert result.mean == pytest.approx(6.0)

    def test_median_odd_n(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        # sorted: 4, 5, 6, 7, 8 → median = 6
        assert result.median == pytest.approx(6.0)

    def test_median_even_n(self):
        subjects = SUBJECTS_5[:4]  # 4 subjects: cl = 5, 7, 6, 8
        result = summarize_pk_data(subjects, "cl")
        # sorted: 5, 6, 7, 8 → median = (6+7)/2 = 6.5
        assert result.median == pytest.approx(6.5)

    def test_sd_positive(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.sd > 0

    def test_cv_pct_positive(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.cv_pct > 0

    def test_min_val_correct(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.min_val == pytest.approx(4.0)

    def test_max_val_correct(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.max_val == pytest.approx(8.0)

    def test_geometric_mean_correct(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        # Geometric mean of (5, 7, 6, 8, 4)
        expected = math.exp(
            (math.log(5) + math.log(7) + math.log(6) + math.log(8) + math.log(4)) / 5
        )
        assert result.geometric_mean == pytest.approx(expected, rel=1e-6)

    def test_geometric_cv_pct_formula(self):
        """geometric_cv = (exp(sd_log) - 1) * 100."""
        result = summarize_pk_data(SUBJECTS_3, "cl")
        # cl = 2, 4, 8
        log_vals = [math.log(2), math.log(4), math.log(8)]
        log_mean = sum(log_vals) / 3
        n = 3
        sd_log = math.sqrt(sum((v - log_mean) ** 2 for v in log_vals) / (n - 1))
        expected_gcv = (math.exp(sd_log) - 1.0) * 100.0
        assert result.geometric_cv_pct == pytest.approx(expected_gcv, rel=1e-6)

    def test_geometric_mean_less_than_arithmetic_mean(self):
        """GM ≤ AM for positive values."""
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert result.geometric_mean <= result.mean + 1e-10

    def test_notes_non_empty(self):
        result = summarize_pk_data(SUBJECTS_5, "cl")
        assert len(result.notes) > 0

    def test_single_subject_sd_zero(self):
        subjects = [{"cl": 5.0}]
        result = summarize_pk_data(subjects, "cl")
        assert result.sd == pytest.approx(0.0)
        assert result.n == 1

    def test_invalid_empty_subjects(self):
        with pytest.raises(ValueError):
            summarize_pk_data([], "cl")

    def test_invalid_missing_param(self):
        with pytest.raises(ValueError):
            summarize_pk_data(SUBJECTS_5, "nonexistent_param")

    def test_invalid_empty_pk_param(self):
        with pytest.raises(ValueError):
            summarize_pk_data(SUBJECTS_5, "")

    def test_invalid_nonpositive_value(self):
        subjects = [{"cl": 0.0}, {"cl": 5.0}]
        with pytest.raises(ValueError):
            summarize_pk_data(subjects, "cl")


# ===========================================================================
# detect_outliers
# ===========================================================================


class TestDetectOutliers:
    def test_iqr_no_outliers(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = detect_outliers(values, "iqr")
        assert indices == []

    def test_iqr_detects_high_outlier(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        indices = detect_outliers(values, "iqr")
        assert 5 in indices

    def test_iqr_detects_low_outlier(self):
        values = [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        indices = detect_outliers(values, "iqr")
        assert 0 in indices

    def test_zscore_no_outliers(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = detect_outliers(values, "zscore")
        assert indices == []

    def test_zscore_detects_extreme_outlier(self):
        values = [1.0] * 20 + [1000.0]
        indices = detect_outliers(values, "zscore")
        assert 20 in indices

    def test_zscore_returns_list_of_ints(self):
        values = [1.0, 2.0, 3.0]
        indices = detect_outliers(values, "zscore")
        assert isinstance(indices, list)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            detect_outliers([1.0, 2.0, 3.0], "invalid_method")

    def test_invalid_empty_values(self):
        with pytest.raises(ValueError):
            detect_outliers([], "iqr")

    def test_single_value_iqr(self):
        indices = detect_outliers([5.0], "iqr")
        assert isinstance(indices, list)

    def test_single_value_zscore(self):
        indices = detect_outliers([5.0], "zscore")
        assert indices == []

    def test_iqr_default_method(self):
        # Default method should be iqr
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = detect_outliers(values)
        assert isinstance(indices, list)

    def test_identical_values_zscore(self):
        values = [3.0, 3.0, 3.0, 3.0]
        indices = detect_outliers(values, "zscore")
        assert indices == []

    def test_outlier_index_correct(self):
        # With many normal values, 200 is clearly an outlier
        values = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 200.0]
        indices = detect_outliers(values, "iqr")
        # The outlier is at the last index
        assert len(values) - 1 in indices


# ===========================================================================
# correlation_analysis
# ===========================================================================


class TestCorrelationAnalysis:
    def test_returns_dict(self):
        result = correlation_analysis(SUBJECTS_5, "cl", "vd")
        assert isinstance(result, dict)

    def test_dict_has_expected_keys(self):
        result = correlation_analysis(SUBJECTS_5, "cl", "vd")
        assert "r" in result
        assert "r_squared" in result
        assert "slope" in result
        assert "intercept" in result
        assert "n" in result

    def test_n_correct(self):
        result = correlation_analysis(SUBJECTS_5, "cl", "vd")
        assert result["n"] == 5

    def test_r_squared_equals_r_squared(self):
        result = correlation_analysis(SUBJECTS_5, "cl", "vd")
        assert result["r_squared"] == pytest.approx(result["r"] ** 2, rel=1e-10)

    def test_r_in_valid_range(self):
        result = correlation_analysis(SUBJECTS_5, "cl", "vd")
        assert -1.0 <= result["r"] <= 1.0

    def test_perfect_positive_correlation(self):
        """x and y identical → r = 1."""
        subjects = [{"x": float(v), "y": float(v)} for v in [1, 2, 3, 4, 5]]
        result = correlation_analysis(subjects, "x", "y")
        assert result["r"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation(self):
        subjects = [
            {"x": 1.0, "y": 5.0},
            {"x": 2.0, "y": 4.0},
            {"x": 3.0, "y": 3.0},
            {"x": 4.0, "y": 2.0},
            {"x": 5.0, "y": 1.0},
        ]
        result = correlation_analysis(subjects, "x", "y")
        assert result["r"] == pytest.approx(-1.0, abs=1e-10)

    def test_slope_positive_for_positive_correlation(self):
        subjects = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}, {"x": 3.0, "y": 6.0}]
        result = correlation_analysis(subjects, "x", "y")
        assert result["slope"] > 0

    def test_intercept_zero_for_y_equals_2x(self):
        subjects = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}, {"x": 3.0, "y": 6.0}]
        result = correlation_analysis(subjects, "x", "y")
        assert result["intercept"] == pytest.approx(0.0, abs=1e-10)

    def test_slope_correct_value(self):
        subjects = [{"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 5.0}, {"x": 3.0, "y": 7.0}]
        result = correlation_analysis(subjects, "x", "y")
        # y = 2x + 1 → slope=2, intercept=1
        assert result["slope"] == pytest.approx(2.0, rel=1e-6)
        assert result["intercept"] == pytest.approx(1.0, rel=1e-6)

    def test_invalid_empty_subjects(self):
        with pytest.raises(ValueError):
            correlation_analysis([], "cl", "vd")

    def test_invalid_single_subject(self):
        with pytest.raises(ValueError):
            correlation_analysis([{"cl": 5.0, "vd": 50.0}], "cl", "vd")

    def test_invalid_empty_param_x(self):
        with pytest.raises(ValueError):
            correlation_analysis(SUBJECTS_5, "", "vd")

    def test_invalid_missing_param_y(self):
        with pytest.raises(ValueError):
            correlation_analysis(SUBJECTS_5, "cl", "nonexistent")

    def test_correlation_cl_auc_negative(self):
        """AUC = dose/CL: higher CL → lower AUC (negative correlation)."""
        # Construct proper inverse relationship
        subjects = [
            {"cl": 2.0, "auc": 250.0},
            {"cl": 4.0, "auc": 125.0},
            {"cl": 5.0, "auc": 100.0},
            {"cl": 10.0, "auc": 50.0},
            {"cl": 20.0, "auc": 25.0},
        ]
        result = correlation_analysis(subjects, "cl", "auc")
        assert result["r"] < 0
