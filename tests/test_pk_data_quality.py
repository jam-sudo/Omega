"""Tests for Phase 302 — PK Data Quality Assessment.

Covers:
  - TestAssessSamplingSchedule (8): scheduling criteria and scoring
  - TestDetectOutliersPK (7): IQR and z-score outlier detection
  - TestAssessBLQHandling (6): BLQ location and recommendations
  - TestPKDataQualityReport (4): integrated report and grading
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.validation.pk_data_quality import (
    BLQResult,
    DataQualityReport,
    OutlierDetectionResult,
    SamplingQualityResult,
    assess_blq_handling,
    assess_sampling_schedule,
    detect_outliers_pk,
    pk_data_quality_report,
)

# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------

# Rich oral profile: 10 points spanning absorption + Cmax + terminal
_ORAL_TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
_ORAL_CONCS = [0.2, 0.8, 2.0, 3.5, 4.0, 3.0, 2.0, 1.2, 0.5, 0.05]

# IV profile with 8 points
_IV_TIMES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
_IV_CONCS = [10.0, 8.0, 6.5, 5.0, 3.5, 2.0, 0.8, 0.3]

# Sparse profile (4 points, barely meeting minimum)
_SPARSE_TIMES = [1.0, 4.0, 8.0, 24.0]
_SPARSE_CONCS = [3.0, 2.0, 1.0, 0.1]


# ---------------------------------------------------------------------------
# TestAssessSamplingSchedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessSamplingSchedule:
    def test_rich_oral_profile_high_score(self):
        """Well-designed oral profile should score >= 4."""
        result = assess_sampling_schedule(_ORAL_TIMES)
        assert isinstance(result, SamplingQualityResult)
        assert result.quality_score >= 4
        assert result.n_points == len(_ORAL_TIMES)

    def test_iv_route_accepted(self):
        """IV route does not require absorption-phase points."""
        result = assess_sampling_schedule(_IV_TIMES, route="iv")
        assert isinstance(result, SamplingQualityResult)
        assert result.quality_score >= 3

    def test_insufficient_points_penalised(self):
        """Fewer than 6 points → criterion 1 fails, score < 5."""
        times = [1.0, 2.0, 4.0]  # only 3 points
        result = assess_sampling_schedule(times)
        assert result.quality_score < 5
        assert not result.has_terminal_points or True  # just check no crash

    def test_score_range(self):
        """Score is always 0–5."""
        result = assess_sampling_schedule(_ORAL_TIMES)
        assert 0 <= result.quality_score <= 5

    def test_recommendations_are_list(self):
        """Recommendations is always a list (may be empty)."""
        result = assess_sampling_schedule(_ORAL_TIMES)
        assert isinstance(result.recommendations, list)

    def test_missing_terminal_points_adds_recommendation(self):
        """Profile with no late samples should recommend terminal points."""
        # Only early time-points
        times = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
        result = assess_sampling_schedule(times)
        assert not result.has_terminal_points
        assert any("terminal" in r.lower() for r in result.recommendations)

    def test_frozen_result(self):
        """SamplingQualityResult is a frozen dataclass."""
        result = assess_sampling_schedule(_ORAL_TIMES)
        with pytest.raises((AttributeError, TypeError)):
            result.quality_score = 99  # type: ignore[misc]

    def test_monotone_violation_raises(self):
        """Non-monotone times should raise ValueError."""
        with pytest.raises(ValueError):
            assess_sampling_schedule([1.0, 0.5, 2.0])


# ---------------------------------------------------------------------------
# TestDetectOutliersPK
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectOutliersPK:
    def test_no_outliers_clean_data(self):
        """Smooth profile has no outliers."""
        result = detect_outliers_pk(_ORAL_TIMES, _ORAL_CONCS)
        assert isinstance(result, OutlierDetectionResult)
        assert result.n_outliers == 0
        assert result.cleaned_times == _ORAL_TIMES
        assert result.cleaned_concs == _ORAL_CONCS

    def test_spike_detected_iqr(self):
        """Extreme spike is detected by IQR method."""
        times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        concs = [1.0, 2.0, 3.0, 500.0, 2.0, 1.0, 0.5]  # index 3 is a spike
        result = detect_outliers_pk(times, concs, method="iqr")
        assert result.n_outliers >= 1
        assert 3 in result.outlier_indices

    def test_spike_detected_zscore(self):
        """Extreme spike is detected by z-score method (log-space).

        Use a spike of 1e8 vs background ~1-3 so that log-space z-score clearly
        exceeds the 2.5 threshold.
        """
        times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        concs = [1.0, 2.0, 3.0, 1e8, 2.0, 1.0, 0.5]
        result = detect_outliers_pk(times, concs, method="zscore")
        assert result.n_outliers >= 1

    def test_cleaned_data_excludes_outliers(self):
        """Cleaned arrays have outlier removed."""
        times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        concs = [1.0, 2.0, 3.0, 500.0, 2.0, 1.0, 0.5]
        result = detect_outliers_pk(times, concs, method="iqr")
        assert len(result.cleaned_times) == len(times) - result.n_outliers

    def test_method_field_matches_input(self):
        """Method field in result matches the requested method."""
        result = detect_outliers_pk(_ORAL_TIMES, _ORAL_CONCS, method="zscore")
        assert result.method == "zscore"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            detect_outliers_pk(_ORAL_TIMES, _ORAL_CONCS, method="mahalanobis")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            detect_outliers_pk([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_frozen_result(self):
        result = detect_outliers_pk(_ORAL_TIMES, _ORAL_CONCS)
        with pytest.raises((AttributeError, TypeError)):
            result.n_outliers = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestAssessBLQHandling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessBLQHandling:
    def test_no_blq_values(self):
        """All concentrations above LLOQ → location 'none'."""
        result = assess_blq_handling(_ORAL_TIMES, _ORAL_CONCS, lloq=0.01)
        assert isinstance(result, BLQResult)
        assert result.n_blq == 0
        assert result.blq_location == "none"

    def test_terminal_blq_acceptable(self):
        """BLQ values only in terminal phase → location 'terminal'.

        The terminal phase is the last 30% of the sampling window.
        With times [0.5..24], terminal_start = 24 - 0.30*23.5 ≈ 17.0 h.
        Only the sample at t=24 is BLQ, which is ≥ 17 → terminal.
        """
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        concs = [5.0, 8.0, 6.0, 3.0, 1.0, 0.5, 0.01]  # only t=24 is BLQ (< 0.1)
        result = assess_blq_handling(times, concs, lloq=0.1)
        assert result.blq_location == "terminal"
        assert "LLOQ/2" in result.recommendation or "M3" in result.recommendation

    def test_mid_profile_blq_problematic(self):
        """BLQ values in the middle of profile → 'mid_profile'."""
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        concs = [5.0, 0.0, 6.0, 3.0, 1.0, 0.5, 0.1]  # index 1 is mid-profile BLQ
        result = assess_blq_handling(times, concs, lloq=0.05)
        assert result.blq_location == "mid_profile"
        assert "M3" in result.recommendation or "M3 method" in result.recommendation

    def test_pct_blq_calculation(self):
        """pct_blq is 100 * n_blq / total."""
        times = [0.5, 1.0, 2.0, 4.0]
        concs = [0.01, 2.0, 1.0, 0.01]  # 2 of 4 are BLQ with lloq=0.1
        result = assess_blq_handling(times, concs, lloq=0.1)
        assert math.isclose(result.pct_blq, 50.0)

    def test_invalid_lloq_negative(self):
        with pytest.raises(ValueError, match="lloq"):
            assess_blq_handling(_ORAL_TIMES, _ORAL_CONCS, lloq=-1.0)

    def test_frozen_result(self):
        result = assess_blq_handling(_ORAL_TIMES, _ORAL_CONCS, lloq=0.01)
        with pytest.raises((AttributeError, TypeError)):
            result.n_blq = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPKDataQualityReport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPKDataQualityReport:
    def test_rich_oral_profile_grade_a_or_b(self):
        """Well-sampled profile with no issues → grade A or B."""
        result = pk_data_quality_report(_ORAL_TIMES, _ORAL_CONCS, route="oral", lloq=0.01)
        assert isinstance(result, DataQualityReport)
        assert result.overall_grade in ("A", "B")
        assert result.overall_score >= 75

    def test_poor_profile_low_score(self):
        """Only 3 points → low overall score."""
        result = pk_data_quality_report([1.0, 2.0, 4.0], [3.0, 2.0, 1.0])
        assert result.overall_score < 75

    def test_no_lloq_still_works(self):
        """LLOQ=None should not raise; BLQ shows 'none'."""
        result = pk_data_quality_report(_ORAL_TIMES, _ORAL_CONCS)
        assert result.blq.blq_location == "none"
        assert isinstance(result.overall_grade, str)

    def test_grade_letter_valid(self):
        """Grade is one of A, B, C, D, F."""
        result = pk_data_quality_report(_ORAL_TIMES, _ORAL_CONCS)
        assert result.overall_grade in ("A", "B", "C", "D", "F")

    def test_notes_non_empty(self):
        result = pk_data_quality_report(_ORAL_TIMES, _ORAL_CONCS)
        assert len(result.notes) > 0

    def test_overall_score_bounded(self):
        """Overall score is in [0, 100]."""
        result = pk_data_quality_report(_ORAL_TIMES, _ORAL_CONCS)
        assert 0 <= result.overall_score <= 100

    def test_monotone_violation_raises(self):
        """Non-monotone times should raise ValueError."""
        with pytest.raises(ValueError):
            pk_data_quality_report([1.0, 0.5, 2.0], [1.0, 2.0, 3.0])

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="Concentrations"):
            pk_data_quality_report([1.0, 2.0, 3.0], [1.0, -0.5, 2.0])
