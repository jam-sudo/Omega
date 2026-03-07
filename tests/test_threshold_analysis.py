"""Tests for analysis/threshold_analysis.py — Phase 318."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.threshold_analysis import (
    AntibioticPKPDResult,
    CoverageResult,
    ThresholdResult,
    pk_pd_breakpoint_analysis,
    therapeutic_coverage,
    time_above_threshold,
    time_below_threshold,
)

# ---------------------------------------------------------------------------
# Simple PK profiles
# ---------------------------------------------------------------------------


def _triangle_profile() -> tuple[list[float], list[float]]:
    """Concentration rises to peak at t=4 then falls."""
    times = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    concs = [0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.5, 0.05]
    return times, concs


def _flat_profile(level: float = 5.0) -> tuple[list[float], list[float]]:
    times = [0.0, 6.0, 12.0, 24.0]
    concs = [level, level, level, level]
    return times, concs


def _always_above(threshold: float = 2.0) -> tuple[list[float], list[float]]:
    times = [0.0, 6.0, 12.0]
    concs = [threshold + 1, threshold + 2, threshold + 0.5]
    return times, concs


def _always_below(threshold: float = 10.0) -> tuple[list[float], list[float]]:
    times = [0.0, 6.0, 12.0]
    concs = [0.1, 0.2, 0.05]
    return times, concs


# ---------------------------------------------------------------------------
# ThresholdResult validation
# ---------------------------------------------------------------------------


class TestTimeAboveThreshold:
    def test_basic_returns_result(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=3.0)
        assert isinstance(result, ThresholdResult)

    def test_frozen_dataclass(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=3.0)
        with pytest.raises((AttributeError, TypeError)):
            result.threshold = 99.0  # type: ignore[misc]

    def test_threshold_stored(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=3.0)
        assert result.threshold == 3.0

    def test_total_time_positive(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=1.0)
        assert result.total_time_above_h > 0.0

    def test_always_above_full_duration(self):
        t, c = _always_above(threshold=2.0)
        result = time_above_threshold(t, c, threshold=2.0)
        # All concentrations > 2, so time above = total duration
        obs = t[-1] - t[0]
        assert pytest.approx(result.total_time_above_h, rel=0.01) == obs

    def test_always_below_zero_time(self):
        t, c = _always_below(threshold=10.0)
        result = time_above_threshold(t, c, threshold=10.0)
        assert result.total_time_above_h == pytest.approx(0.0, abs=1e-10)

    def test_fraction_above_between_0_1(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=3.0)
        assert 0.0 <= result.fraction_above <= 1.0

    def test_n_intervals_correct(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=3.0)
        assert result.n_intervals == len(result.intervals_above)

    def test_intervals_above_tuple_of_tuples(self):
        t, c = _triangle_profile()
        result = time_above_threshold(t, c, threshold=1.0)
        assert isinstance(result.intervals_above, tuple)
        for iv in result.intervals_above:
            assert len(iv) == 2
            assert iv[0] < iv[1]

    def test_validation_too_few_points(self):
        with pytest.raises(ValueError, match="2 data points"):
            time_above_threshold([1.0], [1.0], 0.5)

    def test_validation_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            time_above_threshold([0.0, 1.0, 2.0], [1.0, 0.5], 0.3)

    def test_validation_non_monotonic(self):
        with pytest.raises(ValueError, match="monotonically"):
            time_above_threshold([0.0, 2.0, 1.0], [1.0, 0.8, 0.5], 0.3)

    def test_validation_negative_conc(self):
        with pytest.raises(ValueError, match=">= 0"):
            time_above_threshold([0.0, 1.0, 2.0], [1.0, -0.1, 0.5], 0.3)

    def test_validation_threshold_zero(self):
        with pytest.raises(ValueError, match="positive"):
            time_above_threshold([0.0, 1.0, 2.0], [1.0, 0.5, 0.2], 0.0)

    def test_crossing_interpolation(self):
        # Concentration: t=0:c=0, t=1:c=4, t=2:c=6, t=3:c=0 — threshold=5
        # Entry: crosses 5 between t=1 (c=4) and t=2 (c=6) → t_entry = 1 + (5-4)/(6-4)*1 = 1.5
        # Exit: crosses 5 between t=2 (c=6) and t=3 (c=0) → t_exit = 2 + (6-5)/(6-0)*1 ≈ 2.167
        times = [0.0, 1.0, 2.0, 3.0]
        concs = [0.0, 4.0, 6.0, 0.0]
        result = time_above_threshold(times, concs, threshold=5.0)
        # Should cross up between t=1 and t=2, then down between t=2 and t=3
        assert result.n_intervals == 1
        entry, exit_ = result.intervals_above[0]
        assert pytest.approx(entry, abs=0.01) == 1.5
        assert pytest.approx(exit_, abs=0.01) == 2 + 1.0 / 6.0  # ≈ 2.167


# ---------------------------------------------------------------------------
# time_below_threshold
# ---------------------------------------------------------------------------


class TestTimeBelowThreshold:
    def test_basic_returns_result(self):
        t, c = _triangle_profile()
        result = time_below_threshold(t, c, threshold=1.0)
        assert isinstance(result, ThresholdResult)

    def test_always_above_zero_below(self):
        t, c = _always_above(threshold=2.0)
        result = time_below_threshold(t, c, threshold=2.0)
        assert result.total_time_above_h == pytest.approx(0.0, abs=1e-9)

    def test_always_below_full_duration(self):
        t, c = _always_below(threshold=10.0)
        result = time_below_threshold(t, c, threshold=10.0)
        obs = t[-1] - t[0]
        assert pytest.approx(result.total_time_above_h, rel=0.01) == obs

    def test_complementary_with_above(self):
        t, c = _triangle_profile()
        thresh = 3.0
        above = time_above_threshold(t, c, threshold=thresh)
        below = time_below_threshold(t, c, threshold=thresh)
        obs = t[-1] - t[0]
        assert pytest.approx(above.total_time_above_h + below.total_time_above_h, rel=0.001) == obs

    def test_fraction_complementary(self):
        t, c = _triangle_profile()
        thresh = 2.5
        above = time_above_threshold(t, c, threshold=thresh)
        below = time_below_threshold(t, c, threshold=thresh)
        assert pytest.approx(above.fraction_above + below.fraction_above, abs=0.001) == 1.0


# ---------------------------------------------------------------------------
# therapeutic_coverage
# ---------------------------------------------------------------------------


class TestTherapeuticCoverage:
    def test_returns_coverage_result(self):
        t, c = _triangle_profile()
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=5.0)
        assert isinstance(result, CoverageResult)

    def test_frozen_dataclass(self):
        t, c = _triangle_profile()
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=5.0)
        with pytest.raises((AttributeError, TypeError)):
            result.pct_therapeutic = 99.0  # type: ignore[misc]

    def test_pct_sum_approximately_100(self):
        t, c = _triangle_profile()
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=5.0)
        total = result.pct_therapeutic + result.pct_toxic + result.pct_subtherapeutic
        assert pytest.approx(total, abs=0.5) == 100.0

    def test_flat_always_in_range(self):
        t, c = _flat_profile(level=3.0)
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=10.0)
        assert result.pct_therapeutic == pytest.approx(100.0, abs=0.1)
        assert result.pct_toxic == pytest.approx(0.0, abs=0.1)
        assert result.pct_subtherapeutic == pytest.approx(0.0, abs=0.1)

    def test_flat_always_toxic(self):
        t, c = _flat_profile(level=20.0)
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=10.0)
        assert result.pct_toxic == pytest.approx(100.0, abs=0.1)

    def test_flat_always_subtherapeutic(self):
        t, c = _flat_profile(level=0.5)
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=10.0)
        assert result.pct_subtherapeutic == pytest.approx(100.0, abs=0.1)

    def test_invalid_thresholds_raises(self):
        t, c = _triangle_profile()
        with pytest.raises(ValueError, match="upper_threshold"):
            therapeutic_coverage(t, c, lower_threshold=5.0, upper_threshold=3.0)

    def test_times_and_pcts_non_negative(self):
        t, c = _triangle_profile()
        result = therapeutic_coverage(t, c, lower_threshold=1.0, upper_threshold=5.0)
        assert result.time_therapeutic_h >= 0.0
        assert result.time_toxic_h >= 0.0
        assert result.time_subtherapeutic_h >= 0.0
        assert result.pct_therapeutic >= 0.0
        assert result.pct_toxic >= 0.0
        assert result.pct_subtherapeutic >= 0.0


# ---------------------------------------------------------------------------
# pk_pd_breakpoint_analysis
# ---------------------------------------------------------------------------


class TestPKPDBreakpoint:
    def setup_method(self):
        self.times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        self.concs = [0.0, 5.0, 8.0, 6.0, 3.0, 1.0, 0.1]
        self.mic = 1.0

    def test_returns_result(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic)
        assert isinstance(result, AntibioticPKPDResult)

    def test_frozen_dataclass(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic)
        with pytest.raises((AttributeError, TypeError)):
            result.target_attained = False  # type: ignore[misc]

    def test_auc_mic_ratio_positive(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic, "auc_mic")
        assert result.auc_mic_ratio > 0.0

    def test_cmax_mic_ratio(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic, "cmax_mic")
        assert result.cmax_mic_ratio == pytest.approx(max(self.concs) / self.mic)

    def test_ft_above_mic_between_0_100(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic, "t_above_mic")
        assert 0.0 <= result.fT_above_mic_pct <= 100.0

    def test_auc_mic_target_attained_high_auc(self):
        # Large AUC relative to MIC → target attained
        times = [0.0, 1.0, 12.0, 24.0]
        concs = [0.0, 200.0, 100.0, 50.0]
        result = pk_pd_breakpoint_analysis(times, concs, mic_mg_L=1.0, pkpd_index="auc_mic")
        assert result.target_attained is True

    def test_t_above_mic_attainment(self):
        # Profile mostly above MIC
        times = [0.0, 1.0, 6.0, 12.0]
        concs = [2.0, 5.0, 3.0, 1.5]
        result = pk_pd_breakpoint_analysis(times, concs, mic_mg_L=1.0, pkpd_index="t_above_mic")
        assert result.fT_above_mic_pct == pytest.approx(100.0, abs=0.1)
        assert result.target_attained is True

    def test_invalid_pkpd_index_raises(self):
        with pytest.raises(ValueError, match="pkpd_index"):
            pk_pd_breakpoint_analysis(self.times, self.concs, self.mic, "invalid")

    def test_mic_zero_raises(self):
        with pytest.raises(ValueError, match="mic_mg_L"):
            pk_pd_breakpoint_analysis(self.times, self.concs, 0.0)

    def test_mic_negative_raises(self):
        with pytest.raises(ValueError, match="mic_mg_L"):
            pk_pd_breakpoint_analysis(self.times, self.concs, -1.0)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="2 data points"):
            pk_pd_breakpoint_analysis([0.0], [1.0], mic_mg_L=1.0)

    def test_notes_is_tuple(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic)
        assert isinstance(result.notes, tuple)

    def test_all_three_indices_computed(self):
        result = pk_pd_breakpoint_analysis(self.times, self.concs, self.mic, "auc_mic")
        assert result.auc_total > 0.0
        assert result.auc_mic_ratio > 0.0
        assert result.fT_above_mic_pct >= 0.0
        assert result.cmax_mic_ratio > 0.0
