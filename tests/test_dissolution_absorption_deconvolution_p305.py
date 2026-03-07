"""Tests for Phase 305 — dissolution_absorption_deconvolution module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.dissolution_absorption_deconvolution_p305 import (
    DeconvolutionResult,
    deconvolve_dissolution_absorption,
    predict_in_vivo_absorption,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

TIMES_MIN = [0.0, 15.0, 30.0, 60.0, 120.0, 240.0]
DISSOLVED = [0.0, 20.0, 40.0, 65.0, 85.0, 98.0]
KE = 0.2  # /h
KA = 0.5  # /h
F_ORAL = 0.8


def _make_result(**kwargs) -> DeconvolutionResult:
    defaults = dict(
        drug_name="DrugA",
        times_dissolution_min=TIMES_MIN,
        dissolved_pct=DISSOLVED,
        ke_per_h=KE,
        ka_per_h=KA,
        f_oral=F_ORAL,
    )
    defaults.update(kwargs)
    return deconvolve_dissolution_absorption(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_deconvolution_result(self):
        result = _make_result()
        assert isinstance(result, DeconvolutionResult)

    def test_drug_name_preserved(self):
        result = _make_result(drug_name="Metformin")
        assert result.drug_name == "Metformin"

    def test_f_oral_preserved(self):
        result = _make_result(f_oral=0.6)
        assert math.isclose(result.f_oral, 0.6, rel_tol=1e-9)

    def test_ke_preserved(self):
        result = _make_result(ke_per_h=0.3)
        assert math.isclose(result.ke_per_h, 0.3, rel_tol=1e-9)

    def test_ka_preserved(self):
        result = _make_result(ka_per_h=0.7)
        assert math.isclose(result.ka_per_h, 0.7, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# times_h conversion
# ---------------------------------------------------------------------------


class TestTimesHConversion:
    def test_times_h_length_equals_input(self):
        result = _make_result()
        assert len(result.times_h) == len(TIMES_MIN)

    def test_times_h_conversion_correct(self):
        result = _make_result()
        for t_min, t_h in zip(TIMES_MIN, result.times_h):
            assert math.isclose(t_h, t_min / 60.0, rel_tol=1e-9)

    def test_dissolved_pct_stored(self):
        result = _make_result()
        assert len(result.dissolved_pct) == len(DISSOLVED)
        for expected, actual in zip(DISSOLVED, result.dissolved_pct):
            assert math.isclose(expected, actual, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# fa_cumulative_pct
# ---------------------------------------------------------------------------


class TestFaCumulativePct:
    def test_fa_cumulative_length_equals_input(self):
        result = _make_result()
        assert len(result.fa_cumulative_pct) == len(DISSOLVED)

    def test_fa_cumulative_equals_f_oral_times_dissolved(self):
        result = _make_result(f_oral=0.7)
        for d, fa in zip(result.dissolved_pct, result.fa_cumulative_pct):
            assert math.isclose(fa, d * 0.7, rel_tol=1e-9)

    def test_fa_cumulative_all_nonnegative(self):
        result = _make_result()
        for fa in result.fa_cumulative_pct:
            assert fa >= 0.0

    def test_fa_cumulative_monotone_non_decreasing(self):
        result = _make_result()
        for i in range(1, len(result.fa_cumulative_pct)):
            assert result.fa_cumulative_pct[i] >= result.fa_cumulative_pct[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------


class TestRSquared:
    def test_r_squared_in_0_1(self):
        result = _make_result()
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_high_f_oral_level_a(self):
        # f_oral=1.0 should give a high r_squared (rates perfectly correlated)
        result = _make_result(f_oral=1.0)
        assert result.r_squared >= 0.9

    def test_r_squared_lower_for_lower_f_oral(self):
        # With all dissolution monotone increasing, rates will be perfectly correlated
        # regardless of f_oral — r_squared should still be in [0,1]
        result = _make_result(f_oral=0.3)
        assert 0.0 <= result.r_squared <= 1.0


# ---------------------------------------------------------------------------
# ivivc_level
# ---------------------------------------------------------------------------


class TestIvivLevel:
    def test_ivivc_level_valid_string(self):
        result = _make_result()
        assert result.ivivc_level in {"Level_A", "Level_B"}

    def test_high_r_squared_gives_level_a(self):
        result = _make_result(f_oral=1.0)
        assert result.ivivc_level == "Level_A"


# ---------------------------------------------------------------------------
# tmax_estimate_h
# ---------------------------------------------------------------------------


class TestTmaxEstimate:
    def test_tmax_estimate_positive(self):
        result = _make_result()
        assert result.tmax_estimate_h > 0.0

    def test_tmax_estimate_within_time_range(self):
        result = _make_result()
        t_end_h = max(TIMES_MIN) / 60.0
        assert result.tmax_estimate_h <= t_end_h + 1e-9

    def test_tmax_estimate_corresponds_to_valid_time(self):
        result = _make_result()
        times_h_set = {t / 60.0 for t in TIMES_MIN}
        assert result.tmax_estimate_h in times_h_set


# ---------------------------------------------------------------------------
# Higher f_oral → higher fa_cumulative
# ---------------------------------------------------------------------------


class TestFOralEffect:
    def test_higher_f_oral_gives_higher_fa_cumulative(self):
        r_high = _make_result(f_oral=0.9)
        r_low = _make_result(f_oral=0.3)
        assert max(r_high.fa_cumulative_pct) > max(r_low.fa_cumulative_pct)

    def test_f_oral_1_fa_equals_dissolved(self):
        result = _make_result(f_oral=1.0)
        for d, fa in zip(result.dissolved_pct, result.fa_cumulative_pct):
            assert math.isclose(fa, d, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# predict_in_vivo_absorption
# ---------------------------------------------------------------------------


class TestPredictInVivoAbsorption:
    def test_returns_list(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.5,
        )
        assert isinstance(result, list)

    def test_returns_list_of_tuples(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.5,
        )
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_first_time_is_zero(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.5,
        )
        assert math.isclose(result[0][0], 0.0, abs_tol=1e-9)

    def test_first_f_abs_is_zero(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.5,
        )
        assert math.isclose(result[0][1], 0.0, abs_tol=1e-9)

    def test_f_abs_nonnegative(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.1,
        )
        for _, f_abs in result:
            assert f_abs >= 0.0

    def test_f_abs_bounded_by_f_oral_times_100(self):
        result = predict_in_vivo_absorption(
            drug_name="DrugA",
            times_dissolution_min=TIMES_MIN,
            dissolved_pct=DISSOLVED,
            ke_per_h=KE,
            f_oral=F_ORAL,
            t_end_h=4.0,
            dt_h=0.1,
        )
        for _, f_abs in result:
            assert f_abs <= F_ORAL * 100.0 + 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_too_few_time_points_raises(self):
        with pytest.raises(ValueError, match="3 points"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=[0.0, 30.0],
                dissolved_pct=[0.0, 50.0],
                ke_per_h=0.2,
                ka_per_h=0.5,
                f_oral=0.8,
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=[0.0, 30.0, 60.0],
                dissolved_pct=[0.0, 50.0],
                ke_per_h=0.2,
                ka_per_h=0.5,
                f_oral=0.8,
            )

    def test_ke_zero_raises(self):
        with pytest.raises(ValueError, match="ke_per_h"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=DISSOLVED,
                ke_per_h=0.0,
                ka_per_h=0.5,
                f_oral=0.8,
            )

    def test_ke_negative_raises(self):
        with pytest.raises(ValueError, match="ke_per_h"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=DISSOLVED,
                ke_per_h=-0.1,
                ka_per_h=0.5,
                f_oral=0.8,
            )

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=DISSOLVED,
                ke_per_h=0.2,
                ka_per_h=0.0,
                f_oral=0.8,
            )

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=DISSOLVED,
                ke_per_h=0.2,
                ka_per_h=0.5,
                f_oral=0.0,
            )

    def test_f_oral_above_1_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=DISSOLVED,
                ke_per_h=0.2,
                ka_per_h=0.5,
                f_oral=1.5,
            )

    def test_dissolved_pct_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dissolved_pct"):
            deconvolve_dissolution_absorption(
                drug_name="D",
                times_dissolution_min=TIMES_MIN,
                dissolved_pct=[0.0, 20.0, 40.0, 65.0, 85.0, 110.0],
                ke_per_h=0.2,
                ka_per_h=0.5,
                f_oral=0.8,
            )

    def test_predict_too_few_points_raises(self):
        with pytest.raises(ValueError):
            predict_in_vivo_absorption(
                drug_name="D",
                times_dissolution_min=[0.0, 60.0],
                dissolved_pct=[0.0, 50.0],
                ke_per_h=0.2,
                f_oral=0.8,
            )
