"""Tests for omega_pbpk.clinical.half_life_calculator."""

import math
import pytest
import numpy as np

from omega_pbpk.clinical.half_life_calculator import (
    HalfLifeResult,
    calculate_half_life,
    optimal_dosing_interval,
    half_life_from_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_result():
    return calculate_half_life("DrugA", t_half_h=8.0)


@pytest.fixture
def result_with_interval():
    return calculate_half_life("DrugB", t_half_h=12.0, dosing_interval_h=24.0)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_calculate_half_life_returns_dataclass(self, basic_result):
        assert isinstance(basic_result, HalfLifeResult)

    def test_optimal_dosing_interval_returns_dataclass(self):
        result = optimal_dosing_interval("DrugX", t_half_h=6.0)
        assert isinstance(result, HalfLifeResult)

    def test_dataclass_has_all_fields(self, basic_result):
        expected_fields = {
            "drug_name",
            "t_half_h",
            "t_half_alpha_h",
            "t_half_beta_h",
            "n_half_lives_to_steady_state",
            "n_half_lives_to_washout",
            "tss_h",
            "twash_h",
            "accumulation_ratio",
            "dosing_interval_h",
            "recommended_doses_per_day",
        }
        for field in expected_fields:
            assert hasattr(basic_result, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 2. n_half_lives_to_steady_state ≈ 4.32
# ---------------------------------------------------------------------------

class TestNHalfLivesToSteadyState:
    def test_n_ss_approx_4_32_short_half_life(self):
        result = calculate_half_life("DrugA", t_half_h=2.0)
        assert abs(result.n_half_lives_to_steady_state - 4.32) < 0.1

    def test_n_ss_approx_4_32_medium_half_life(self):
        result = calculate_half_life("DrugB", t_half_h=12.0)
        assert abs(result.n_half_lives_to_steady_state - 4.32) < 0.1

    def test_n_ss_approx_4_32_long_half_life(self):
        result = calculate_half_life("DrugC", t_half_h=48.0)
        assert abs(result.n_half_lives_to_steady_state - 4.32) < 0.1

    def test_n_ss_independent_of_dosing_interval(self):
        r1 = calculate_half_life("DrugD", t_half_h=10.0, dosing_interval_h=8.0)
        r2 = calculate_half_life("DrugD", t_half_h=10.0, dosing_interval_h=24.0)
        assert abs(r1.n_half_lives_to_steady_state - r2.n_half_lives_to_steady_state) < 0.01


# ---------------------------------------------------------------------------
# 3. tss_h = n_ss * t_half_h
# ---------------------------------------------------------------------------

class TestTssH:
    def test_tss_h_equals_n_ss_times_t_half(self):
        result = calculate_half_life("DrugA", t_half_h=8.0)
        expected = result.n_half_lives_to_steady_state * result.t_half_h
        assert abs(result.tss_h - expected) < 0.01

    def test_tss_h_scales_with_t_half(self):
        r1 = calculate_half_life("Drug1", t_half_h=4.0)
        r2 = calculate_half_life("Drug2", t_half_h=8.0)
        ratio = r2.tss_h / r1.tss_h
        assert abs(ratio - 2.0) < 0.05

    def test_tss_h_approx_value(self):
        result = calculate_half_life("DrugA", t_half_h=10.0)
        assert abs(result.tss_h - 43.2) < 1.0


# ---------------------------------------------------------------------------
# 4. accumulation_ratio >= 1
# ---------------------------------------------------------------------------

class TestAccumulationRatio:
    def test_accumulation_ratio_geq_1_default(self, basic_result):
        assert basic_result.accumulation_ratio >= 1.0

    def test_accumulation_ratio_geq_1_short_interval(self):
        result = calculate_half_life("DrugA", t_half_h=12.0, dosing_interval_h=6.0)
        assert result.accumulation_ratio >= 1.0

    def test_accumulation_ratio_geq_1_long_interval(self):
        result = calculate_half_life("DrugA", t_half_h=4.0, dosing_interval_h=24.0)
        assert result.accumulation_ratio >= 1.0

    def test_accumulation_ratio_higher_for_shorter_interval(self):
        r_short = calculate_half_life("DrugA", t_half_h=12.0, dosing_interval_h=6.0)
        r_long = calculate_half_life("DrugA", t_half_h=12.0, dosing_interval_h=24.0)
        assert r_short.accumulation_ratio > r_long.accumulation_ratio


# ---------------------------------------------------------------------------
# 5. recommended_doses_per_day ≈ 24 / tau
# ---------------------------------------------------------------------------

class TestRecommendedDosesPerDay:
    def test_doses_per_day_approx_24_over_tau(self):
        tau = 8.0
        result = calculate_half_life("DrugA", t_half_h=6.0, dosing_interval_h=tau)
        expected = 24.0 / tau
        assert abs(result.recommended_doses_per_day - expected) < 0.5

    def test_doses_per_day_once_daily(self):
        result = calculate_half_life("DrugA", t_half_h=20.0, dosing_interval_h=24.0)
        assert abs(result.recommended_doses_per_day - 1.0) < 0.1

    def test_doses_per_day_three_times_daily(self):
        result = calculate_half_life("DrugA", t_half_h=4.0, dosing_interval_h=8.0)
        assert abs(result.recommended_doses_per_day - 3.0) < 0.1


# ---------------------------------------------------------------------------
# 6. ValueError for t_half_h <= 0
# ---------------------------------------------------------------------------

class TestValueError:
    def test_zero_t_half_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_half_life("DrugA", t_half_h=0.0)

    def test_negative_t_half_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_half_life("DrugA", t_half_h=-5.0)

    def test_very_small_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_half_life("DrugA", t_half_h=-0.001)

    def test_optimal_dosing_negative_t_half_raises_value_error(self):
        with pytest.raises(ValueError):
            optimal_dosing_interval("DrugA", t_half_h=-1.0)


# ---------------------------------------------------------------------------
# 7. optimal_dosing_interval: tau ≈ t_half when R = 2
# ---------------------------------------------------------------------------

class TestOptimalDosingInterval:
    def test_tau_equals_t_half_for_r2(self):
        t_half = 8.0
        result = optimal_dosing_interval("DrugA", t_half_h=t_half, target_accumulation_ratio=2.0)
        assert abs(result.dosing_interval_h - t_half) < 0.5

    def test_tau_equals_t_half_for_r2_different_drug(self):
        t_half = 24.0
        result = optimal_dosing_interval("DrugB", t_half_h=t_half, target_accumulation_ratio=2.0)
        assert abs(result.dosing_interval_h - t_half) < 1.0

    def test_higher_target_ratio_gives_shorter_interval(self):
        r2 = optimal_dosing_interval("DrugA", t_half_h=10.0, target_accumulation_ratio=2.0)
        r4 = optimal_dosing_interval("DrugA", t_half_h=10.0, target_accumulation_ratio=4.0)
        assert r4.dosing_interval_h < r2.dosing_interval_h

    def test_optimal_result_accumulation_ratio_near_target(self):
        target = 3.0
        result = optimal_dosing_interval("DrugA", t_half_h=12.0, target_accumulation_ratio=target)
        assert abs(result.accumulation_ratio - target) < 0.2


# ---------------------------------------------------------------------------
# 8. half_life_from_data
# ---------------------------------------------------------------------------

class TestHalfLifeFromData:
    def _monoexp_data(self, t_half, c0=100.0, n_points=20):
        ke = math.log(2) / t_half
        times = np.linspace(0, 4 * t_half, n_points)
        conc = c0 * np.exp(-ke * times)
        return times.tolist(), conc.tolist()

    def test_monoexp_returns_float(self):
        times, conc = self._monoexp_data(8.0)
        result = half_life_from_data(times, conc)
        assert isinstance(result, float)

    def test_monoexp_t_half_within_10_pct(self):
        true_t_half = 8.0
        times, conc = self._monoexp_data(true_t_half)
        estimated = half_life_from_data(times, conc)
        assert abs(estimated - true_t_half) / true_t_half < 0.10

    def test_monoexp_short_half_life(self):
        true_t_half = 2.0
        times, conc = self._monoexp_data(true_t_half)
        estimated = half_life_from_data(times, conc)
        assert abs(estimated - true_t_half) / true_t_half < 0.10

    def test_monoexp_long_half_life(self):
        true_t_half = 48.0
        times, conc = self._monoexp_data(true_t_half)
        estimated = half_life_from_data(times, conc)
        assert abs(estimated - true_t_half) / true_t_half < 0.10

    def test_len_less_than_3_raises_value_error(self):
        with pytest.raises(ValueError):
            half_life_from_data([0.0, 1.0], [100.0, 50.0])

    def test_len_exactly_2_raises_value_error(self):
        with pytest.raises(ValueError):
            half_life_from_data([0.0, 8.0], [100.0, 50.0])

    def test_len_1_raises_value_error(self):
        with pytest.raises(ValueError):
            half_life_from_data([0.0], [100.0])

    def test_len_0_raises_value_error(self):
        with pytest.raises(ValueError):
            half_life_from_data([], [])

    def test_result_is_positive(self):
        times, conc = self._monoexp_data(10.0)
        result = half_life_from_data(times, conc)
        assert result > 0.0


# ---------------------------------------------------------------------------
# 9. t_half_h preserved in result
# ---------------------------------------------------------------------------

class TestTHalfPreserved:
    def test_t_half_preserved_calculate(self):
        t_half = 13.5
        result = calculate_half_life("DrugA", t_half_h=t_half)
        assert result.t_half_h == pytest.approx(t_half)

    def test_t_half_preserved_optimal(self):
        t_half = 6.75
        result = optimal_dosing_interval("DrugA", t_half_h=t_half)
        assert result.t_half_h == pytest.approx(t_half)

    def test_drug_name_preserved(self):
        result = calculate_half_life("Warfarin", t_half_h=40.0)
        assert result.drug_name == "Warfarin"

    def test_drug_name_preserved_optimal(self):
        result = optimal_dosing_interval("Metformin", t_half_h=5.0)
        assert result.drug_name == "Metformin"
