"""Tests for Phase 590 — Sample Size and Statistical Power Analysis."""

import pytest

from omega_pbpk.analysis.power_analysis import (
    be_sample_size,
    crossover_vs_parallel_n,
    minimum_detectable_difference,
    power_t_test,
    sample_size_t_test,
)

# ---------------------------------------------------------------------------
# power_t_test tests
# ---------------------------------------------------------------------------


def test_power_t_test_large_effect_high_power():
    # n=100, large effect → high power
    power = power_t_test(n=100, delta=1.0, sigma=1.0)
    assert power > 0.90


def test_power_t_test_small_n_small_effect_low_power():
    # n=2, tiny effect → very low power
    power = power_t_test(n=2, delta=0.1, sigma=1.0)
    assert power < 0.20


def test_power_t_test_in_range():
    power = power_t_test(n=30, delta=0.5, sigma=1.0)
    assert 0.0 <= power <= 1.0


def test_power_t_test_increases_with_n():
    p1 = power_t_test(n=20, delta=0.5, sigma=1.0)
    p2 = power_t_test(n=100, delta=0.5, sigma=1.0)
    assert p2 > p1


def test_power_t_test_invalid_alpha_zero_raises():
    with pytest.raises(ValueError):
        power_t_test(n=50, delta=1.0, sigma=1.0, alpha=0.0)


def test_power_t_test_invalid_alpha_one_raises():
    with pytest.raises(ValueError):
        power_t_test(n=50, delta=1.0, sigma=1.0, alpha=1.0)


# ---------------------------------------------------------------------------
# sample_size_t_test tests
# ---------------------------------------------------------------------------


def test_sample_size_t_test_larger_delta_smaller_n():
    n_small_delta = sample_size_t_test(delta=0.2, sigma=1.0)
    n_large_delta = sample_size_t_test(delta=1.0, sigma=1.0)
    assert n_large_delta < n_small_delta


def test_sample_size_t_test_higher_power_larger_n():
    n_80 = sample_size_t_test(delta=0.5, sigma=1.0, power=0.80)
    n_90 = sample_size_t_test(delta=0.5, sigma=1.0, power=0.90)
    assert n_90 > n_80


def test_sample_size_t_test_returns_int():
    n = sample_size_t_test(delta=0.5, sigma=1.0)
    assert isinstance(n, int)


def test_sample_size_t_test_invalid_power_raises():
    with pytest.raises(ValueError):
        sample_size_t_test(delta=0.5, sigma=1.0, power=1.0)


def test_sample_size_t_test_achieved_power():
    # Verify the returned n actually achieves the requested power
    delta, sigma = 0.5, 1.0
    n = sample_size_t_test(delta=delta, sigma=sigma, power=0.80)
    assert power_t_test(n=n, delta=delta, sigma=sigma) >= 0.80


# ---------------------------------------------------------------------------
# be_sample_size tests
# ---------------------------------------------------------------------------


def test_be_sample_size_returns_dict():
    result = be_sample_size(cv_intra=0.20)
    assert "n_per_group" in result
    assert "n_total" in result
    assert "cv_intra" in result


def test_be_sample_size_minimum_12():
    # Very low CV → formula might give <12, but regulatory minimum is 12
    result = be_sample_size(cv_intra=0.01)
    assert result["n_per_group"] >= 12


def test_be_sample_size_higher_cv_larger_n():
    n_low = be_sample_size(cv_intra=0.10)["n_per_group"]
    n_high = be_sample_size(cv_intra=0.40)["n_per_group"]
    assert n_high > n_low


def test_be_sample_size_n_total_double_per_group():
    result = be_sample_size(cv_intra=0.25)
    assert result["n_total"] == result["n_per_group"] * 2


def test_be_sample_size_cv_preserved():
    result = be_sample_size(cv_intra=0.30)
    assert result["cv_intra"] == 0.30


# ---------------------------------------------------------------------------
# minimum_detectable_difference tests
# ---------------------------------------------------------------------------


def test_mdd_larger_n_smaller_delta():
    mdd_small_n = minimum_detectable_difference(n=10, sigma=1.0)
    mdd_large_n = minimum_detectable_difference(n=100, sigma=1.0)
    assert mdd_large_n < mdd_small_n


def test_mdd_positive():
    mdd = minimum_detectable_difference(n=50, sigma=1.0)
    assert mdd > 0.0


def test_mdd_scales_with_sigma():
    mdd_1 = minimum_detectable_difference(n=50, sigma=1.0)
    mdd_2 = minimum_detectable_difference(n=50, sigma=2.0)
    assert abs(mdd_2 / mdd_1 - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# crossover_vs_parallel_n tests
# ---------------------------------------------------------------------------


def test_crossover_vs_parallel_crossover_smaller_or_equal():
    result = crossover_vs_parallel_n(cv_intra=0.20, cv_inter=0.30, delta=0.5)
    assert result["n_crossover"] <= result["n_parallel"]


def test_crossover_vs_parallel_efficiency_gain_gte_1():
    result = crossover_vs_parallel_n(cv_intra=0.20, cv_inter=0.30, delta=0.5)
    assert result["efficiency_gain"] >= 1.0


def test_crossover_vs_parallel_returns_keys():
    result = crossover_vs_parallel_n(cv_intra=0.15, cv_inter=0.25, delta=0.3)
    assert "n_crossover" in result
    assert "n_parallel" in result
    assert "efficiency_gain" in result


def test_crossover_vs_parallel_zero_inter_cv():
    # With zero inter-CV, crossover and parallel should be equally efficient
    result = crossover_vs_parallel_n(cv_intra=0.20, cv_inter=0.0, delta=0.5)
    assert result["efficiency_gain"] == pytest.approx(1.0, abs=0.01)
