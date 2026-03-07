"""Tests for Phase 684 — Drug washout time calculator."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.drug_washout_calculator import (
    WashoutResult,
    WashoutSimResult,
    estimate_washout_time,
    multi_dose_washout,
    simulate_washout_profile,
)

# ---------------------------------------------------------------------------
# estimate_washout_time
# ---------------------------------------------------------------------------


def test_estimate_washout_time_returns_result():
    result = estimate_washout_time(12.0)
    assert isinstance(result, WashoutResult)


def test_estimate_washout_time_n_half_lives_for_1pct():
    """1% remaining requires ~6.644 half-lives."""
    result = estimate_washout_time(10.0, target_fraction_remaining=0.01)
    assert abs(result.n_half_lives - math.log(100.0) / math.log(2.0)) < 1e-6


def test_estimate_washout_time_washout_time_equals_n_times_thalf():
    t_half = 8.0
    result = estimate_washout_time(t_half, target_fraction_remaining=0.01)
    expected = result.n_half_lives * t_half
    assert abs(result.washout_time_h - expected) < 1e-9


def test_estimate_washout_time_five_half_life_rule():
    t_half = 6.0
    result = estimate_washout_time(t_half)
    assert abs(result.five_half_life_time_h - 5.0 * t_half) < 1e-9


def test_estimate_washout_time_5_half_lives_not_1pct():
    """After 5 half-lives ~3.125% remains (not 1%)."""
    t_half = 10.0
    result = estimate_washout_time(t_half, target_fraction_remaining=0.01)
    fraction_after_5 = 0.5**5
    assert abs(fraction_after_5 - 0.03125) < 1e-6
    # The 5 half-life time gives more residual than the 1% target
    assert result.five_half_life_time_h < result.washout_time_h


def test_estimate_washout_time_n_half_lives_near_664():
    """n ≈ 6.644 for target=0.01."""
    result = estimate_washout_time(1.0, target_fraction_remaining=0.01)
    assert abs(result.n_half_lives - 6.6439) < 0.001


def test_estimate_washout_time_residual_fraction_equals_target():
    target = 0.05
    result = estimate_washout_time(12.0, target_fraction_remaining=target)
    assert abs(result.residual_fraction - target) < 1e-12


def test_estimate_washout_time_stores_t_half():
    result = estimate_washout_time(24.0, target_fraction_remaining=0.01)
    assert abs(result.t_half_h - 24.0) < 1e-9


def test_estimate_washout_time_notes_non_empty():
    result = estimate_washout_time(8.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# estimate_washout_time — validation
# ---------------------------------------------------------------------------


def test_estimate_washout_time_t_half_zero_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        estimate_washout_time(0.0)


def test_estimate_washout_time_t_half_negative_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        estimate_washout_time(-5.0)


def test_estimate_washout_time_target_fraction_zero_raises():
    with pytest.raises(ValueError, match="target_fraction"):
        estimate_washout_time(12.0, target_fraction_remaining=0.0)


def test_estimate_washout_time_target_fraction_one_raises():
    with pytest.raises(ValueError, match="target_fraction"):
        estimate_washout_time(12.0, target_fraction_remaining=1.0)


def test_estimate_washout_time_target_fraction_gt1_raises():
    with pytest.raises(ValueError):
        estimate_washout_time(12.0, target_fraction_remaining=1.5)


# ---------------------------------------------------------------------------
# simulate_washout_profile
# ---------------------------------------------------------------------------


def test_simulate_washout_profile_returns_result():
    result = simulate_washout_profile(12.0)
    assert isinstance(result, WashoutSimResult)


def test_simulate_washout_profile_c0_equals_dose_over_vd():
    dose, vd = 200.0, 40.0
    result = simulate_washout_profile(10.0, dose_mg=dose, vd_L=vd)
    assert abs(result.c0_mg_L - dose / vd) < 1e-9


def test_simulate_washout_profile_times_list_populated():
    result = simulate_washout_profile(12.0)
    assert len(result.times_h) > 0


def test_simulate_washout_profile_concs_list_populated():
    result = simulate_washout_profile(12.0)
    assert len(result.c_plasma_mg_L) > 0


def test_simulate_washout_profile_lists_same_length():
    result = simulate_washout_profile(12.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_simulate_washout_profile_concs_strictly_decreasing():
    result = simulate_washout_profile(12.0, dt_h=0.5)
    concs = result.c_plasma_mg_L
    for i in range(1, len(concs)):
        assert concs[i] < concs[i - 1], f"Not decreasing at index {i}"


def test_simulate_washout_profile_time_order():
    """time_to_50pct < time_to_10pct < time_to_5pct < time_to_1pct."""
    result = simulate_washout_profile(10.0, t_end_h=None)
    assert result.time_to_50pct_h < result.time_to_10pct_h
    assert result.time_to_10pct_h < result.time_to_5pct_h
    assert result.time_to_5pct_h < result.time_to_1pct_h


def test_simulate_washout_profile_time_to_50pct_approx_thalf():
    """Time to 50% of C0 ≈ t_half (within 5% of simulation step tolerance)."""
    t_half = 10.0
    result = simulate_washout_profile(t_half, dt_h=0.1, t_end_h=80.0)
    # Should be within one simulation step of the true half-life
    assert abs(result.time_to_50pct_h - t_half) < 0.6  # within ~1 step


def test_simulate_washout_profile_t_half_stored():
    result = simulate_washout_profile(15.0)
    assert abs(result.t_half_h - 15.0) < 1e-9


def test_simulate_washout_profile_default_end_is_7_half_lives():
    t_half = 10.0
    result = simulate_washout_profile(t_half, dt_h=0.5)
    # Time array should extend to ~7*t_half
    assert max(result.times_h) >= 7.0 * t_half - 0.5


# ---------------------------------------------------------------------------
# simulate_washout_profile — validation
# ---------------------------------------------------------------------------


def test_simulate_washout_profile_t_half_zero_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        simulate_washout_profile(0.0)


def test_simulate_washout_profile_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_washout_profile(10.0, dose_mg=0.0)


def test_simulate_washout_profile_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_washout_profile(10.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# multi_dose_washout
# ---------------------------------------------------------------------------


def test_multi_dose_washout_returns_dict():
    result = multi_dose_washout([100.0], [0.0], 10.0)
    assert isinstance(result, dict)


def test_multi_dose_washout_dict_has_required_keys():
    result = multi_dose_washout([100.0], [0.0], 10.0)
    assert "c_at_start" in result
    assert "washout_time_h" in result
    assert "notes" in result


def test_multi_dose_washout_single_dose_washout_time_positive():
    result = multi_dose_washout([100.0], [0.0], 10.0, target_fraction=0.01)
    assert result["washout_time_h"] > 0.0


def test_multi_dose_washout_repeated_doses_longer_washout():
    """Multiple equal doses should require longer washout than single dose."""
    single = multi_dose_washout([100.0], [0.0], 10.0, target_fraction=0.01)
    multi = multi_dose_washout(
        [100.0, 100.0, 100.0],
        [0.0, 12.0, 24.0],
        10.0,
        target_fraction=0.01,
        t_eval_start_h=24.0,
    )
    assert multi["washout_time_h"] >= single["washout_time_h"]


def test_multi_dose_washout_c_at_start_positive():
    result = multi_dose_washout([100.0, 100.0], [0.0, 12.0], 10.0, t_eval_start_h=12.0)
    assert result["c_at_start"] > 0.0


def test_multi_dose_washout_notes_non_empty():
    result = multi_dose_washout([100.0], [0.0], 10.0)
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# multi_dose_washout — validation
# ---------------------------------------------------------------------------


def test_multi_dose_washout_t_half_zero_raises():
    with pytest.raises(ValueError, match="t_half_h"):
        multi_dose_washout([100.0], [0.0], 0.0)


def test_multi_dose_washout_target_fraction_one_raises():
    with pytest.raises(ValueError, match="target_fraction"):
        multi_dose_washout([100.0], [0.0], 10.0, target_fraction=1.0)


def test_multi_dose_washout_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        multi_dose_washout([100.0], [0.0], 10.0, vd_L=0.0)
