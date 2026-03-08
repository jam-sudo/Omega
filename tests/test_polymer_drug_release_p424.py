"""
Tests for Phase 424: Polymer Drug Release Prediction
~22 tests covering result type, profile properties, polymer comparisons,
particle size effects, and validation errors.
"""

import pytest

from omega_pbpk.prediction.polymer_drug_release_p424 import (
    POLYMERS,
    PolymerReleaseResult,
    compare_polymers,
    simulate_polymer_release,
)

# ─── Return type ─────────────────────────────────────────────────────────────


def test_return_type():
    result = simulate_polymer_release("drug_A", "PLGA_50_50", 10.0, 20.0)
    assert isinstance(result, PolymerReleaseResult)


# ─── Fields preservation ─────────────────────────────────────────────────────


def test_drug_name_preserved():
    result = simulate_polymer_release("myDrug", "PLGA_50_50", 10.0, 20.0)
    assert result.drug_name == "myDrug"


def test_polymer_type_preserved():
    result = simulate_polymer_release("myDrug", "PLGA_75_25", 10.0, 20.0)
    assert result.polymer_type == "PLGA_75_25"


# ─── Time / cumulative shape ──────────────────────────────────────────────────


def test_times_start_at_zero():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    assert result.times_days[0] == pytest.approx(0.0)


def test_cumulative_starts_at_burst_pct():
    """At t=0 the burst is instantaneously released; cumulative equals burst_factor * 100."""
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    expected_burst_pct = POLYMERS["PLGA_50_50"]["burst_factor"] * 100.0
    assert result.cumulative_release_pct[0] == pytest.approx(expected_burst_pct, rel=1e-6)


def test_cumulative_monotonically_increases():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    for i in range(1, len(result.cumulative_release_pct)):
        assert result.cumulative_release_pct[i] >= result.cumulative_release_pct[i - 1] - 1e-9


def test_final_cumulative_leq_100():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    assert result.cumulative_release_pct[-1] <= 100.0 + 1e-9


def test_arrays_same_length():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    n = len(result.times_days)
    assert len(result.cumulative_release_pct) == n
    assert len(result.release_rate_pct_per_day) == n


# ─── Burst release ────────────────────────────────────────────────────────────


def test_burst_release_nonnegative():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0)
    assert result.burst_release_pct >= 0.0


def test_burst_release_at_day1():
    """burst_release_pct should match cumulative at ~1 day."""
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0, dt_days=1.0)
    # Day 1 is index 1
    assert result.burst_release_pct == pytest.approx(result.cumulative_release_pct[1], rel=1e-6)


# ─── t50 / t90 ────────────────────────────────────────────────────────────────


def test_t50_positive():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0, t_end_days=90.0)
    assert result.t50_days > 0.0
    assert result.t50_days != float("inf")


def test_t50_less_than_t_end_for_fast_polymer():
    result = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0, t_end_days=90.0)
    assert result.t50_days < 90.0


# ─── Polymer comparison ───────────────────────────────────────────────────────


def test_plga5050_faster_than_pla():
    """PLGA_50_50 has shorter half-life than PLA -> faster release."""
    r1 = simulate_polymer_release("drug", "PLGA_50_50", 10.0, 20.0, t_end_days=200.0)
    r2 = simulate_polymer_release("drug", "PLA", 10.0, 20.0, t_end_days=200.0)
    assert r1.t50_days < r2.t50_days


def test_smaller_particle_faster_release():
    """Smaller particle -> higher particle_size_factor -> faster release."""
    r_small = simulate_polymer_release("drug", "PLGA_50_50", 2.0, 20.0, t_end_days=90.0)
    r_large = simulate_polymer_release("drug", "PLGA_50_50", 50.0, 20.0, t_end_days=90.0)
    assert r_small.t50_days < r_large.t50_days


# ─── compare_polymers ─────────────────────────────────────────────────────────


def test_compare_polymers_returns_4():
    results = compare_polymers("drug", 10.0, 20.0)
    assert len(results) == 4


def test_compare_polymers_sorted_ascending():
    results = compare_polymers("drug", 10.0, 20.0)
    t50_values = [r.t50_days for r in results]
    assert t50_values == sorted(t50_values)


def test_compare_polymers_all_polymer_types():
    results = compare_polymers("drug", 10.0, 20.0)
    polymer_types = {r.polymer_type for r in results}
    assert polymer_types == set(POLYMERS.keys())


# ─── Validation errors ────────────────────────────────────────────────────────


def test_invalid_polymer_raises():
    with pytest.raises(ValueError, match="polymer_type"):
        simulate_polymer_release("drug", "UNKNOWN_POLYMER", 10.0, 20.0)


def test_particle_diameter_zero_raises():
    with pytest.raises(ValueError, match="particle_diameter"):
        simulate_polymer_release("drug", "PLGA_50_50", 0.0, 20.0)


def test_particle_diameter_negative_raises():
    with pytest.raises(ValueError, match="particle_diameter"):
        simulate_polymer_release("drug", "PLGA_50_50", -5.0, 20.0)


def test_drug_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading"):
        simulate_polymer_release("drug", "PLGA_50_50", 10.0, 0.0)


def test_drug_loading_over_100_raises():
    with pytest.raises(ValueError, match="drug_loading"):
        simulate_polymer_release("drug", "PLGA_50_50", 10.0, 101.0)
