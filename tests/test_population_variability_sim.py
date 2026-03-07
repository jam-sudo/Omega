"""Tests for analysis/population_variability_sim.py — Phase 398."""

from __future__ import annotations

import random

import pytest

from omega_pbpk.analysis.population_variability_sim import (
    PopulationVariabilityResult,
    _box_muller,
    _lognormal_sample,
    _percentile,
    simulate_population_pk,
    therapeutic_range_analysis,
)

# ---------------------------------------------------------------------------
# Tests: _box_muller
# ---------------------------------------------------------------------------


def test_box_muller_returns_float():
    rng = random.Random(1)
    z = _box_muller(rng)
    assert isinstance(z, float)


def test_box_muller_distribution():
    """Many samples should be approximately N(0,1)."""
    rng = random.Random(42)
    samples = [_box_muller(rng) for _ in range(10000)]
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    assert abs(mean) < 0.1
    assert abs(var - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Tests: _lognormal_sample
# ---------------------------------------------------------------------------


def test_lognormal_sample_positive():
    rng = random.Random(7)
    for _ in range(100):
        val = _lognormal_sample(10.0, 30.0, rng)
        assert val > 0


def test_lognormal_sample_zero_cv():
    """CV=0 should return value close to mean (omega=0)."""
    rng = random.Random(1)
    val = _lognormal_sample(5.0, 0.0, rng)
    assert abs(val - 5.0) < 1e-6


def test_lognormal_sample_mean():
    """Mean of many log-normal samples should be close to expected mean."""
    rng = random.Random(99)
    cv = 30.0
    mean = 10.0
    samples = [_lognormal_sample(mean, cv, rng) for _ in range(5000)]
    sample_mean = sum(samples) / len(samples)
    # Log-normal mean = mu * exp(omega^2 / 2); for large samples should be within 20%
    assert 7.0 < sample_mean < 14.0


# ---------------------------------------------------------------------------
# Tests: _percentile
# ---------------------------------------------------------------------------


def test_percentile_median():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(_percentile(vals, 50.0) - 3.0) < 1e-9


def test_percentile_p0():
    vals = [1.0, 2.0, 3.0]
    assert abs(_percentile(vals, 0.0) - 1.0) < 1e-9


def test_percentile_p100():
    vals = [1.0, 2.0, 3.0]
    assert abs(_percentile(vals, 100.0) - 3.0) < 1e-9


def test_percentile_single_element():
    assert _percentile([7.0], 50.0) == 7.0


def test_percentile_empty():
    assert _percentile([], 50.0) == 0.0


# ---------------------------------------------------------------------------
# Tests: simulate_population_pk — validation
# ---------------------------------------------------------------------------


def test_validate_n_subjects_zero():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_population_pk(0, 100.0, 5.0, 50.0)


def test_validate_dose_nonpositive():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_population_pk(10, 0.0, 5.0, 50.0)


def test_validate_cl_nonpositive():
    with pytest.raises(ValueError, match="cl_mean_L_per_h"):
        simulate_population_pk(10, 100.0, 0.0, 50.0)


def test_validate_vd_nonpositive():
    with pytest.raises(ValueError, match="vd_mean_L"):
        simulate_population_pk(10, 100.0, 5.0, 0.0)


def test_validate_cl_cv_negative():
    with pytest.raises(ValueError, match="cl_cv_pct"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, cl_cv_pct=-1.0)


def test_validate_vd_cv_negative():
    with pytest.raises(ValueError, match="vd_cv_pct"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, vd_cv_pct=-5.0)


def test_validate_t_end_nonpositive():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, t_end_h=0.0)


def test_validate_dt_nonpositive():
    with pytest.raises(ValueError, match="dt_h"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, dt_h=0.0)


def test_validate_interval_nonpositive():
    with pytest.raises(ValueError, match="interval_h"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, interval_h=0.0)


def test_validate_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, n_doses=0)


def test_validate_ka_negative():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, ka_per_h=-1.0)


def test_validate_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, f_oral=0.0)


def test_validate_f_oral_too_high():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_population_pk(10, 100.0, 5.0, 50.0, f_oral=1.5)


# ---------------------------------------------------------------------------
# Tests: simulate_population_pk — behavior
# ---------------------------------------------------------------------------


def test_result_type():
    result = simulate_population_pk(20, 100.0, 5.0, 50.0, seed=42)
    assert isinstance(result, PopulationVariabilityResult)


def test_n_subjects_stored():
    result = simulate_population_pk(30, 100.0, 5.0, 50.0, seed=1)
    assert result.n_subjects == 30
    assert len(result.individual_troughs) == 30
    assert len(result.individual_cl) == 30
    assert len(result.individual_vd) == 30


def test_troughs_nonnegative():
    result = simulate_population_pk(50, 100.0, 5.0, 50.0, seed=10)
    assert all(t >= 0 for t in result.individual_troughs)


def test_percentiles_ordered():
    result = simulate_population_pk(100, 100.0, 5.0, 50.0, seed=42)
    assert result.p5 <= result.p25 <= result.p50 <= result.p75 <= result.p95


def test_cv_trough_positive_with_variability():
    result = simulate_population_pk(50, 100.0, 5.0, 50.0, cl_cv_pct=40.0, seed=42)
    assert result.cv_trough_pct > 0


def test_zero_variability():
    """With CV=0, all subjects should have identical troughs."""
    result = simulate_population_pk(20, 100.0, 5.0, 50.0, cl_cv_pct=0.0, vd_cv_pct=0.0, seed=99)
    troughs = result.individual_troughs
    assert max(troughs) - min(troughs) < 1e-9


def test_seed_reproducibility():
    r1 = simulate_population_pk(20, 100.0, 5.0, 50.0, seed=7)
    r2 = simulate_population_pk(20, 100.0, 5.0, 50.0, seed=7)
    assert r1.individual_troughs == r2.individual_troughs


def test_different_seeds_differ():
    r1 = simulate_population_pk(30, 100.0, 5.0, 50.0, seed=1)
    r2 = simulate_population_pk(30, 100.0, 5.0, 50.0, seed=2)
    # With variability, different seeds should yield different results
    assert r1.individual_troughs != r2.individual_troughs


def test_notes_nonempty():
    result = simulate_population_pk(10, 100.0, 5.0, 50.0)
    assert len(result.notes) > 0


def test_higher_cl_lower_trough():
    """Higher clearance should lead to lower troughs (on average)."""
    r_low_cl = simulate_population_pk(50, 100.0, 2.0, 50.0, cl_cv_pct=0.0, seed=42)
    r_high_cl = simulate_population_pk(50, 100.0, 10.0, 50.0, cl_cv_pct=0.0, seed=42)
    assert r_low_cl.p50 > r_high_cl.p50


# ---------------------------------------------------------------------------
# Tests: therapeutic_range_analysis
# ---------------------------------------------------------------------------


def test_therapeutic_range_all_in():
    troughs = [2.0, 3.0, 4.0, 5.0]
    result = therapeutic_range_analysis(troughs, 1.0, 10.0)
    assert abs(result["pct_in_range"] - 100.0) < 1e-9
    assert abs(result["pct_below"] - 0.0) < 1e-9
    assert abs(result["pct_above"] - 0.0) < 1e-9


def test_therapeutic_range_all_below():
    troughs = [0.1, 0.2, 0.3]
    result = therapeutic_range_analysis(troughs, 1.0, 5.0)
    assert abs(result["pct_below"] - 100.0) < 1e-9
    assert abs(result["pct_in_range"] - 0.0) < 1e-9


def test_therapeutic_range_all_above():
    troughs = [10.0, 20.0, 30.0]
    result = therapeutic_range_analysis(troughs, 1.0, 5.0)
    assert abs(result["pct_above"] - 100.0) < 1e-9
    assert abs(result["pct_in_range"] - 0.0) < 1e-9


def test_therapeutic_range_mixed():
    troughs = [0.5, 2.0, 7.0, 12.0]  # 1 below, 2 in, 1 above (range 1-8)
    result = therapeutic_range_analysis(troughs, 1.0, 8.0)
    assert abs(result["pct_below"] - 25.0) < 1e-9
    assert abs(result["pct_in_range"] - 50.0) < 1e-9
    assert abs(result["pct_above"] - 25.0) < 1e-9


def test_therapeutic_range_sum_to_100():
    troughs = [0.1, 2.0, 5.0, 15.0, 1.5]
    result = therapeutic_range_analysis(troughs, 1.0, 10.0)
    total = result["pct_below"] + result["pct_in_range"] + result["pct_above"]
    assert abs(total - 100.0) < 1e-9


def test_therapeutic_range_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        therapeutic_range_analysis([], 1.0, 5.0)


def test_therapeutic_range_invalid_bounds():
    with pytest.raises(ValueError, match="upper_mg_L"):
        therapeutic_range_analysis([1.0, 2.0], 5.0, 3.0)


def test_therapeutic_range_negative_lower():
    with pytest.raises(ValueError, match="lower_mg_L"):
        therapeutic_range_analysis([1.0], -1.0, 5.0)
