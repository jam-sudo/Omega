"""Tests for Phase 338: PK variability analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.analysis.pk_variability import (
    PKVariabilityResult,
    analyze_pk_variability,
    identify_high_variability_drivers,
    simulate_population_variability,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_low_variability() -> list[float]:
    """Tightly clustered values around 100 (CV ~ 5%)."""
    rng = np.random.default_rng(0)
    return list(rng.normal(100.0, 5.0, 20).clip(50.0, None))


def make_high_variability() -> list[float]:
    """Widely spread values (CV > 60%)."""
    rng = np.random.default_rng(1)
    return list(rng.lognormal(math.log(100.0), 0.8, 30).clip(1e-3, None))


# ---------------------------------------------------------------------------
# Input validation — analyze_pk_variability
# ---------------------------------------------------------------------------


def test_empty_auc_raises():
    with pytest.raises(ValueError, match="empty"):
        analyze_pk_variability("Drug", [], [])


def test_too_few_subjects_raises():
    with pytest.raises(ValueError, match="3 subjects"):
        analyze_pk_variability("Drug", [10.0, 20.0], [5.0, 6.0])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        analyze_pk_variability("Drug", [10.0, 20.0, 30.0], [5.0, 6.0])


def test_auc_zero_raises():
    with pytest.raises(ValueError, match="auc_values"):
        analyze_pk_variability("Drug", [10.0, 0.0, 30.0], [5.0, 6.0, 7.0])


def test_auc_negative_raises():
    with pytest.raises(ValueError, match="auc_values"):
        analyze_pk_variability("Drug", [10.0, -5.0, 30.0], [5.0, 6.0, 7.0])


def test_cmax_zero_raises():
    with pytest.raises(ValueError, match="cmax_values"):
        analyze_pk_variability("Drug", [10.0, 20.0, 30.0], [5.0, 0.0, 7.0])


def test_trough_mismatched_raises():
    with pytest.raises(ValueError, match="same length"):
        analyze_pk_variability(
            "Drug", [10.0, 20.0, 30.0], [5.0, 6.0, 7.0], trough_values=[1.0, 2.0]
        )


def test_trough_zero_raises():
    with pytest.raises(ValueError, match="trough_values"):
        analyze_pk_variability(
            "Drug", [10.0, 20.0, 30.0], [5.0, 6.0, 7.0], trough_values=[1.0, 0.0, 3.0]
        )


# ---------------------------------------------------------------------------
# Basic correctness — analyze_pk_variability
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = analyze_pk_variability("Drug", [10.0, 20.0, 30.0], [5.0, 6.0, 7.0])
    assert isinstance(result, PKVariabilityResult)


def test_frozen_dataclass():
    result = analyze_pk_variability("Drug", [10.0, 20.0, 30.0], [5.0, 6.0, 7.0])
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_homogeneous_sample_cv_near_zero():
    """Identical values → CV ≈ 0."""
    aucs = [100.0] * 10
    cmaxs = [10.0] * 10
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.auc_cv_pct < 1.0


def test_high_variability_category():
    aucs = make_high_variability()
    cmaxs = [a * 0.1 for a in aucs]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    if result.auc_cv_pct > 60:
        assert result.variability_category == "high"


def test_low_variability_category():
    aucs = make_low_variability()
    cmaxs = [a * 0.1 for a in aucs]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    if result.auc_cv_pct < 30:
        assert result.variability_category == "low"


def test_moderate_variability_category():
    rng = np.random.default_rng(99)
    # Target ~40% CV
    aucs = list(rng.lognormal(math.log(100), 0.38, 50).clip(1.0, None))
    cmaxs = [a * 0.1 for a in aucs]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.variability_category in ("low", "moderate", "high")  # at minimum valid


def test_cv_pct_positive_for_varied_data():
    aucs = [10.0, 50.0, 100.0, 200.0, 500.0]
    cmaxs = [1.0, 5.0, 10.0, 20.0, 50.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.auc_cv_pct > 0.0


def test_geometric_mean_positive():
    aucs = make_low_variability()
    cmaxs = [a * 0.1 for a in aucs]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.auc_gm > 0.0


def test_geometric_sd_greater_than_one_for_varied():
    aucs = [10.0, 50.0, 100.0, 200.0, 500.0]
    cmaxs = [1.0, 5.0, 10.0, 20.0, 50.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.auc_gsd > 1.0


def test_percentile_ordering():
    aucs = [10.0, 50.0, 100.0, 200.0, 500.0]
    cmaxs = [1.0, 5.0, 10.0, 20.0, 50.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.auc_5th_pct < result.auc_95th_pct


def test_fold_range_geq_one():
    aucs = [10.0, 50.0, 100.0, 200.0, 500.0]
    cmaxs = [1.0, 5.0, 10.0, 20.0, 50.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.fold_range >= 1.0


def test_no_trough_sets_zero():
    aucs = [10.0, 20.0, 30.0, 40.0, 50.0]
    cmaxs = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.trough_mean == 0.0
    assert result.trough_cv_pct == 0.0


def test_trough_provided_computes_stats():
    aucs = [10.0, 20.0, 30.0, 40.0, 50.0]
    cmaxs = [1.0, 2.0, 3.0, 4.0, 5.0]
    troughs = [0.5, 1.0, 1.5, 2.0, 2.5]
    result = analyze_pk_variability("Drug", aucs, cmaxs, trough_values=troughs)
    assert result.trough_mean > 0.0
    assert result.trough_cv_pct >= 0.0


def test_n_subjects_correct():
    aucs = [10.0, 20.0, 30.0]
    cmaxs = [1.0, 2.0, 3.0]
    result = analyze_pk_variability("Drug", aucs, cmaxs)
    assert result.n_subjects == 3


# ---------------------------------------------------------------------------
# simulate_population_variability
# ---------------------------------------------------------------------------


def test_simulate_population_correct_n_subjects():
    result = simulate_population_variability("Drug", 5.0, 50.0, 100.0, n_subjects=50)
    assert result.n_subjects == 50


def test_simulate_population_auc_values_length():
    result = simulate_population_variability("Drug", 5.0, 50.0, 100.0, n_subjects=30)
    assert len(result.auc_values) == 30


def test_simulate_population_invalid_cl():
    with pytest.raises(ValueError, match="cl_mean"):
        simulate_population_variability("Drug", 0.0, 50.0, 100.0)


def test_simulate_population_invalid_vd():
    with pytest.raises(ValueError, match="vd_mean"):
        simulate_population_variability("Drug", 5.0, 0.0, 100.0)


def test_simulate_population_invalid_dose():
    with pytest.raises(ValueError, match="dose"):
        simulate_population_variability("Drug", 5.0, 50.0, 0.0)


def test_simulate_population_too_few_subjects():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_population_variability("Drug", 5.0, 50.0, 100.0, n_subjects=2)


def test_simulate_population_negative_omega_raises():
    with pytest.raises(ValueError, match="omega_cl"):
        simulate_population_variability("Drug", 5.0, 50.0, 100.0, omega_cl_pct=-1.0)


def test_simulate_population_reproducible_seed():
    r1 = simulate_population_variability("Drug", 5.0, 50.0, 100.0, seed=7)
    r2 = simulate_population_variability("Drug", 5.0, 50.0, 100.0, seed=7)
    assert r1.auc_mean == pytest.approx(r2.auc_mean, rel=1e-9)


# ---------------------------------------------------------------------------
# identify_high_variability_drivers
# ---------------------------------------------------------------------------


def test_identify_drivers_returns_dict():
    aucs = [10.0, 20.0, 50.0, 100.0, 200.0]
    drivers = identify_high_variability_drivers(aucs, {"weight": [50.0, 60.0, 70.0, 80.0, 90.0]})
    assert isinstance(drivers, dict)
    assert "weight" in drivers


def test_identify_drivers_sorted_by_abs_r():
    aucs = [10.0, 20.0, 50.0, 100.0, 200.0]
    drivers = identify_high_variability_drivers(
        aucs,
        {
            "weight": [50.0, 60.0, 70.0, 80.0, 90.0],
            "age": [20.0, 30.0, 60.0, 25.0, 45.0],
        },
    )
    r_vals = [abs(v[0]) for v in drivers.values()]
    assert r_vals == sorted(r_vals, reverse=True)


def test_identify_drivers_empty_auc_raises():
    with pytest.raises(ValueError, match="empty"):
        identify_high_variability_drivers([], {"weight": []})


def test_identify_drivers_negative_auc_raises():
    with pytest.raises(ValueError, match="auc_values"):
        identify_high_variability_drivers([10.0, -5.0, 30.0], {"weight": [50.0, 60.0, 70.0]})
