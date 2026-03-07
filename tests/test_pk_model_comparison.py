"""Tests for analysis/pk_model_comparison.py — Phase 652."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_model_comparison import (
    PKModelFitResult,
    compare_pk_models,
    fit_one_compartment,
    fit_two_compartment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_monoexp_data(c0: float = 2.0, ke: float = 0.3, n: int = 12) -> tuple:
    """Generate monoexponential IV concentration-time data."""
    times = [i * 0.5 for i in range(n)]
    concs = [c0 * math.exp(-ke * t) for t in times]
    return times, concs


def make_biexp_data(
    a: float = 1.5, alpha: float = 2.0, b: float = 0.5, beta: float = 0.2, n: int = 12
) -> tuple:
    """Generate biexponential IV concentration-time data."""
    times = [i * 0.5 for i in range(n)]
    concs = [a * math.exp(-alpha * t) + b * math.exp(-beta * t) for t in times]
    return times, concs


# ---------------------------------------------------------------------------
# fit_one_compartment — basic
# ---------------------------------------------------------------------------


def test_fit_one_compartment_returns_result():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert isinstance(result, PKModelFitResult)


def test_fit_two_compartment_returns_result():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert isinstance(result, PKModelFitResult)


def test_rss_non_negative_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert result.rss >= 0.0


def test_rmse_non_negative_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert result.rmse >= 0.0


def test_rss_non_negative_2cpt():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert result.rss >= 0.0


def test_rmse_non_negative_2cpt():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert result.rmse >= 0.0


def test_r_squared_in_valid_range_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert -10.0 <= result.r_squared <= 1.0


def test_r_squared_in_valid_range_2cpt():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert -10.0 <= result.r_squared <= 1.0


def test_aic_is_finite_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert math.isfinite(result.aic)


def test_bic_is_finite_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert math.isfinite(result.bic)


def test_n_parameters_1cpt_iv():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0, route="iv")
    assert result.n_parameters == 2


def test_n_parameters_2cpt():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert result.n_parameters == 4


def test_2cpt_has_more_params_than_1cpt():
    times, concs = make_monoexp_data()
    r1 = fit_one_compartment(times, concs, dose_mg=100.0)
    r2 = fit_two_compartment(times, concs, dose_mg=100.0)
    assert r2.n_parameters > r1.n_parameters


# ---------------------------------------------------------------------------
# compare_pk_models
# ---------------------------------------------------------------------------


def test_compare_returns_list_of_length_2():
    times, concs = make_monoexp_data()
    results = compare_pk_models(times, concs, dose_mg=100.0)
    assert len(results) == 2


def test_compare_one_selected():
    times, concs = make_monoexp_data()
    results = compare_pk_models(times, concs, dose_mg=100.0)
    n_selected = sum(1 for r in results if r.selected)
    assert n_selected == 1


def test_compare_min_delta_aic_is_zero():
    times, concs = make_monoexp_data()
    results = compare_pk_models(times, concs, dose_mg=100.0)
    min_delta = min(r.delta_aic for r in results)
    assert abs(min_delta) < 1e-9


def test_compare_sorted_by_aic_ascending():
    times, concs = make_monoexp_data()
    results = compare_pk_models(times, concs, dose_mg=100.0)
    aic_vals = [r.aic for r in results]
    assert aic_vals == sorted(aic_vals)


def test_akaike_weights_sum_to_one():
    times, concs = make_monoexp_data()
    results = compare_pk_models(times, concs, dose_mg=100.0)
    total = sum(r.akaike_weight for r in results)
    assert abs(total - 1.0) < 1e-9


def test_monoexp_data_1cpt_selected():
    """True monoexponential data should favor 1-cpt model."""
    times, concs = make_monoexp_data(c0=2.0, ke=0.3, n=15)
    results = compare_pk_models(times, concs, dose_mg=100.0)
    selected = next(r for r in results if r.selected)
    assert "1-cpt" in selected.model_name


def test_monoexp_r_squared_greater_than_0_5():
    """Well-fitted 1-cpt to monoexp data should have high R²."""
    times, concs = make_monoexp_data(c0=2.0, ke=0.3, n=12)
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert result.r_squared > 0.5


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty_1cpt():
    times, concs = make_monoexp_data()
    result = fit_one_compartment(times, concs, dose_mg=100.0)
    assert len(result.notes) > 0


def test_notes_nonempty_2cpt():
    times, concs = make_monoexp_data()
    result = fit_two_compartment(times, concs, dose_mg=100.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_too_few_observations():
    with pytest.raises(ValueError):
        fit_one_compartment([0.0, 1.0], [1.0, 0.5], dose_mg=100.0)


def test_validation_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_one_compartment([0.0, 1.0, 2.0], [1.0, 0.5], dose_mg=100.0)


def test_validation_zero_dose():
    times, concs = make_monoexp_data()
    with pytest.raises(ValueError):
        fit_one_compartment(times, concs, dose_mg=0.0)


def test_validation_non_monotone_times():
    with pytest.raises(ValueError):
        fit_one_compartment([0.0, 2.0, 1.0, 3.0], [2.0, 1.0, 1.2, 0.8], dose_mg=100.0)
