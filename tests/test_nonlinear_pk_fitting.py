"""Tests for Phase 258: nonlinear_pk_fitting.py"""

import math

import numpy as np
import pytest

from omega_pbpk.analysis.nonlinear_pk_fitting import (
    NonlinearPKFitResult,
    _mm_pk_model,
    fit_nonlinear_pk,
)

# --- Helpers ----------------------------------------------------------------


def _generate_mm_data(
    vmax: float = 10.0,
    km: float = 2.0,
    vd: float = 30.0,
    dose: float = 100.0,
    t_end: float = 12.0,
    n_points: int = 12,
    noise_frac: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Generate synthetic MM-PK concentration-time data."""
    times = np.linspace(0.5, t_end, n_points)
    c0 = dose / vd
    c_pred = _mm_pk_model(times, vmax, km, vd, c0)
    if noise_frac > 0:
        rng = np.random.default_rng(42)
        c_pred = c_pred * (1 + noise_frac * rng.standard_normal(len(c_pred)))
        c_pred = np.clip(c_pred, 0.01, None)
    return times.tolist(), c_pred.tolist()


# --- Input validation -------------------------------------------------------


def test_too_few_timepoints_raises():
    with pytest.raises(ValueError, match="4 time points"):
        fit_nonlinear_pk("Drug", [1.0, 2.0, 3.0], [1.0, 0.8, 0.6], dose_mg=100.0)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        fit_nonlinear_pk("Drug", [1.0, 2.0, 3.0, 4.0], [1.0, 0.8, 0.6], dose_mg=100.0)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        fit_nonlinear_pk("Drug", [1, 2, 3, 4], [1, 0.8, 0.6, 0.4], dose_mg=0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        fit_nonlinear_pk("Drug", [1, 2, 3, 4], [1, 0.8, 0.6, 0.4], dose_mg=-50)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_guess_L"):
        fit_nonlinear_pk("Drug", [1, 2, 3, 4], [1, 0.8, 0.6, 0.4], dose_mg=100, vd_guess_L=0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_guess_L"):
        fit_nonlinear_pk("Drug", [1, 2, 3, 4], [1, 0.8, 0.6, 0.4], dose_mg=100, vd_guess_L=-10)


# --- Basic fit quality ------------------------------------------------------


def test_vmax_close_to_true():
    """Fitted Vmax should be within 30% of true value for noise-free data."""
    true_vmax = 10.0
    times, concs = _generate_mm_data(vmax=true_vmax, km=2.0)
    result = fit_nonlinear_pk("Phenytoin", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.vmax_mg_per_h == pytest.approx(true_vmax, rel=0.30)


def test_km_close_to_true():
    """Fitted Km should be within 50% of true value for noise-free data."""
    true_km = 2.0
    times, concs = _generate_mm_data(vmax=10.0, km=true_km)
    result = fit_nonlinear_pk("Phenytoin", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.km_mg_L == pytest.approx(true_km, rel=0.50)


def test_r_squared_high_for_good_fit():
    """R² should be close to 1 for noise-free synthetic data."""
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.r_squared >= 0.95


def test_r_squared_is_valid_float():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert isinstance(result.r_squared, float)
    assert not math.isnan(result.r_squared)


def test_rmse_nonnegative():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert result.rmse >= 0.0


def test_rmse_small_for_good_fit():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    # RMSE should be very small for noise-free data
    assert result.rmse < 0.5


def test_aafe_at_least_one():
    """AAFE >= 1.0 by definition."""
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.aafe >= 1.0


def test_predicted_length_matches_timepoints():
    times, concs = _generate_mm_data(n_points=10)
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert len(result.predicted) == len(times)


def test_vmax_positive():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert result.vmax_mg_per_h > 0


def test_km_positive():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert result.km_mg_L > 0


def test_drug_name_stored():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Phenytoin", times, concs, dose_mg=100.0)
    assert result.drug_name == "Phenytoin"


def test_result_type():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert isinstance(result, NonlinearPKFitResult)


def test_vd_stored_in_result():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.vd_L == pytest.approx(30.0)


def test_notes_nonempty():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert len(result.notes) > 0


def test_with_custom_vmax_km_guess():
    """Providing custom initial guesses should still converge."""
    times, concs = _generate_mm_data(vmax=10.0, km=2.0)
    result = fit_nonlinear_pk(
        "Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0, vmax_guess=15.0, km_guess=3.0
    )
    assert result.r_squared >= 0.90


def test_noisy_data_still_fits():
    """With modest noise, fit should still be reasonable."""
    times, concs = _generate_mm_data(noise_frac=0.05)
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0, vd_guess_L=30.0)
    assert result.vmax_mg_per_h > 0
    assert result.km_mg_L > 0
    assert result.rmse >= 0


def test_predicted_concentrations_nonnegative():
    times, concs = _generate_mm_data()
    result = fit_nonlinear_pk("Drug", times, concs, dose_mg=100.0)
    assert all(c >= 0 for c in result.predicted)
