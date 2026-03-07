"""Tests for PK concentration-time curve fitting module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_curve_fitting import (
    OneCptFitResult,
    TwoCptFitResult,
    fit_one_compartment_iv,
    fit_one_compartment_oral,
    fit_two_compartment_iv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iv_data(c0: float, ke: float, times: list[float]) -> list[float]:
    """Generate synthetic one-cpt IV data."""
    return [c0 * math.exp(-ke * t) for t in times]


def _make_oral_data(
    ka: float, ke: float, vd: float, dose: float, times: list[float]
) -> list[float]:
    """Generate synthetic one-cpt oral data (F=1)."""
    denom = ka - ke
    return [(dose / vd) * (ka / denom) * (math.exp(-ke * t) - math.exp(-ka * t)) for t in times]


def _make_2cpt_data(
    A: float, alpha: float, B: float, beta: float, times: list[float]
) -> list[float]:
    return [A * math.exp(-alpha * t) + B * math.exp(-beta * t) for t in times]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_fewer_than_3_points_raises(self):
        with pytest.raises(ValueError, match="3"):
            fit_one_compartment_iv([0.0, 1.0], [10.0, 5.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            fit_one_compartment_iv([0.0, 1.0, 2.0], [10.0, 5.0])

    def test_non_increasing_times_raises(self):
        with pytest.raises(ValueError):
            fit_one_compartment_iv([0.0, 2.0, 1.0], [10.0, 7.0, 5.0])

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError):
            fit_one_compartment_iv([0.0, 1.0, 2.0], [10.0, 0.0, 5.0])

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError):
            fit_one_compartment_iv([0.0, 1.0, 2.0], [10.0, -1.0, 5.0])

    def test_oral_negative_dose_raises(self):
        with pytest.raises(ValueError):
            fit_one_compartment_oral([0.5, 1.0, 2.0], [5.0, 8.0, 6.0], dose_mg=-100.0)


# ---------------------------------------------------------------------------
# One-compartment IV tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFitOneCptIV:
    TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    C0 = 50.0
    KE = 0.2  # h⁻¹

    @pytest.fixture
    def synthetic_data(self):
        return _make_iv_data(self.C0, self.KE, self.TIMES)

    @pytest.fixture
    def result(self, synthetic_data):
        return fit_one_compartment_iv(self.TIMES, synthetic_data)

    def test_returns_one_cpt_fit_result(self, result):
        assert isinstance(result, OneCptFitResult)

    def test_model_label(self, result):
        assert result.model == "one_compartment_iv"

    def test_c0_recovered_within_5pct(self, result):
        assert abs(result.c0_or_cmax - self.C0) / self.C0 < 0.05

    def test_ke_recovered_within_5pct(self, result):
        assert abs(result.ke_per_h - self.KE) / self.KE < 0.05

    def test_ka_is_none_for_iv(self, result):
        assert result.ka_per_h is None

    def test_t_half_correct(self, result):
        expected_thalf = math.log(2.0) / self.KE
        assert abs(result.t_half_h - expected_thalf) / expected_thalf < 0.05

    def test_r_squared_near_1_for_noiseless(self, result):
        assert result.r_squared > 0.99

    def test_r_squared_in_0_1(self, result):
        assert 0.0 <= result.r_squared <= 1.0

    def test_rmse_near_zero_for_noiseless(self, result):
        assert result.rmse < 0.5  # relative to C0=50

    def test_rmse_nonnegative(self, result):
        assert result.rmse >= 0.0

    def test_residuals_length(self, result):
        assert len(result.residuals) == len(self.TIMES)

    def test_c_fitted_length(self, result):
        assert len(result.c_fitted) == len(self.TIMES)

    def test_times_fit_matches_input(self, result):
        assert result.times_fit == self.TIMES

    def test_n_params(self, result):
        assert result.n_params == 2

    def test_aic_is_float(self, result):
        assert isinstance(result.aic, float)


# ---------------------------------------------------------------------------
# One-compartment oral tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFitOneCptOral:
    TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    KA = 1.5  # h⁻¹
    KE = 0.15  # h⁻¹
    VD = 30.0  # L
    DOSE = 200.0  # mg

    @pytest.fixture
    def synthetic_data(self):
        return _make_oral_data(self.KA, self.KE, self.VD, self.DOSE, self.TIMES)

    @pytest.fixture
    def result(self, synthetic_data):
        return fit_one_compartment_oral(self.TIMES, synthetic_data, dose_mg=self.DOSE)

    def test_returns_one_cpt_fit_result(self, result):
        assert isinstance(result, OneCptFitResult)

    def test_model_label(self, result):
        assert result.model == "one_compartment_oral"

    def test_ka_not_none(self, result):
        assert result.ka_per_h is not None

    def test_ka_recovered_within_20pct(self, result):
        """ka can be harder to estimate; use 20% tolerance."""
        assert abs(result.ka_per_h - self.KA) / self.KA < 0.20

    def test_ke_recovered_within_10pct(self, result):
        assert abs(result.ke_per_h - self.KE) / self.KE < 0.10

    def test_r_squared_high_for_noiseless(self, result):
        assert result.r_squared > 0.95

    def test_t_half_reasonable(self, result):
        expected = math.log(2.0) / self.KE
        assert 0.5 * expected < result.t_half_h < 3.0 * expected

    def test_residuals_length(self, result):
        assert len(result.residuals) == len(self.TIMES)

    def test_n_params(self, result):
        assert result.n_params == 3

    def test_c0_or_cmax_positive(self, result):
        assert result.c0_or_cmax > 0


# ---------------------------------------------------------------------------
# Two-compartment IV tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFitTwoCptIV:
    TIMES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    A = 40.0
    ALPHA = 2.0  # h⁻¹ (fast distribution)
    B = 20.0
    BETA = 0.1  # h⁻¹ (slow elimination)

    @pytest.fixture
    def synthetic_data(self):
        return _make_2cpt_data(self.A, self.ALPHA, self.B, self.BETA, self.TIMES)

    @pytest.fixture
    def result(self, synthetic_data):
        return fit_two_compartment_iv(self.TIMES, synthetic_data)

    def test_returns_two_cpt_fit_result(self, result):
        assert isinstance(result, TwoCptFitResult)

    def test_model_label(self, result):
        assert result.model == "two_compartment_iv"

    def test_a_plus_b_approx_c0(self, result):
        """A+B should approximate C(0) = A+B by definition."""
        expected_c0 = self.A + self.B
        assert abs(result.A + result.B - expected_c0) / expected_c0 < 0.10

    def test_alpha_greater_than_beta(self, result):
        """Distribution phase should be faster than elimination."""
        assert result.alpha > result.beta

    def test_t_half_alpha_shorter_than_beta(self, result):
        assert result.t_half_alpha < result.t_half_beta

    def test_r_squared_high_for_noiseless(self, result):
        assert result.r_squared > 0.99

    def test_r_squared_in_range(self, result):
        assert 0.0 <= result.r_squared <= 1.0

    def test_rmse_nonnegative(self, result):
        assert result.rmse >= 0.0

    def test_residuals_length(self, result):
        assert len(result.residuals) == len(self.TIMES)

    def test_c_fitted_length(self, result):
        assert len(result.c_fitted) == len(self.TIMES)

    def test_n_params(self, result):
        assert result.n_params == 4

    def test_t_half_beta_correct(self, result):
        expected = math.log(2.0) / self.BETA
        assert abs(result.t_half_beta - expected) / expected < 0.10

    def test_a_positive(self, result):
        assert result.A > 0

    def test_b_positive(self, result):
        assert result.B > 0
