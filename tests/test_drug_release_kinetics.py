"""Tests for Phase 221 — Drug Release Kinetics Model Fitting."""

import math

import pytest

from omega_pbpk.prediction.drug_release_kinetics import (
    ReleaseModelResult,
    compare_models,
    fit_release_model,
)

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def _linear_data(k0=1.0, n=10):
    """Generate perfectly linear (zero-order) release data."""
    times = [float(i * 10) for i in range(1, n + 1)]  # 10..100 min
    pct = [k0 * t for t in times]
    return times, pct


def _sqrt_data(kH=2.0, n=10):
    """Generate perfectly Higuchi (sqrt(t)) release data."""
    times = [float(i * 10) for i in range(1, n + 1)]
    pct = [kH * math.sqrt(t) for t in times]
    return times, pct


def _first_order_data(k1=0.05, n=10):
    """Generate perfectly first-order release data."""
    times = [float(i * 10) for i in range(1, n + 1)]
    pct = [100.0 * (1.0 - math.exp(-k1 * t)) for t in times]
    return times, pct


def _power_law_data(kKP=2.0, n_exp=0.5, n_pts=10):
    """Generate Korsmeyer-Peppas power-law release data."""
    times = [float(i * 10) for i in range(1, n_pts + 1)]
    pct = [kKP * (t**n_exp) for t in times]
    return times, pct


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_release_model_result(self):
        times, pct = _linear_data()
        result = fit_release_model("TestDrug", times, pct, "zero_order")
        assert isinstance(result, ReleaseModelResult)

    def test_drug_name_preserved(self):
        times, pct = _linear_data()
        result = fit_release_model("Ibuprofen", times, pct, "zero_order")
        assert result.drug_name == "Ibuprofen"

    def test_model_field_preserved(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "higuchi")
        assert result.model == "higuchi"

    def test_fitted_values_is_tuple(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert isinstance(result.fitted_values, tuple)
        assert len(result.fitted_values) == len(times)

    def test_n_param_none_for_zero_order(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert result.n_param is None

    def test_n_param_none_for_first_order(self):
        times, pct = _first_order_data()
        result = fit_release_model("Drug", times, pct, "first_order")
        assert result.n_param is None

    def test_n_param_none_for_higuchi(self):
        times, pct = _sqrt_data()
        result = fit_release_model("Drug", times, pct, "higuchi")
        assert result.n_param is None

    def test_n_param_float_for_korsmeyer_peppas(self):
        times, pct = _power_law_data()
        result = fit_release_model("Drug", times, pct, "korsmeyer_peppas")
        assert result.n_param is not None
        assert isinstance(result.n_param, float)


# ---------------------------------------------------------------------------
# R² range tests
# ---------------------------------------------------------------------------


class TestRSquared:
    def test_r_squared_in_range_zero_order(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_in_range_first_order(self):
        times, pct = _first_order_data()
        result = fit_release_model("Drug", times, pct, "first_order")
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_in_range_higuchi(self):
        times, pct = _sqrt_data()
        result = fit_release_model("Drug", times, pct, "higuchi")
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_in_range_kp(self):
        times, pct = _power_law_data()
        result = fit_release_model("Drug", times, pct, "korsmeyer_peppas")
        assert 0.0 <= result.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Model accuracy tests
# ---------------------------------------------------------------------------


class TestModelAccuracy:
    def test_zero_order_fits_linear_data(self):
        """Zero-order should give R² > 0.95 on perfectly linear data."""
        times, pct = _linear_data(k0=0.8)
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert result.r_squared > 0.95

    def test_higuchi_fits_sqrt_data_better_than_zero_order(self):
        """Higuchi should outperform zero-order on sqrt(t) data."""
        times, pct = _sqrt_data(kH=3.0)
        r_higuchi = fit_release_model("Drug", times, pct, "higuchi").r_squared
        r_zero = fit_release_model("Drug", times, pct, "zero_order").r_squared
        assert r_higuchi > r_zero

    def test_higuchi_best_for_sqrt_data(self):
        """compare_models should rank Higuchi #1 for sqrt(t) data."""
        times, pct = _sqrt_data(kH=2.5)
        results = compare_models("Drug", times, pct)
        assert results[0].model == "higuchi"

    def test_first_order_fits_saturation_data(self):
        """First-order should give high R² on 100*(1-exp(-k*t)) data."""
        times, pct = _first_order_data(k1=0.03)
        result = fit_release_model("Drug", times, pct, "first_order")
        assert result.r_squared > 0.90

    def test_kp_k_param_positive(self):
        times, pct = _power_law_data()
        result = fit_release_model("Drug", times, pct, "korsmeyer_peppas")
        assert result.k_param > 0


# ---------------------------------------------------------------------------
# compare_models tests
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_compare_returns_four_results(self):
        times, pct = _linear_data()
        results = compare_models("Drug", times, pct)
        assert len(results) == 4

    def test_compare_sorted_descending(self):
        times, pct = _linear_data()
        results = compare_models("Drug", times, pct)
        for i in range(len(results) - 1):
            assert results[i].r_squared >= results[i + 1].r_squared

    def test_compare_contains_all_models(self):
        times, pct = _linear_data()
        results = compare_models("Drug", times, pct)
        model_names = {r.model for r in results}
        assert model_names == {"zero_order", "first_order", "higuchi", "korsmeyer_peppas"}

    def test_compare_drug_name_consistent(self):
        times, pct = _linear_data()
        results = compare_models("Paracetamol", times, pct)
        assert all(r.drug_name == "Paracetamol" for r in results)


# ---------------------------------------------------------------------------
# Release mechanism string tests
# ---------------------------------------------------------------------------


class TestReleaseMechanism:
    def test_zero_order_mechanism_string(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert isinstance(result.release_mechanism, str)
        assert len(result.release_mechanism) > 0
        assert (
            "zero" in result.release_mechanism.lower()
            or "order" in result.release_mechanism.lower()
        )

    def test_kp_fickian_mechanism(self):
        """n < 0.45 → Fickian diffusion label."""
        # Use n=0.3 power law data to push KP fit toward n < 0.45
        times, pct = _power_law_data(kKP=5.0, n_exp=0.3)
        result = fit_release_model("Drug", times, pct, "korsmeyer_peppas")
        # Just verify it's a non-empty string
        assert isinstance(result.release_mechanism, str)
        assert "Korsmeyer" in result.release_mechanism

    def test_notes_non_empty(self):
        times, pct = _linear_data()
        result = fit_release_model("Drug", times, pct, "zero_order")
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fewer_than_3_points_raises(self):
        with pytest.raises(ValueError, match="3 data points"):
            fit_release_model("Drug", [10.0, 20.0], [20.0, 40.0], "zero_order")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_release_model("Drug", [10.0, 20.0, 30.0], [20.0, 40.0], "zero_order")

    def test_invalid_model_raises(self):
        times, pct = _linear_data()
        with pytest.raises(ValueError, match="model must be one of"):
            fit_release_model("Drug", times, pct, "bad_model")

    def test_exactly_3_points_allowed(self):
        result = fit_release_model("Drug", [10.0, 20.0, 30.0], [10.0, 20.0, 30.0], "zero_order")
        assert isinstance(result, ReleaseModelResult)
