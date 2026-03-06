"""Tests for concentration-response modeling module (Phase 230)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.analysis.concentration_response import (
    EmaxFitResult,
    fit_emax_model,
    fit_hill_model,
    ic50_from_fit,
    predict_response,
    therapeutic_index,
)


def _make_synthetic_emax_data(
    e0: float = 5.0,
    emax: float = 90.0,
    ec50: float = 1.0,
    hill_n: float = 1.5,
    n_points: int = 12,
    log_c_min: float = -2.0,
    log_c_max: float = 2.0,
) -> tuple[list[float], list[float]]:
    """Generate synthetic Emax data for testing."""
    concentrations = [10**x for x in np.linspace(log_c_min, log_c_max, n_points)]
    responses = [e0 + emax * (c**hill_n) / (ec50**hill_n + c**hill_n) for c in concentrations]
    return concentrations, responses


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fewer_than_4_points_raises(self):
        with pytest.raises(ValueError, match="4"):
            fit_emax_model([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    def test_exactly_3_points_raises_for_hill(self):
        with pytest.raises(ValueError, match="4"):
            fit_hill_model([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    def test_mismatched_lengths_raises_emax(self):
        with pytest.raises(ValueError, match="length"):
            fit_emax_model([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0])

    def test_mismatched_lengths_raises_hill(self):
        with pytest.raises(ValueError, match="length"):
            fit_hill_model([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0])

    def test_non_positive_concentration_raises_emax(self):
        with pytest.raises(ValueError):
            fit_emax_model([0.0, 1.0, 2.0, 3.0], [5.0, 15.0, 25.0, 35.0])

    def test_non_positive_concentration_raises_hill(self):
        with pytest.raises(ValueError):
            fit_hill_model([-1.0, 1.0, 2.0, 3.0], [5.0, 15.0, 25.0, 35.0])

    def test_predict_response_zero_concentration_raises(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        with pytest.raises(ValueError):
            predict_response(fit, [0.0, 1.0])

    def test_therapeutic_index_zero_efficacy_raises(self):
        with pytest.raises(ValueError):
            therapeutic_index(0.0, 5.0)

    def test_therapeutic_index_zero_toxicity_raises(self):
        with pytest.raises(ValueError):
            therapeutic_index(1.0, 0.0)


# ---------------------------------------------------------------------------
# fit_emax_model tests
# ---------------------------------------------------------------------------


class TestFitEmaxModel:
    def test_fits_synthetic_data_e0(self):
        concs, resps = _make_synthetic_emax_data(e0=5.0, emax=90.0, ec50=1.0)
        fit = fit_emax_model(concs, resps)
        assert abs(fit.e0 - 5.0) / 5.0 < 0.10  # within 10%

    def test_fits_synthetic_data_emax(self):
        concs, resps = _make_synthetic_emax_data(e0=5.0, emax=90.0, ec50=1.0)
        fit = fit_emax_model(concs, resps)
        assert abs(fit.emax - 90.0) / 90.0 < 0.10

    def test_fits_synthetic_data_ec50(self):
        concs, resps = _make_synthetic_emax_data(e0=5.0, emax=90.0, ec50=1.0)
        fit = fit_emax_model(concs, resps)
        assert abs(fit.ec50 - 1.0) / 1.0 < 0.10

    def test_r_squared_high_for_noiseless_data(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert fit.r_squared > 0.98

    def test_predict_response_at_ec50(self):
        """At C=EC50, response should be E0 + Emax/2."""
        e0, emax, ec50 = 5.0, 90.0, 1.0
        concs, resps = _make_synthetic_emax_data(e0=e0, emax=emax, ec50=ec50)
        fit = fit_emax_model(concs, resps)
        pred = predict_response(fit, [fit.ec50])
        expected = fit.e0 + fit.emax / 2.0
        assert abs(pred[0] - expected) < 5.0  # within 5 response units

    def test_result_is_emax_fit_result(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert isinstance(fit, EmaxFitResult)

    def test_model_name_is_emax(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert fit.model == "emax"

    def test_ec50_is_positive(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert fit.ec50 > 0

    def test_hill_n_is_positive(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert fit.hill_n > 0

    def test_slope_at_ec50_positive(self):
        concs, resps = _make_synthetic_emax_data(emax=80.0)
        fit = fit_emax_model(concs, resps)
        assert fit.slope_at_ec50 > 0

    def test_result_field_types(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        assert isinstance(fit.model, str)
        assert isinstance(fit.e0, float)
        assert isinstance(fit.emax, float)
        assert isinstance(fit.ec50, float)
        assert isinstance(fit.hill_n, float)
        assert isinstance(fit.r_squared, float)
        assert isinstance(fit.rmse, float)
        assert isinstance(fit.aic, float)
        assert isinstance(fit.c_range, tuple)
        assert isinstance(fit.concentrations_fit, list)
        assert isinstance(fit.responses_fit, list)
        assert isinstance(fit.responses_predicted, list)
        assert isinstance(fit.slope_at_ec50, float)
        assert isinstance(fit.notes, str)


# ---------------------------------------------------------------------------
# fit_hill_model tests
# ---------------------------------------------------------------------------


class TestFitHillModel:
    def test_hill_model_ec50_close_to_emax_ec50(self):
        """Hill and Emax models should give similar EC50 for the same data."""
        concs, resps = _make_synthetic_emax_data(e0=0.0, emax=100.0, ec50=2.0, hill_n=1.0)
        emax_fit = fit_emax_model(concs, resps)
        hill_fit = fit_hill_model(concs, resps)
        # EC50s should be within 20% of each other
        ratio = abs(hill_fit.ec50 - emax_fit.ec50) / emax_fit.ec50
        assert ratio < 0.20

    def test_model_name_is_hill(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_hill_model(concs, resps)
        assert fit.model == "hill"

    def test_r_squared_positive(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_hill_model(concs, resps)
        assert fit.r_squared >= 0.0

    def test_slope_at_ec50_positive(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_hill_model(concs, resps)
        assert fit.slope_at_ec50 > 0


# ---------------------------------------------------------------------------
# ic50_from_fit tests
# ---------------------------------------------------------------------------


class TestIc50FromFit:
    def test_returns_positive_value(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        ic50 = ic50_from_fit(fit)
        assert ic50 > 0

    def test_equals_ec50(self):
        concs, resps = _make_synthetic_emax_data()
        fit = fit_emax_model(concs, resps)
        ic50 = ic50_from_fit(fit)
        assert ic50 == fit.ec50


# ---------------------------------------------------------------------------
# therapeutic_index tests
# ---------------------------------------------------------------------------


class TestTherapeuticIndex:
    def test_ti_greater_than_one_when_tox_ec50_larger(self):
        ti = therapeutic_index(ec50_efficacy=1.0, ec50_toxicity=10.0)
        assert ti > 1.0

    def test_ti_equals_one_when_ec50s_equal(self):
        ti = therapeutic_index(ec50_efficacy=5.0, ec50_toxicity=5.0)
        assert math.isclose(ti, 1.0)

    def test_ti_less_than_one_when_tox_ec50_smaller(self):
        ti = therapeutic_index(ec50_efficacy=10.0, ec50_toxicity=1.0)
        assert ti < 1.0

    def test_ti_is_ratio(self):
        ti = therapeutic_index(ec50_efficacy=2.0, ec50_toxicity=10.0)
        assert math.isclose(ti, 5.0)
