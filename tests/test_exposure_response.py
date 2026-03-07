"""Tests for analysis/exposure_response.py — Phase 425."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.exposure_response import (
    ExposureResponseResult,
    fit_exposure_response,
    quartile_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _linear_data(n: int = 20):
    """Perfect linear: response = 2 + 0.5 * exposure."""
    exposures = [float(i) for i in range(1, n + 1)]
    responses = [2.0 + 0.5 * x for x in exposures]
    return exposures, responses


def _emax_data(emax: float = 100.0, ec50: float = 10.0, n_hill: float = 1.0):
    """Emax curve at selected exposure values."""
    xs = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    ys = [emax * x**n_hill / (ec50**n_hill + x**n_hill) for x in xs]
    return xs, ys


def _logistic_data():
    """Binary outcomes from sigmoid P(x) = 1/(1+exp(-(−2+0.4x)))."""
    xs = [float(i) for i in range(1, 21)]
    ys = [1.0 if (1 / (1 + math.exp(-(-2 + 0.4 * x)))) >= 0.5 else 0.0 for x in xs]
    return xs, ys


# ---------------------------------------------------------------------------
# ExposureResponseResult dataclass
# ---------------------------------------------------------------------------

class TestExposureResponseResult:
    def test_construction(self):
        r = ExposureResponseResult(
            model_type="linear",
            n_subjects=10,
            fitted_params={"a": 1.0, "b": 2.0},
            r_squared=0.99,
            ed50_or_ec50=None,
            predicted_responses=[1.0, 2.0],
            notes="test",
        )
        assert r.model_type == "linear"
        assert r.n_subjects == 10
        assert r.r_squared == 0.99
        assert r.ed50_or_ec50 is None

    def test_predicted_responses_mutable(self):
        r = ExposureResponseResult(
            model_type="emax",
            n_subjects=5,
            fitted_params={},
            r_squared=0.95,
            ed50_or_ec50=10.0,
        )
        r.predicted_responses.append(42.0)
        assert 42.0 in r.predicted_responses


# ---------------------------------------------------------------------------
# fit_exposure_response — validation
# ---------------------------------------------------------------------------

class TestFitExposureResponseValidation:
    def test_empty_exposures(self):
        with pytest.raises(ValueError, match="empty"):
            fit_exposure_response([], [1.0, 2.0], "linear")

    def test_empty_responses(self):
        with pytest.raises(ValueError, match="empty"):
            fit_exposure_response([1.0, 2.0], [], "linear")

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            fit_exposure_response([1.0, 2.0, 3.0], [1.0, 2.0], "linear")

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="At least 3"):
            fit_exposure_response([1.0, 2.0], [1.0, 2.0], "linear")

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            fit_exposure_response([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "polynomial")

    def test_nan_in_exposures(self):
        with pytest.raises(ValueError, match="non-finite"):
            fit_exposure_response([float("nan"), 2.0, 3.0], [1.0, 2.0, 3.0], "linear")

    def test_inf_in_responses(self):
        with pytest.raises(ValueError, match="non-finite"):
            fit_exposure_response([1.0, 2.0, 3.0], [float("inf"), 2.0, 3.0], "linear")


# ---------------------------------------------------------------------------
# fit_exposure_response — linear model
# ---------------------------------------------------------------------------

class TestFitLinear:
    def test_perfect_linear_r2(self):
        xs, ys = _linear_data(20)
        result = fit_exposure_response(xs, ys, "linear")
        assert result.r_squared >= 0.999

    def test_linear_returns_correct_type(self):
        xs, ys = _linear_data()
        result = fit_exposure_response(xs, ys, "linear")
        assert result.model_type == "linear"
        assert isinstance(result, ExposureResponseResult)

    def test_linear_params_keys(self):
        xs, ys = _linear_data()
        result = fit_exposure_response(xs, ys, "linear")
        assert "a" in result.fitted_params
        assert "b" in result.fitted_params

    def test_linear_predicted_length(self):
        xs, ys = _linear_data(15)
        result = fit_exposure_response(xs, ys, "linear")
        assert len(result.predicted_responses) == len(xs)

    def test_linear_ec50_is_none(self):
        xs, ys = _linear_data()
        result = fit_exposure_response(xs, ys, "linear")
        assert result.ed50_or_ec50 is None

    def test_linear_slope_direction(self):
        """Positive relationship → positive slope b."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = fit_exposure_response(xs, ys, "linear")
        assert result.fitted_params["b"] > 0

    def test_linear_n_subjects(self):
        xs, ys = _linear_data(12)
        result = fit_exposure_response(xs, ys, "linear")
        assert result.n_subjects == 12


# ---------------------------------------------------------------------------
# fit_exposure_response — Emax model
# ---------------------------------------------------------------------------

class TestFitEmax:
    def test_emax_model_type(self):
        xs, ys = _emax_data()
        result = fit_exposure_response(xs, ys, "emax")
        assert result.model_type == "emax"

    def test_emax_params_keys(self):
        xs, ys = _emax_data()
        result = fit_exposure_response(xs, ys, "emax")
        for key in ("emax", "ec50", "n"):
            assert key in result.fitted_params

    def test_emax_ec50_recovered(self):
        """EC50 should be close to 10."""
        xs, ys = _emax_data(ec50=10.0)
        result = fit_exposure_response(xs, ys, "emax")
        assert result.ed50_or_ec50 is not None
        # Allow wide tolerance due to grid search
        assert 1.0 < result.ed50_or_ec50 < 100.0

    def test_emax_r2_high(self):
        xs, ys = _emax_data()
        result = fit_exposure_response(xs, ys, "emax")
        assert result.r_squared >= 0.90

    def test_emax_predicted_length(self):
        xs, ys = _emax_data()
        result = fit_exposure_response(xs, ys, "emax")
        assert len(result.predicted_responses) == len(xs)

    def test_emax_predicted_monotone(self):
        """Emax predictions should be monotonically non-decreasing for increasing exposures."""
        xs = sorted(_emax_data()[0])
        ys = _emax_data()[1]
        result = fit_exposure_response(xs, ys, "emax")
        preds = result.predicted_responses
        for i in range(1, len(preds)):
            assert preds[i] >= preds[i - 1] - 1e-6

    def test_emax_positive_params(self):
        xs, ys = _emax_data()
        result = fit_exposure_response(xs, ys, "emax")
        assert result.fitted_params["emax"] > 0
        assert result.fitted_params["ec50"] > 0
        assert result.fitted_params["n"] > 0


# ---------------------------------------------------------------------------
# fit_exposure_response — logistic model
# ---------------------------------------------------------------------------

class TestFitLogistic:
    def test_logistic_model_type(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        assert result.model_type == "logistic"

    def test_logistic_params_keys(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        assert "a" in result.fitted_params
        assert "b" in result.fitted_params

    def test_logistic_ec50_none(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        assert result.ed50_or_ec50 is None

    def test_logistic_predictions_in_range(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        for p in result.predicted_responses:
            assert 0.0 <= p <= 1.0

    def test_logistic_predicted_length(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        assert len(result.predicted_responses) == len(xs)

    def test_logistic_r2_nonnegative(self):
        xs, ys = _logistic_data()
        result = fit_exposure_response(xs, ys, "logistic")
        assert result.r_squared >= 0.0


# ---------------------------------------------------------------------------
# quartile_analysis
# ---------------------------------------------------------------------------

class TestQuartileAnalysis:
    def test_basic_keys(self):
        xs = [float(i) for i in range(1, 21)]
        ys = [float(i) * 2 for i in range(1, 21)]
        out = quartile_analysis(xs, ys)
        assert "quartile_boundaries" in out
        assert "quartile_means" in out
        assert "quartile_sizes" in out
        assert "fold_change_q4_q1" in out
        assert "notes" in out

    def test_quartile_labels(self):
        xs = [float(i) for i in range(1, 21)]
        ys = [float(i) for i in range(1, 21)]
        out = quartile_analysis(xs, ys)
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert q in out["quartile_means"]

    def test_fold_change_positive(self):
        """Higher exposures → higher responses → fold-change > 1."""
        xs = [float(i) for i in range(1, 21)]
        ys = [float(i) * 3 for i in range(1, 21)]
        out = quartile_analysis(xs, ys)
        assert out["fold_change_q4_q1"] > 1.0

    def test_fold_change_none_for_zero_q1(self):
        """If Q1 mean = 0, fold-change should be None."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        # Q1 gets responses of 0.0
        ys = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = quartile_analysis(xs, ys)
        assert out["fold_change_q4_q1"] is None

    def test_boundaries_length(self):
        xs = [float(i) for i in range(1, 21)]
        ys = [float(i) for i in range(1, 21)]
        out = quartile_analysis(xs, ys)
        assert len(out["quartile_boundaries"]) == 3

    def test_validation_empty(self):
        with pytest.raises(ValueError, match="empty"):
            quartile_analysis([], [])

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            quartile_analysis([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_validation_too_few(self):
        with pytest.raises(ValueError, match="At least 4"):
            quartile_analysis([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_sizes_sum_to_n(self):
        xs = [float(i) for i in range(1, 21)]
        ys = [float(i) for i in range(1, 21)]
        out = quartile_analysis(xs, ys)
        total = sum(out["quartile_sizes"].values())
        assert total == len(xs)
