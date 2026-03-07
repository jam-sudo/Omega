"""Tests for analysis/exposure_response_model module (Phase 677)."""

import math

import pytest

from omega_pbpk.analysis.exposure_response_model import (
    ExposureResponseResult,
    er_quartile_analysis,
    fit_logistic_er,
    predict_response_probability,
)

# ---------------------------------------------------------------------------
# predict_response_probability
# ---------------------------------------------------------------------------


def test_predict_response_probability_returns_float():
    p = predict_response_probability(1.0, 0.0, 1.0)
    assert isinstance(p, float)


def test_predict_response_probability_bounds():
    for exp in [-100.0, 0.0, 1.0, 10.0, 100.0]:
        p = predict_response_probability(exp, 0.0, 1.0)
        assert 0.0 <= p <= 1.0


def test_predict_response_probability_large_positive_beta():
    # With large positive beta and large exposure → P near 1
    p = predict_response_probability(100.0, 0.0, 5.0)
    assert p > 0.99


def test_predict_response_probability_large_negative_exposure():
    # Large negative exposure with positive beta → P near 0
    p = predict_response_probability(-100.0, 0.0, 5.0)
    assert p < 0.01


def test_predict_response_probability_zero_exposure():
    # P(0) = 1/(1+exp(-alpha))
    alpha = 1.5
    beta = 2.0
    p = predict_response_probability(0.0, alpha, beta)
    expected = 1.0 / (1.0 + math.exp(-alpha))
    assert abs(p - expected) < 1e-9


def test_predict_response_probability_symmetry():
    # P(x) + P(-x) = 1 when alpha=0
    x = 3.0
    p_pos = predict_response_probability(x, 0.0, 1.0)
    p_neg = predict_response_probability(-x, 0.0, 1.0)
    assert abs(p_pos + p_neg - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# fit_logistic_er
# ---------------------------------------------------------------------------


def test_fit_logistic_er_returns_dict():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0]
    responses = [0, 0, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    assert isinstance(result, dict)


def test_fit_logistic_er_keys():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0]
    responses = [0, 0, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    for key in ("alpha", "beta", "ec50", "hill", "auc_roc", "n_subjects", "notes"):
        assert key in result, f"Missing key: {key}"


def test_fit_logistic_er_n_subjects():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0]
    responses = [0, 0, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    assert result["n_subjects"] == 5


def test_fit_logistic_er_auc_roc_range():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    assert 0.0 <= result["auc_roc"] <= 1.0


def test_fit_logistic_er_hill_is_1():
    exposures = [1.0, 2.0, 3.0, 4.0]
    responses = [0, 0, 1, 1]
    result = fit_logistic_er(exposures, responses)
    assert result["hill"] == 1


def test_fit_logistic_er_ec50_formula():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    alpha = result["alpha"]
    beta = result["beta"]
    ec50 = result["ec50"]
    if not math.isnan(ec50) and abs(beta) > 1e-10:
        assert abs(ec50 - (-alpha / beta)) < 1e-9


def test_fit_logistic_er_positive_beta_for_increasing_data():
    # Clear increasing response with exposure → beta should be positive
    exposures = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = fit_logistic_er(exposures, responses)
    assert result["beta"] >= 0


def test_fit_logistic_er_notes_nonempty():
    exposures = [1.0, 2.0, 3.0, 4.0]
    responses = [0, 1, 0, 1]
    result = fit_logistic_er(exposures, responses)
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_fit_logistic_er_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        fit_logistic_er([1.0, 2.0], [0])


def test_fit_logistic_er_responses_not_binary_raises():
    with pytest.raises(ValueError, match="0 or 1"):
        fit_logistic_er([1.0, 2.0, 3.0, 4.0], [0, 1, 2, 0])


def test_fit_logistic_er_too_few_subjects_raises():
    with pytest.raises(ValueError, match="at least 4"):
        fit_logistic_er([1.0, 2.0, 3.0], [0, 1, 0])


# ---------------------------------------------------------------------------
# ExposureResponseResult dataclass
# ---------------------------------------------------------------------------


def test_exposure_response_result_creation():
    result = ExposureResponseResult(
        compound_name="TestDrug",
        model_type="logistic",
        alpha=-2.0,
        beta=1.0,
        ec50=2.0,
        auc_roc=0.85,
        n_subjects=20,
        notes="Test notes",
    )
    assert result.compound_name == "TestDrug"
    assert result.alpha == -2.0
    assert result.beta == 1.0
    assert result.ec50 == 2.0
    assert result.auc_roc == 0.85
    assert result.n_subjects == 20


def test_exposure_response_result_is_frozen():
    result = ExposureResponseResult(
        compound_name="TestDrug",
        model_type="logistic",
        alpha=0.0,
        beta=1.0,
        ec50=0.0,
        auc_roc=0.7,
        n_subjects=10,
        notes="",
    )
    with pytest.raises((AttributeError, TypeError)):
        result.alpha = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# er_quartile_analysis
# ---------------------------------------------------------------------------


def test_er_quartile_analysis_returns_dict():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert isinstance(result, dict)


def test_er_quartile_analysis_keys():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert "quartile_bounds" in result
    assert "response_rates" in result
    assert "trend" in result
    assert "notes" in result


def test_er_quartile_analysis_quartile_bounds_len():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert len(result["quartile_bounds"]) == 5


def test_er_quartile_analysis_response_rates_len():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert len(result["response_rates"]) == 4


def test_er_quartile_analysis_trend_values():
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    responses = [0, 0, 0, 0, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert result["trend"] in ("increasing", "decreasing", "flat")


def test_er_quartile_analysis_increasing_trend():
    # Low exposure → no response, high exposure → response
    exposures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    responses = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    result = er_quartile_analysis(exposures, responses)
    assert result["trend"] == "increasing"


def test_er_quartile_analysis_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        er_quartile_analysis([1.0, 2.0, 3.0], [0, 1])


def test_er_quartile_analysis_too_few_raises():
    with pytest.raises(ValueError, match="at least 4"):
        er_quartile_analysis([1.0, 2.0], [0, 1])
