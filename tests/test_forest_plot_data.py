"""Tests for forest plot data and meta-analysis module."""

import math

import pytest

from omega_pbpk.analysis.forest_plot_data import (
    ForestPlotResult,
    StudyEffect,
    auc_ratio_to_log_effect,
    compute_forest_plot,
    create_study_effect,
)


def _make_studies(effects, se=0.1, n=20):
    """Helper to create a list of StudyEffect objects."""
    return [create_study_effect(f"S{i + 1}", es, se, n) for i, es in enumerate(effects)]


def test_create_study_returns_result():
    s = create_study_effect("S1", 0.5, 0.1, 20)
    assert isinstance(s, StudyEffect)


def test_weight_is_inverse_variance():
    se = 0.2
    s = create_study_effect("S1", 0.5, se, 20)
    assert abs(s.weight - 1.0 / (se * se)) < 1e-9


def test_ci_correct():
    se = 0.1
    es = 0.5
    s = create_study_effect("S1", es, se, 20)
    assert abs(s.ci_low - (es - 1.96 * se)) < 1e-9


def test_ci_high_correct():
    se = 0.1
    es = 0.5
    s = create_study_effect("S1", es, se, 20)
    assert abs(s.ci_high - (es + 1.96 * se)) < 1e-9


def test_compute_forest_returns_result():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert isinstance(result, ForestPlotResult)


def test_n_studies_correct():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.n_studies == 3


def test_pooled_effect_in_range():
    effects = [0.3, 0.5, 0.4]
    studies = _make_studies(effects)
    result = compute_forest_plot(studies)
    assert min(effects) <= result.pooled_effect <= max(effects)


def test_pooled_se_positive():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.pooled_se > 0


def test_q_statistic_nonneg():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.q_statistic >= 0


def test_i_squared_in_0_100():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert 0.0 <= result.i_squared_pct <= 100.0


def test_tau_squared_nonneg():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.tau_squared >= 0


def test_re_pooled_effect_positive_for_positive_effects():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.re_pooled_effect > 0


def test_re_ci_low_lt_re_pooled():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.re_ci_low < result.re_pooled_effect


def test_re_ci_high_gt_re_pooled():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.re_ci_high > result.re_pooled_effect


def test_identical_studies_zero_q():
    """3 identical studies → Q ≈ 0, I² = 0."""
    studies = _make_studies([0.5, 0.5, 0.5])
    result = compute_forest_plot(studies)
    assert result.q_statistic < 1e-9
    assert result.i_squared_pct == 0.0


def test_heterogeneous_studies_high_i2():
    """Very different effects → I² > 50%."""
    # Use large variance between studies relative to within-study variance
    studies = [
        create_study_effect("S1", 0.1, 0.05, 50),
        create_study_effect("S2", 2.0, 0.05, 50),
        create_study_effect("S3", 4.0, 0.05, 50),
    ]
    result = compute_forest_plot(studies)
    assert result.i_squared_pct > 50.0


def test_recommendation_valid():
    valid = {
        "low_heterogeneity_fixed_effect_preferred",
        "moderate_heterogeneity_random_effects_preferred",
        "high_heterogeneity_interpret_cautiously",
    }
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert result.recommendation in valid


def test_auc_ratio_to_log_returns_result():
    s = auc_ratio_to_log_effect(2.0, 20)
    assert isinstance(s, StudyEffect)


def test_auc_ratio_log_effect_correct():
    s = auc_ratio_to_log_effect(2.0, 20)
    assert abs(s.effect_size - math.log(2.0)) < 1e-9


def test_auc_ratio_se_positive():
    s = auc_ratio_to_log_effect(2.0, 20)
    assert s.se > 0


def test_single_study_raises():
    studies = _make_studies([0.5])
    with pytest.raises(ValueError):
        compute_forest_plot(studies)


def test_notes_nonempty():
    studies = _make_studies([0.3, 0.5, 0.4])
    result = compute_forest_plot(studies)
    assert len(result.notes) > 0
