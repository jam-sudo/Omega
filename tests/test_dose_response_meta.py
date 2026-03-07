"""Tests for analysis/dose_response_meta.py (Phase 594)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.dose_response_meta import (
    MetaAnalysisResult,
    StudyResult,
    analyze_study,
    fit_hill_equation,
    meta_analyze,
)


def _study(study_id="S1", emax=100.0, ec50=5.0, n=1.0):
    """Simulate perfect Hill data for testing."""
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [emax * d**n / (ec50**n + d**n) for d in doses]
    return analyze_study(study_id, doses, responses)


def test_fit_returns_tuple():
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [100.0 * d / (5.0 + d) for d in doses]
    result = fit_hill_equation(doses, responses)
    assert len(result) == 4


def test_emax_close_to_true():
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [100.0 * d / (5.0 + d) for d in doses]
    emax, ec50, hill_n, r2 = fit_hill_equation(doses, responses)
    assert abs(emax - 100.0) / 100.0 < 0.20


def test_ec50_close_to_true():
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [100.0 * d / (5.0 + d) for d in doses]
    emax, ec50, hill_n, r2 = fit_hill_equation(doses, responses)
    assert abs(ec50 - 5.0) / 5.0 < 0.50


def test_r_squared_positive():
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [100.0 * d / (5.0 + d) for d in doses]
    emax, ec50, hill_n, r2 = fit_hill_equation(doses, responses)
    assert r2 >= 0.0


def test_r_squared_max_1():
    doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    responses = [100.0 * d / (5.0 + d) for d in doses]
    emax, ec50, hill_n, r2 = fit_hill_equation(doses, responses)
    assert r2 <= 1.0


def test_analyze_study_returns_dataclass():
    s = _study()
    assert isinstance(s, StudyResult)


def test_study_emax_positive():
    s = _study()
    assert s.emax > 0


def test_study_ec50_positive():
    s = _study()
    assert s.ec50 > 0


def test_study_r_squared_in_range():
    s = _study()
    assert 0.0 <= s.r_squared <= 1.0


def test_meta_analyze_returns_result():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    assert isinstance(result, MetaAnalysisResult)


def test_meta_n_studies():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    assert result.n_studies == 2


def test_pooled_emax_close():
    s1 = _study("S1", emax=100.0, ec50=5.0)
    s2 = _study("S2", emax=100.0, ec50=5.0)
    result = meta_analyze([s1, s2])
    assert abs(result.pooled_emax - 100.0) / 100.0 < 0.20


def test_pooled_ec50_close():
    s1 = _study("S1", emax=100.0, ec50=5.0)
    s2 = _study("S2", emax=100.0, ec50=5.0)
    result = meta_analyze([s1, s2])
    assert abs(result.pooled_ec50 - 5.0) / 5.0 < 0.50


def test_ec50_cv_nonnegative():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    assert result.ec50_cv_pct >= 0.0


def test_forest_plot_data_length():
    s1 = _study("S1")
    s2 = _study("S2")
    s3 = _study("S3")
    result = meta_analyze([s1, s2, s3])
    assert len(result.forest_plot_data) == result.n_studies


def test_forest_plot_ci_low_lt_ec50():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    for entry in result.forest_plot_data:
        assert entry["ci_low"] < entry["ec50"]


def test_heterogeneity_i2_in_0_100():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    assert 0.0 <= result.heterogeneity_i2 <= 100.0


def test_recommendation_valid():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    valid = {"consistent", "moderate_heterogeneity", "high_heterogeneity"}
    assert result.recommendation in valid


def test_identical_studies_consistent():
    s1 = _study("S1", emax=100.0, ec50=5.0)
    s2 = _study("S2", emax=100.0, ec50=5.0)
    s3 = _study("S3", emax=100.0, ec50=5.0)
    result = meta_analyze([s1, s2, s3])
    assert result.recommendation == "consistent"


def test_notes_nonempty():
    s1 = _study("S1")
    s2 = _study("S2")
    result = meta_analyze([s1, s2])
    assert len(result.notes) > 0


def test_single_study_raises():
    s1 = _study("S1")
    with pytest.raises(ValueError):
        meta_analyze([s1])


def test_high_variability_recommendation():
    # Studies with very different EC50s -> high heterogeneity
    s1 = _study("S1", ec50=1.0)
    s2 = _study("S2", ec50=100.0)
    result = meta_analyze([s1, s2])
    assert result.recommendation == "high_heterogeneity"
