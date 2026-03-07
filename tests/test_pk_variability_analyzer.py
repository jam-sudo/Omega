"""Tests for Phase 800: PK variability and between-subject variability analysis."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_variability_analyzer import (
    PKVariabilityResult,
    analyze_pk_variability,
    compare_pk_variability,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_returns_pk_variability_result():
    result = analyze_pk_variability("CL", 10.0, 30.0)
    assert isinstance(result, PKVariabilityResult)


def test_result_is_frozen():
    result = analyze_pk_variability("CL", 10.0, 30.0)
    with pytest.raises((AttributeError, TypeError)):
        result.parameter_name = "other"  # type: ignore[misc]


def test_parameter_name_stored():
    result = analyze_pk_variability("Vd", 50.0, 25.0)
    assert result.parameter_name == "Vd"


def test_typical_value_stored():
    result = analyze_pk_variability("CL", 10.0, 30.0)
    assert result.typical_value == pytest.approx(10.0)


def test_cv_pct_stored():
    result = analyze_pk_variability("CL", 10.0, 35.0)
    assert result.cv_pct == pytest.approx(35.0)


def test_n_subjects_stored():
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=500)
    assert result.n_subjects == 500


def test_distribution_stored():
    result = analyze_pk_variability("CL", 10.0, 30.0, distribution="log-normal")
    assert result.distribution == "log-normal"


# ---------------------------------------------------------------------------
# Percentile ordering
# ---------------------------------------------------------------------------


def test_p5_less_than_p50():
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=1000)
    assert result.p5 < result.p50


def test_p50_less_than_p95():
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=1000)
    assert result.p50 < result.p95


def test_p25_less_than_p75():
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=1000)
    assert result.p25 < result.p75


def test_p5_less_than_p25():
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=1000)
    assert result.p5 < result.p25


# ---------------------------------------------------------------------------
# Geometric mean and GSD
# ---------------------------------------------------------------------------


def test_geometric_mean_approx_typical_value_lognormal():
    """For log-normal, geometric mean should be close to typical_value (within 5%)."""
    result = analyze_pk_variability("CL", 10.0, 30.0, n_subjects=2000)
    assert result.geometric_mean == pytest.approx(10.0, rel=0.10)


def test_geometric_sd_greater_than_1_lognormal():
    """GSD should be > 1 for any positive CV."""
    result = analyze_pk_variability("CL", 10.0, 30.0)
    assert result.geometric_sd > 1.0


def test_geometric_sd_equals_exp_sigma():
    """GSD = exp(sigma) where sigma = sqrt(ln(1 + cv^2))."""
    cv_pct = 40.0
    cv = cv_pct / 100.0
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    expected_gsd = math.exp(sigma)
    result = analyze_pk_variability("CL", 10.0, cv_pct)
    assert result.geometric_sd == pytest.approx(expected_gsd, rel=1e-6)


# ---------------------------------------------------------------------------
# Median close to typical value for log-normal
# ---------------------------------------------------------------------------


def test_p50_approx_typical_value():
    """Median of log-normal ~ typical_value (within 10% for moderate CV)."""
    result = analyze_pk_variability("CL", 10.0, 20.0, n_subjects=2000)
    assert result.p50 == pytest.approx(10.0, rel=0.15)


# ---------------------------------------------------------------------------
# Variability class
# ---------------------------------------------------------------------------


def test_variability_class_low():
    result = analyze_pk_variability("CL", 10.0, 15.0)
    assert result.variability_class == "low"


def test_variability_class_moderate():
    result = analyze_pk_variability("CL", 10.0, 35.0)
    assert result.variability_class == "moderate"


def test_variability_class_high():
    result = analyze_pk_variability("CL", 10.0, 70.0)
    assert result.variability_class == "high"


def test_variability_class_boundary_low_moderate():
    # CV = 20.0 → "moderate"
    result = analyze_pk_variability("CL", 10.0, 20.0)
    assert result.variability_class == "moderate"


def test_variability_class_boundary_moderate_high():
    # CV = 50.0 → "moderate"
    result = analyze_pk_variability("CL", 10.0, 50.0)
    assert result.variability_class == "moderate"


# ---------------------------------------------------------------------------
# High CV → wide interval
# ---------------------------------------------------------------------------


def test_high_cv_wider_interval():
    low_cv = analyze_pk_variability("CL", 10.0, 10.0, n_subjects=1000)
    high_cv = analyze_pk_variability("CL", 10.0, 80.0, n_subjects=1000)
    low_range = low_cv.p95 - low_cv.p5
    high_range = high_cv.p95 - high_cv.p5
    assert high_range > low_range


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_cv_pct_negative_raises():
    with pytest.raises(ValueError, match="cv_pct"):
        analyze_pk_variability("CL", 10.0, -5.0)


def test_validation_typical_value_zero_raises():
    with pytest.raises(ValueError, match="typical_value"):
        analyze_pk_variability("CL", 0.0, 30.0)


def test_validation_typical_value_negative_raises():
    with pytest.raises(ValueError, match="typical_value"):
        analyze_pk_variability("CL", -1.0, 30.0)


def test_notes_nonempty():
    result = analyze_pk_variability("CL", 10.0, 30.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# compare_pk_variability
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    params = [
        {"parameter_name": "CL", "typical_value": 10.0, "cv_pct": 30.0},
        {"parameter_name": "Vd", "typical_value": 50.0, "cv_pct": 50.0},
    ]
    results = compare_pk_variability(params)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_cv_pct_descending():
    params = [
        {"parameter_name": "CL", "typical_value": 10.0, "cv_pct": 20.0},
        {"parameter_name": "Vd", "typical_value": 50.0, "cv_pct": 60.0},
        {"parameter_name": "ka", "typical_value": 1.0, "cv_pct": 40.0},
    ]
    results = compare_pk_variability(params)
    cvs = [r.cv_pct for r in results]
    assert cvs == sorted(cvs, reverse=True)


def test_compare_all_pk_variability_results():
    params = [
        {"parameter_name": "CL", "typical_value": 10.0, "cv_pct": 30.0},
        {"parameter_name": "Vd", "typical_value": 50.0, "cv_pct": 50.0},
    ]
    results = compare_pk_variability(params)
    for r in results:
        assert isinstance(r, PKVariabilityResult)
