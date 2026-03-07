"""Tests for analysis/dose_response_variability.py (Phase 664)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.dose_response_variability import (
    DoseResponseVariabilityResult,
    _hill_response,
    analyze_dose_response_variability,
    simulate_population_dose_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def typical_ec50_values() -> list[float]:
    """Moderate variability EC50 population."""
    return [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]


@pytest.fixture()
def default_result(typical_ec50_values) -> DoseResponseVariabilityResult:
    return analyze_dose_response_variability(
        ec50_values=typical_ec50_values,
        standard_dose_conc_mg_L=1.0,
    )


@pytest.fixture()
def high_cv_ec50_values() -> list[float]:
    """High variability EC50 population (>60% CV)."""
    return [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 50.0, 100.0, 200.0]


@pytest.fixture()
def low_cv_ec50_values() -> list[float]:
    """Low variability EC50 population (<30% CV)."""
    return [0.95, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06]


# ---------------------------------------------------------------------------
# Type and basic presence
# ---------------------------------------------------------------------------


def test_returns_dataclass_instance(default_result):
    assert isinstance(default_result, DoseResponseVariabilityResult)


def test_ec50_mean_positive(default_result):
    assert default_result.ec50_mean > 0


def test_ec50_sd_nonnegative(default_result):
    assert default_result.ec50_sd >= 0


def test_ec50_cv_pct_nonnegative(default_result):
    assert default_result.ec50_cv_pct >= 0


def test_ec50_p5_lte_mean_lte_p95(default_result):
    assert default_result.ec50_p5 <= default_result.ec50_mean
    assert default_result.ec50_mean <= default_result.ec50_p95


def test_hill_n_mean_positive(default_result):
    assert default_result.hill_n_mean > 0


def test_emax_mean_positive(default_result):
    assert default_result.emax_mean > 0


def test_responder_fraction_in_range(default_result):
    assert 0.0 <= default_result.responder_fraction <= 1.0


def test_n_responders_lte_n_subjects(default_result):
    assert default_result.n_responders <= default_result.n_subjects


def test_pd_variability_class_valid(default_result):
    assert default_result.pd_variability_class in {"low", "moderate", "high"}


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


# ---------------------------------------------------------------------------
# Variability class thresholds
# ---------------------------------------------------------------------------


def test_high_cv_classified_as_high(high_cv_ec50_values):
    result = analyze_dose_response_variability(ec50_values=high_cv_ec50_values)
    assert result.pd_variability_class == "high"


def test_low_cv_classified_as_low(low_cv_ec50_values):
    result = analyze_dose_response_variability(ec50_values=low_cv_ec50_values)
    assert result.pd_variability_class == "low"


# ---------------------------------------------------------------------------
# simulate_population_dose_response
# ---------------------------------------------------------------------------


def test_simulate_returns_dict():
    result = simulate_population_dose_response(n_subjects=20, seed=42)
    assert isinstance(result, dict)


def test_simulate_has_expected_keys():
    result = simulate_population_dose_response(n_subjects=20, seed=42)
    expected_keys = {
        "ec50_values",
        "emax_values",
        "mean_response_curve",
        "p5_response_curve",
        "p95_response_curve",
        "dose_concentrations",
        "analysis",
    }
    assert expected_keys.issubset(result.keys())


def test_simulate_mean_response_curve_length():
    concs = [0.1, 0.3, 1.0, 3.0, 10.0]
    result = simulate_population_dose_response(n_subjects=20, dose_concentrations=concs, seed=42)
    assert len(result["mean_response_curve"]) == len(concs)


def test_simulate_mean_response_in_emax_range():
    emax_mean = 80.0
    result = simulate_population_dose_response(
        n_subjects=50, emax_mean=emax_mean, emax_cv_pct=10.0, seed=42
    )
    # Allow some tolerance due to population variability
    for resp in result["mean_response_curve"]:
        assert 0.0 <= resp <= emax_mean * 2.0


def test_simulate_analysis_is_result_type():
    result = simulate_population_dose_response(n_subjects=20, seed=42)
    assert isinstance(result["analysis"], DoseResponseVariabilityResult)


def test_simulate_n_subjects_ec50_count():
    n = 30
    result = simulate_population_dose_response(n_subjects=n, seed=42)
    assert len(result["ec50_values"]) == n


# ---------------------------------------------------------------------------
# Hill response at EC50 gives ~50% Emax
# ---------------------------------------------------------------------------


def test_hill_response_at_ec50_is_half_emax():
    ec50 = 2.0
    emax = 100.0
    n = 1.5
    resp = _hill_response(ec50, ec50, emax, n)
    assert abs(resp - emax / 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_empty_ec50():
    with pytest.raises(ValueError):
        analyze_dose_response_variability(ec50_values=[])


def test_validation_single_ec50():
    with pytest.raises(ValueError):
        analyze_dose_response_variability(ec50_values=[1.0])


def test_validation_ec50_contains_zero():
    with pytest.raises(ValueError):
        analyze_dose_response_variability(ec50_values=[1.0, 0.0, 2.0])


def test_validation_n_subjects_one():
    with pytest.raises(ValueError):
        simulate_population_dose_response(n_subjects=1)


def test_validation_negative_cv():
    with pytest.raises(ValueError):
        simulate_population_dose_response(n_subjects=10, ec50_cv_pct=-1.0)
