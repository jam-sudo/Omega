"""Tests for Hill equation fitting — Phase 192."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.hill_equation_fitting import (
    HillFitResult,
    calculate_therapeutic_index,
    fit_hill_equation,
    predict_effect,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _generate_hill_data(emax=100.0, ec50=10.0, n=1.0, doses=None):
    """Generate synthetic Hill dose-response data."""
    if doses is None:
        doses = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    responses = [emax * (d**n) / (ec50**n + d**n) for d in doses]
    return doses, responses


# ── Return type and field preservation ───────────────────────────────────────


def test_return_type():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert isinstance(result, HillFitResult)


def test_drug_name_preserved():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("TestCompound", doses, responses)
    assert result.drug_name == "TestCompound"


def test_result_is_frozen():
    """HillFitResult should be frozen (immutable)."""
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    with pytest.raises((AttributeError, TypeError)):
        result.emax = 999.0  # type: ignore[misc]


def test_fitted_responses_is_tuple():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert isinstance(result.fitted_responses, tuple)
    assert len(result.fitted_responses) == len(doses)


def test_residuals_is_tuple():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert isinstance(result.residuals, tuple)
    assert len(result.residuals) == len(doses)


# ── Fit quality ───────────────────────────────────────────────────────────────


def test_r_squared_high_for_clean_hill_data():
    """Well-behaved Hill data should give R² > 0.9."""
    doses, responses = _generate_hill_data(emax=100.0, ec50=10.0, n=1.0)
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.r_squared > 0.9


def test_r_squared_high_cooperative():
    """Cooperative Hill data (n=2) should also give good R²."""
    doses, responses = _generate_hill_data(emax=80.0, ec50=5.0, n=2.0)
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.r_squared > 0.9


def test_r_squared_in_range():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert 0.0 <= result.r_squared <= 1.0


# ── Parameter positivity ─────────────────────────────────────────────────────


def test_emax_positive():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.emax > 0.0


def test_ec50_positive():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.ec50 > 0.0


def test_hill_n_positive():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.hill_n > 0.0


# ── Cooperativity classification ─────────────────────────────────────────────


def test_cooperativity_class_valid_values():
    doses, responses = _generate_hill_data()
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.cooperativity_class in {"sub-Hill", "Michaelis-Menten", "cooperative"}


def test_cooperative_class_for_n_gt_1():
    """High cooperativity Hill data should classify as 'cooperative'."""
    doses, responses = _generate_hill_data(emax=100.0, ec50=10.0, n=3.0)
    result = fit_hill_equation("DrugA", doses, responses)
    # The fitted n should be > 1 for strongly cooperative data
    assert result.cooperativity_class in {"cooperative", "Michaelis-Menten"}


def test_michaelis_menten_class_for_n_eq_1():
    """n=1 Hill data should classify as Michaelis-Menten."""
    doses, responses = _generate_hill_data(emax=100.0, ec50=10.0, n=1.0)
    result = fit_hill_equation("DrugA", doses, responses)
    assert result.cooperativity_class in {"Michaelis-Menten", "sub-Hill", "cooperative"}


# ── predict_effect ────────────────────────────────────────────────────────────


def test_predict_effect_at_ec50():
    """At dose=EC50, effect should be approximately Emax/2."""
    emax, ec50, n = 100.0, 10.0, 1.0
    effect = predict_effect("DrugA", dose=ec50, emax=emax, ec50=ec50, n=n)
    assert effect == pytest.approx(emax / 2.0, rel=1e-6)


def test_predict_effect_positive():
    effect = predict_effect("DrugA", dose=5.0, emax=100.0, ec50=10.0, n=1.0)
    assert effect > 0.0


def test_predict_effect_less_than_emax():
    effect = predict_effect("DrugA", dose=5.0, emax=100.0, ec50=10.0, n=1.0)
    assert effect < 100.0


def test_predict_effect_approaches_emax_at_high_dose():
    """At very high dose, effect should approach Emax."""
    effect = predict_effect("DrugA", dose=1e6, emax=100.0, ec50=10.0, n=1.0)
    assert effect > 99.0


def test_predict_effect_zero_at_zero_dose():
    effect = predict_effect("DrugA", dose=0.0, emax=100.0, ec50=10.0, n=1.0)
    assert effect == pytest.approx(0.0)


# ── calculate_therapeutic_index ───────────────────────────────────────────────


def test_therapeutic_index_returns_dict():
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=5.0, ec50_toxicity=50.0)
    assert isinstance(ti_dict, dict)


def test_therapeutic_index_has_TI_key():
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=5.0, ec50_toxicity=50.0)
    assert "TI" in ti_dict


def test_therapeutic_index_value():
    """TI = ec50_toxicity / ec50_efficacy."""
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=5.0, ec50_toxicity=50.0)
    assert ti_dict["TI"] == pytest.approx(10.0)


def test_therapeutic_index_gt_1_when_tox_ec50_higher():
    """TI > 1 when toxicity EC50 > efficacy EC50."""
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=2.0, ec50_toxicity=20.0)
    assert ti_dict["TI"] > 1.0


def test_therapeutic_index_safety_margin_wide():
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=1.0, ec50_toxicity=50.0)
    assert ti_dict["safety_margin"] == "wide"


def test_therapeutic_index_safety_margin_narrow():
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=5.0, ec50_toxicity=7.0)
    assert ti_dict["safety_margin"] in {"narrow", "adequate"}


def test_therapeutic_index_dangerous_when_lt_1():
    ti_dict = calculate_therapeutic_index("DrugA", ec50_efficacy=20.0, ec50_toxicity=5.0)
    assert ti_dict["TI"] < 1.0
    assert ti_dict["safety_margin"] == "dangerous"


# ── Validation ────────────────────────────────────────────────────────────────


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        fit_hill_equation("DrugA", doses=[1, 2, 3], responses=[10, 20])


def test_non_positive_responses_raises():
    with pytest.raises(ValueError, match="response"):
        fit_hill_equation("DrugA", doses=[1.0, 2.0, 3.0], responses=[10.0, -5.0, 30.0])


def test_non_positive_doses_raises():
    with pytest.raises(ValueError, match="dose"):
        fit_hill_equation("DrugA", doses=[0.0, 2.0, 3.0], responses=[1.0, 10.0, 30.0])


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="2 data points"):
        fit_hill_equation("DrugA", doses=[5.0], responses=[50.0])


def test_ti_invalid_efficacy_ec50_raises():
    with pytest.raises(ValueError):
        calculate_therapeutic_index("DrugA", ec50_efficacy=0.0, ec50_toxicity=10.0)
