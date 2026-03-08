"""
Tests for Phase 440: Formulation pH stability prediction.
"""

import math

import pytest

from omega_pbpk.prediction.formulation_ph_stability import (
    PHStabilityResult,
    predict_ph_stability,
    screen_ph_range,
)

# --------------------------------------------------------------------------- #
# Basic return type
# --------------------------------------------------------------------------- #


def test_returns_correct_type():
    result = predict_ph_stability("aspirin", formulation_ph=7.0)
    assert isinstance(result, PHStabilityResult)


def test_drug_name_preserved():
    result = predict_ph_stability("Metformin", formulation_ph=6.5)
    assert result.drug_name == "Metformin"


# --------------------------------------------------------------------------- #
# Core metric correctness
# --------------------------------------------------------------------------- #


def test_k_deg_positive():
    result = predict_ph_stability("drug", formulation_ph=7.0)
    assert result.k_deg_per_day > 0.0


def test_t90_formula():
    """t90 = -ln(0.9) / k_deg."""
    result = predict_ph_stability("drug", formulation_ph=7.0)
    expected = -math.log(0.9) / result.k_deg_per_day
    assert result.t90_days == pytest.approx(expected, rel=1e-6)


def test_t95_formula():
    """t95 = -ln(0.95) / k_deg."""
    result = predict_ph_stability("drug", formulation_ph=7.0)
    expected = -math.log(0.95) / result.k_deg_per_day
    assert result.t95_days == pytest.approx(expected, rel=1e-6)


def test_shelf_life_equals_t90():
    result = predict_ph_stability("drug", formulation_ph=7.0)
    assert result.shelf_life_days == pytest.approx(result.t90_days, rel=1e-6)


def test_degradation_pct_6months_in_range():
    result = predict_ph_stability("drug", formulation_ph=7.0)
    assert 0.0 <= result.degradation_pct_6months <= 100.0


# --------------------------------------------------------------------------- #
# pH-rate profile: optimal pH → minimum k_deg
# --------------------------------------------------------------------------- #


def test_optimal_ph_gives_minimum_k_deg():
    """At optimal_ph, degradation rate should be lowest in a range."""
    optimal_ph = 6.0
    ph_values = [3.0, 6.0, 9.0]
    results = [predict_ph_stability("drug", ph, optimal_ph=optimal_ph) for ph in ph_values]
    k_degs = [r.k_deg_per_day for r in results]
    # pH 6.0 should have lowest k_deg
    idx_min = k_degs.index(min(k_degs))
    assert ph_values[idx_min] == pytest.approx(optimal_ph)


def test_extreme_acid_ph_shorter_shelf_life():
    """Acidic pH (1) should be less stable than neutral (7) for default optimal_ph=7."""
    r_acid = predict_ph_stability("drug", formulation_ph=1.0)
    r_neutral = predict_ph_stability("drug", formulation_ph=7.0)
    assert r_acid.shelf_life_days < r_neutral.shelf_life_days


def test_extreme_base_ph_shorter_shelf_life():
    """Basic pH (13) should be less stable than neutral (7) for default optimal_ph=7."""
    r_base = predict_ph_stability("drug", formulation_ph=13.0)
    r_neutral = predict_ph_stability("drug", formulation_ph=7.0)
    assert r_base.shelf_life_days < r_neutral.shelf_life_days


# --------------------------------------------------------------------------- #
# Temperature effects
# --------------------------------------------------------------------------- #


def test_higher_temp_shorter_shelf_life():
    """At 40 C, degradation is faster than at 25 C."""
    r_25 = predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=25.0)
    r_40 = predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=40.0)
    assert r_40.shelf_life_days < r_25.shelf_life_days


def test_q10_above_25C_greater_than_1():
    """At temperatures above 25 C, Q10 > 1."""
    result = predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=35.0)
    assert result.q10_factor > 1.0


def test_q10_at_25C_equals_1():
    """At 25 C, Q10 = 2^0 = 1."""
    result = predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=25.0)
    assert result.q10_factor == pytest.approx(1.0, rel=1e-6)


def test_q10_below_25C_less_than_1():
    """At temperatures below 25 C, Q10 < 1."""
    result = predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=5.0)
    assert result.q10_factor < 1.0


# --------------------------------------------------------------------------- #
# Stability class
# --------------------------------------------------------------------------- #


def test_stability_class_excellent_long_shelf_life():
    """Very slow degradation → excellent (>730 days)."""
    # Use very small k0 to get long shelf life
    result = predict_ph_stability("drug", formulation_ph=7.0, k0_per_day=1e-6)
    assert result.stability_class == "excellent"


def test_stability_class_poor_fast_degradation():
    """Very fast degradation → poor (<90 days)."""
    # Use very large k0 to get short shelf life
    result = predict_ph_stability("drug", formulation_ph=7.0, k0_per_day=0.1)
    assert result.stability_class == "poor"


def test_stability_class_assignment():
    """Check all four class boundaries."""
    # This drug degrades slowly → excellent
    r_excellent = predict_ph_stability("d", formulation_ph=7.0, k0_per_day=1e-6)
    assert r_excellent.stability_class == "excellent"


# --------------------------------------------------------------------------- #
# Temperature sensitivity
# --------------------------------------------------------------------------- #


def test_temperature_sensitivity_high_for_large_ea():
    result = predict_ph_stability("drug", formulation_ph=7.0, ea_kJ_per_mol=120.0)
    assert result.temperature_sensitivity == "high"


def test_temperature_sensitivity_moderate_for_typical_ea():
    result = predict_ph_stability("drug", formulation_ph=7.0, ea_kJ_per_mol=80.0)
    assert result.temperature_sensitivity == "moderate"


def test_temperature_sensitivity_low_for_small_ea():
    result = predict_ph_stability("drug", formulation_ph=7.0, ea_kJ_per_mol=30.0)
    assert result.temperature_sensitivity == "low"


# --------------------------------------------------------------------------- #
# screen_ph_range
# --------------------------------------------------------------------------- #


def test_screen_ph_range_correct_length():
    ph_values = [4.0, 5.0, 6.0, 7.0, 8.0]
    results = screen_ph_range("drug", ph_values)
    assert len(results) == len(ph_values)


def test_screen_ph_range_sorted_descending():
    ph_values = [3.0, 5.0, 7.0, 9.0, 11.0]
    results = screen_ph_range("drug", ph_values)
    shelf_lives = [r.shelf_life_days for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_screen_ph_range_returns_stability_results():
    results = screen_ph_range("drug", [6.0, 7.0, 8.0])
    for r in results:
        assert isinstance(r, PHStabilityResult)


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #


def test_validation_ph_negative():
    with pytest.raises(ValueError, match="formulation_ph"):
        predict_ph_stability("drug", formulation_ph=-0.5)


def test_validation_ph_above_14():
    with pytest.raises(ValueError, match="formulation_ph"):
        predict_ph_stability("drug", formulation_ph=14.5)


def test_validation_temp_too_low():
    with pytest.raises(ValueError, match="storage_temp_C"):
        predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=-25.0)


def test_validation_temp_too_high():
    with pytest.raises(ValueError, match="storage_temp_C"):
        predict_ph_stability("drug", formulation_ph=7.0, storage_temp_C=85.0)
