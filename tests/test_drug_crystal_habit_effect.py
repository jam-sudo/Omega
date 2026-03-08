"""Tests for Phase 997 — drug_crystal_habit_effect.py"""

import pytest

from omega_pbpk.prediction.drug_crystal_habit_effect import (
    CrystalHabitEffectResult,
    compare_crystal_habits,
    predict_crystal_habit_effect,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_crystal_habit_effect_result():
    result = predict_crystal_habit_effect("TestDrug", "sphere")
    assert isinstance(result, CrystalHabitEffectResult)


def test_drug_name_preserved():
    result = predict_crystal_habit_effect("Ibuprofen", "needle")
    assert result.drug_name == "Ibuprofen"


def test_crystal_habit_preserved():
    for habit in ("sphere", "cube", "plate", "needle", "rod", "irregular"):
        r = predict_crystal_habit_effect("Drug", habit)
        assert r.crystal_habit == habit


# ---------------------------------------------------------------------------
# Numeric sanity
# ---------------------------------------------------------------------------


def test_dissolution_rate_factor_positive():
    result = predict_crystal_habit_effect("Drug", "sphere")
    assert result.dissolution_rate_factor > 0


def test_effective_dissolution_rate_positive():
    result = predict_crystal_habit_effect("Drug", "cube")
    assert result.effective_dissolution_rate_mg_min > 0


def test_estimated_tmax_positive():
    result = predict_crystal_habit_effect("Drug", "plate")
    assert result.estimated_tmax_min > 0


def test_surface_area_factor_positive():
    result = predict_crystal_habit_effect("Drug", "rod")
    assert result.surface_area_factor > 0


def test_wettability_factor_in_range():
    for habit in ("sphere", "cube", "plate", "needle", "rod", "irregular"):
        r = predict_crystal_habit_effect("Drug", habit)
        assert 0 < r.wettability_factor <= 1.0


def test_intrinsic_solubility_positive():
    r = predict_crystal_habit_effect("Drug", "sphere", logp=1.5)
    assert r.intrinsic_solubility_mg_mL > 0


# ---------------------------------------------------------------------------
# Crystal habit comparisons
# ---------------------------------------------------------------------------


def test_needle_higher_surface_area_than_sphere():
    needle = predict_crystal_habit_effect("Drug", "needle")
    sphere = predict_crystal_habit_effect("Drug", "sphere")
    assert needle.surface_area_factor > sphere.surface_area_factor


def test_needle_lower_wettability_than_sphere():
    needle = predict_crystal_habit_effect("Drug", "needle")
    sphere = predict_crystal_habit_effect("Drug", "sphere")
    assert needle.wettability_factor < sphere.wettability_factor


def test_needle_dissolution_factor_less_than_sphere():
    """Poor wetting dominates — needle has lower dissolution_rate_factor than sphere."""
    needle = predict_crystal_habit_effect("Drug", "needle")
    sphere = predict_crystal_habit_effect("Drug", "sphere")
    assert needle.dissolution_rate_factor < sphere.dissolution_rate_factor


# ---------------------------------------------------------------------------
# compare_crystal_habits
# ---------------------------------------------------------------------------


def test_compare_returns_six_results():
    results = compare_crystal_habits("Drug")
    assert len(results) == 6


def test_compare_sorted_descending():
    results = compare_crystal_habits("Drug")
    factors = [r.dissolution_rate_factor for r in results]
    assert factors == sorted(factors, reverse=True)


def test_compare_contains_all_habits():
    results = compare_crystal_habits("Drug")
    habits = {r.crystal_habit for r in results}
    assert habits == {"sphere", "cube", "plate", "needle", "rod", "irregular"}


# ---------------------------------------------------------------------------
# absorption_enhancement
# ---------------------------------------------------------------------------


def test_absorption_enhancement_valid_values():
    allowed = {"none", "moderate", "significant"}
    for habit in ("sphere", "cube", "plate", "needle", "rod", "irregular"):
        r = predict_crystal_habit_effect("Drug", habit)
        assert r.absorption_enhancement in allowed


# ---------------------------------------------------------------------------
# Yalkowsky solubility check
# ---------------------------------------------------------------------------


def test_intrinsic_solubility_yalkowsky_formula():
    r = predict_crystal_habit_effect("Drug", "sphere", logp=2.0)
    expected = 10 ** (-0.7 * 2.0 + 0.44)
    assert abs(r.intrinsic_solubility_mg_mL - expected) < 1e-9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_crystal_habit_raises_value_error():
    with pytest.raises(ValueError, match="crystal_habit"):
        predict_crystal_habit_effect("Drug", "donut")


def test_invalid_mw_raises_value_error():
    with pytest.raises(ValueError, match="mw"):
        predict_crystal_habit_effect("Drug", "sphere", mw=0.0)


def test_negative_mw_raises_value_error():
    with pytest.raises(ValueError, match="mw"):
        predict_crystal_habit_effect("Drug", "sphere", mw=-10.0)


def test_invalid_particle_size_raises_value_error():
    with pytest.raises(ValueError, match="particle_size_um"):
        predict_crystal_habit_effect("Drug", "sphere", particle_size_um=0.0)


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_crystal_habit_effect("Drug", "sphere", dose_mg=-1.0)
