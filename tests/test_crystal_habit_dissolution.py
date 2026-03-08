"""Tests for crystal habit dissolution prediction — Phase 422."""

import pytest

from omega_pbpk.prediction.crystal_habit_dissolution import (
    CrystalHabitResult,
    compare_crystal_habits,
    predict_crystal_habit_dissolution,
)

# --- Helper ---


def make_result(habit="sphere", diameter=50.0, aspect_ratio=1.0, **kwargs) -> CrystalHabitResult:
    params = dict(
        drug_name="TestDrug",
        crystal_habit=habit,
        particle_diameter_um=diameter,
        aspect_ratio=aspect_ratio,
    )
    params.update(kwargs)
    return predict_crystal_habit_dissolution(**params)


# --- Return type ---


def test_return_type():
    result = make_result()
    assert isinstance(result, CrystalHabitResult)


def test_drug_name_preserved():
    result = make_result(drug_name="Ibuprofen")
    assert result.drug_name == "Ibuprofen"


def test_habit_preserved():
    result = make_result(habit="needle", aspect_ratio=3.0)
    assert result.crystal_habit == "needle"


# --- Surface area factors ---


def test_sphere_sa_factor_is_one():
    result = make_result(habit="sphere")
    assert result.surface_area_factor == 1.0


def test_all_habits_sa_factor_at_least_one():
    for habit in ("sphere", "cube", "plate", "needle", "irregular"):
        aspect = 3.0 if habit in ("needle", "plate") else 1.0
        result = make_result(habit=habit, aspect_ratio=aspect)
        assert result.surface_area_factor >= 1.0, f"Failed for habit={habit}"


def test_needle_greater_than_plate_greater_than_cube_greater_than_sphere():
    """With aspect_ratio=3.0, needle > plate > cube > sphere."""
    sphere = make_result(habit="sphere")
    cube = make_result(habit="cube")
    plate = make_result(habit="plate", aspect_ratio=3.0)
    needle = make_result(habit="needle", aspect_ratio=3.0)
    assert needle.surface_area_factor > plate.surface_area_factor
    assert plate.surface_area_factor > cube.surface_area_factor
    assert cube.surface_area_factor > sphere.surface_area_factor


# --- Dissolution metrics ---


def test_dissolution_rate_relative_equals_sa_factor():
    for habit in ("sphere", "cube", "irregular"):
        result = make_result(habit=habit)
        assert abs(result.dissolution_rate_relative - result.surface_area_factor) < 1e-9


def test_t50_relative_is_inverse_of_sa_factor():
    for habit in ("sphere", "cube", "irregular"):
        result = make_result(habit=habit)
        expected = 1.0 / result.surface_area_factor
        assert abs(result.t50_relative - expected) < 1e-9


# --- Effective surface area ---


def test_effective_surface_area_positive():
    for habit in ("sphere", "cube", "plate", "needle", "irregular"):
        aspect = 3.0 if habit in ("needle", "plate") else 1.0
        result = make_result(habit=habit, aspect_ratio=aspect)
        assert result.effective_surface_area_cm2_per_g > 0.0


def test_effective_surface_area_sphere_formula():
    diameter_um = 50.0
    density = 1.2
    d_cm = diameter_um * 1e-4
    expected = (6.0 / d_cm) / density  # SA/V * 1/density
    result = make_result(habit="sphere", diameter=diameter_um, drug_density_g_cm3=density)
    assert abs(result.effective_surface_area_cm2_per_g - expected) < 0.1


# --- Aspect ratio effect ---


def test_larger_aspect_ratio_larger_sa_factor_for_needles():
    r1 = make_result(habit="needle", aspect_ratio=1.0)
    r2 = make_result(habit="needle", aspect_ratio=5.0)
    assert r2.surface_area_factor > r1.surface_area_factor


def test_larger_aspect_ratio_larger_sa_factor_for_plates():
    r1 = make_result(habit="plate", aspect_ratio=1.0)
    r2 = make_result(habit="plate", aspect_ratio=5.0)
    assert r2.surface_area_factor > r1.surface_area_factor


# --- Process sensitivity ---


def test_needle_process_sensitivity_high():
    result = make_result(habit="needle", aspect_ratio=2.0)
    assert result.process_sensitivity == "high"


def test_sphere_process_sensitivity_low():
    result = make_result(habit="sphere")
    assert result.process_sensitivity == "low"


# --- compare_crystal_habits ---


def test_compare_returns_five_results():
    results = compare_crystal_habits("Drug", 50.0, 0.1)
    assert len(results) == 5


def test_compare_sorted_descending():
    results = compare_crystal_habits("Drug", 50.0, 0.1)
    rates = [r.dissolution_rate_relative for r in results]
    assert rates == sorted(rates, reverse=True)


def test_compare_all_valid_types():
    results = compare_crystal_habits("Drug", 50.0, 0.1)
    for r in results:
        assert isinstance(r, CrystalHabitResult)


# --- Validation errors ---


def test_invalid_habit_raises():
    with pytest.raises(ValueError, match="crystal_habit"):
        predict_crystal_habit_dissolution("Drug", "rhombus", 50.0)


def test_invalid_particle_diameter_zero():
    with pytest.raises(ValueError, match="particle_diameter_um"):
        predict_crystal_habit_dissolution("Drug", "sphere", 0.0)


def test_invalid_particle_diameter_negative():
    with pytest.raises(ValueError, match="particle_diameter_um"):
        predict_crystal_habit_dissolution("Drug", "sphere", -10.0)


def test_invalid_aspect_ratio_below_one():
    with pytest.raises(ValueError, match="aspect_ratio"):
        predict_crystal_habit_dissolution("Drug", "needle", 50.0, aspect_ratio=0.5)
