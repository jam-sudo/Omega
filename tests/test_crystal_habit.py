"""Tests for Phase 339 — Drug Crystal Habit Effects."""

import pytest

from omega_pbpk.biopharmaceutics.crystal_habit import (
    calculate_crystal_habit_dissolution,
    compare_crystal_habits,
)

# ---------------------------------------------------------------------------
# Helper: convenience wrapper with small dose for faster simulation
# ---------------------------------------------------------------------------


def _calc(habit: str, particle_size_um: float = 10.0, dose_mg: float = 100.0, **kw):
    return calculate_crystal_habit_dissolution(
        drug_name="TestDrug",
        habit=habit,
        particle_size_um=particle_size_um,
        dose_mg=dose_mg,
        **kw,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_habit_raises(self):
        with pytest.raises(ValueError, match="habit"):
            _calc(habit="rod")

    def test_particle_size_zero_raises(self):
        with pytest.raises(ValueError, match="particle_size_um"):
            _calc(habit="cube", particle_size_um=0.0)

    def test_particle_size_negative_raises(self):
        with pytest.raises(ValueError, match="particle_size_um"):
            _calc(habit="cube", particle_size_um=-5.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _calc(habit="sphere", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _calc(habit="sphere", dose_mg=-50.0)

    def test_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            _calc(habit="needle", solubility_mg_mL=0.0)

    def test_solubility_negative_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            _calc(habit="needle", solubility_mg_mL=-1.0)


# ---------------------------------------------------------------------------
# SSA ordering
# ---------------------------------------------------------------------------


class TestSSAOrdering:
    def test_needle_has_highest_ssa(self):
        needle = _calc("needle")
        plate = _calc("plate")
        cube = _calc("cube")
        sphere = _calc("sphere")
        assert needle.specific_surface_area_cm2_per_g > plate.specific_surface_area_cm2_per_g
        assert needle.specific_surface_area_cm2_per_g > cube.specific_surface_area_cm2_per_g
        assert needle.specific_surface_area_cm2_per_g > sphere.specific_surface_area_cm2_per_g

    def test_plate_higher_ssa_than_cube(self):
        plate = _calc("plate")
        cube = _calc("cube")
        assert plate.specific_surface_area_cm2_per_g > cube.specific_surface_area_cm2_per_g

    def test_sphere_higher_ssa_than_cube(self):
        sphere = _calc("sphere")
        cube = _calc("cube")
        assert sphere.specific_surface_area_cm2_per_g > cube.specific_surface_area_cm2_per_g

    def test_ssa_positive(self):
        for habit in ("needle", "plate", "cube", "sphere"):
            r = _calc(habit)
            assert r.specific_surface_area_cm2_per_g > 0, habit


# ---------------------------------------------------------------------------
# Dissolution ordering (faster dissolution = lower t90)
# ---------------------------------------------------------------------------


class TestDissolutionOrdering:
    def test_needle_fastest_dissolution(self):
        needle = _calc("needle", solubility_mg_mL=2.0)
        plate = _calc("plate", solubility_mg_mL=2.0)
        cube = _calc("cube", solubility_mg_mL=2.0)
        # needle should dissolve faster (lower t90)
        assert needle.t90_dissolution_h <= plate.t90_dissolution_h
        assert needle.t90_dissolution_h <= cube.t90_dissolution_h

    def test_cube_slowest_among_non_sphere(self):
        plate = _calc("plate")
        cube = _calc("cube")
        assert cube.t90_dissolution_h >= plate.t90_dissolution_h

    def test_t90_positive(self):
        for habit in ("needle", "plate", "cube", "sphere"):
            r = _calc(habit)
            assert r.t90_dissolution_h > 0, habit


# ---------------------------------------------------------------------------
# Fraction dissolved monotonicity and bounds
# ---------------------------------------------------------------------------


class TestFractionDissolved:
    @pytest.mark.parametrize("habit", ["needle", "plate", "cube", "sphere"])
    def test_f4h_ge_f2h(self, habit):
        r = _calc(habit, solubility_mg_mL=0.5)
        assert r.f_dissolved_4h >= r.f_dissolved_2h - 1e-9, habit

    @pytest.mark.parametrize("habit", ["needle", "plate", "cube", "sphere"])
    def test_f_dissolved_in_bounds(self, habit):
        r = _calc(habit)
        assert 0.0 <= r.f_dissolved_2h <= 1.0, f"2h {habit}"
        assert 0.0 <= r.f_dissolved_4h <= 1.0, f"4h {habit}"

    def test_high_solubility_fast_dissolution(self):
        r = _calc("needle", solubility_mg_mL=10.0, dose_mg=50.0)
        assert r.f_dissolved_4h > 0.5

    def test_low_solubility_slow_dissolution(self):
        # Very low solubility: drug should dissolve slowly
        r_low = _calc("cube", solubility_mg_mL=0.01)
        r_high = _calc("cube", solubility_mg_mL=5.0)
        assert r_low.f_dissolved_4h < r_high.f_dissolved_4h


# ---------------------------------------------------------------------------
# Dissolution rate
# ---------------------------------------------------------------------------


class TestDissolutionRate:
    @pytest.mark.parametrize("habit", ["needle", "plate", "cube", "sphere"])
    def test_dissolution_rate_positive(self, habit):
        r = _calc(habit)
        assert r.dissolution_rate_mg_per_h > 0, habit

    def test_needle_faster_initial_rate_than_cube(self):
        needle = _calc("needle")
        cube = _calc("cube")
        assert needle.dissolution_rate_mg_per_h > cube.dissolution_rate_mg_per_h


# ---------------------------------------------------------------------------
# Bioavailability factors
# ---------------------------------------------------------------------------


class TestBioavailabilityFactors:
    def test_needle_highest_ba_factor(self):
        results = {h: _calc(h) for h in ("needle", "plate", "cube", "sphere")}
        assert results["needle"].bioavailability_factor >= results["plate"].bioavailability_factor
        assert results["needle"].bioavailability_factor >= results["cube"].bioavailability_factor

    def test_cube_ba_factor_is_one(self):
        r = _calc("cube")
        assert r.bioavailability_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    @pytest.mark.parametrize("habit", ["needle", "plate", "cube", "sphere"])
    def test_notes_contain_drug_name(self, habit):
        r = calculate_crystal_habit_dissolution(
            drug_name="Ibuprofen", habit=habit, particle_size_um=20.0, dose_mg=400.0
        )
        assert "Ibuprofen" in r.notes

    def test_notes_nonempty(self):
        r = _calc("plate")
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_frozen_dataclass(self):
        r = _calc("sphere")
        with pytest.raises((AttributeError, TypeError)):
            r.habit = "cube"  # type: ignore[misc]

    def test_result_fields_accessible(self):
        r = _calc("needle")
        assert hasattr(r, "drug_name")
        assert hasattr(r, "habit")
        assert hasattr(r, "t90_dissolution_h")
        assert hasattr(r, "bioavailability_factor")


# ---------------------------------------------------------------------------
# compare_crystal_habits
# ---------------------------------------------------------------------------


class TestCompareHabits:
    def test_returns_all_four_habits(self):
        result = compare_crystal_habits("Drug", 10.0, 100.0)
        for h in ("needle", "plate", "cube", "sphere"):
            assert h in result

    def test_fastest_dissolving_key_present(self):
        result = compare_crystal_habits("Drug", 10.0, 100.0)
        assert "fastest_dissolving" in result

    def test_fastest_dissolving_is_valid_habit(self):
        result = compare_crystal_habits("Drug", 10.0, 100.0)
        assert result["fastest_dissolving"] in {"needle", "plate", "cube", "sphere"}

    def test_fastest_dissolving_has_minimum_t90(self):
        result = compare_crystal_habits("Drug", 10.0, 100.0, solubility_mg_mL=2.0)
        fastest = result["fastest_dissolving"]
        min_t90 = min(result[h].t90_dissolution_h for h in ("needle", "plate", "cube", "sphere"))
        assert result[fastest].t90_dissolution_h == pytest.approx(min_t90, rel=1e-6)

    def test_needle_typically_fastest(self):
        # With high solubility, needle and plate both dissolve nearly instantly;
        # "fastest_dissolving" should be needle or plate (highest SSA habits).
        result = compare_crystal_habits("Drug", 10.0, 100.0, solubility_mg_mL=1.0)
        assert result["fastest_dissolving"] in {"needle", "plate"}
