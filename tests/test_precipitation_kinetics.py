"""Tests for biopharmaceutics.precipitation_kinetics (Phase 321)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.precipitation_kinetics import (
    PrecipitationResult,
    compare_polymer_inhibition,
    crystal_growth_rate,
    nucleation_rate,
    simulate_precipitation,
)


# ---------------------------------------------------------------------------
# nucleation_rate
# ---------------------------------------------------------------------------

class TestNucleationRate:
    def test_no_supersaturation_returns_zero(self):
        assert nucleation_rate(1.0) == 0.0

    def test_undersaturated_returns_zero(self):
        assert nucleation_rate(0.5) == 0.0

    def test_supersaturated_returns_positive(self):
        J = nucleation_rate(10.0)
        assert J > 0.0

    def test_higher_supersaturation_higher_rate(self):
        J1 = nucleation_rate(2.0)
        J2 = nucleation_rate(5.0)
        J3 = nucleation_rate(20.0)
        assert J1 < J2 < J3

    def test_default_params_finite(self):
        J = nucleation_rate(3.0)
        assert math.isfinite(J)
        assert J >= 0.0

    def test_high_interfacial_tension_lowers_rate(self):
        J_low = nucleation_rate(5.0, interfacial_tension_mJ_per_m2=1.0)
        J_high = nucleation_rate(5.0, interfacial_tension_mJ_per_m2=20.0)
        # Higher interfacial tension => higher nucleation barrier => lower rate
        assert J_low >= J_high

    def test_temperature_effect(self):
        # Higher temperature should increase rate (lower barrier)
        J_cold = nucleation_rate(5.0, temperature_C=25.0)
        J_warm = nucleation_rate(5.0, temperature_C=37.0)
        # Both should be positive and finite
        assert math.isfinite(J_cold)
        assert math.isfinite(J_warm)


# ---------------------------------------------------------------------------
# crystal_growth_rate
# ---------------------------------------------------------------------------

class TestCrystalGrowthRate:
    def test_no_supersaturation_returns_zero(self):
        assert crystal_growth_rate(1.0) == 0.0

    def test_undersaturated_returns_zero(self):
        assert crystal_growth_rate(0.8) == 0.0

    def test_supersaturated_returns_positive(self):
        G = crystal_growth_rate(2.0)
        assert G > 0.0

    def test_parabolic_growth_law(self):
        # G = k_g * (S-1)^n_g with k_g=1e-8, n_g=2
        S = 3.0
        expected = 1e-8 * (S - 1.0) ** 2.0
        assert crystal_growth_rate(S) == pytest.approx(expected, rel=1e-6)

    def test_custom_parameters(self):
        G = crystal_growth_rate(2.0, k_g=1e-6, n_g=1.5)
        expected = 1e-6 * (2.0 - 1.0) ** 1.5
        assert G == pytest.approx(expected, rel=1e-6)

    def test_higher_supersaturation_higher_growth(self):
        G1 = crystal_growth_rate(1.5)
        G2 = crystal_growth_rate(3.0)
        assert G1 < G2


# ---------------------------------------------------------------------------
# simulate_precipitation — basic validation
# ---------------------------------------------------------------------------

class TestSimulatePrecipitationValidation:
    def test_invalid_c_initial_zero(self):
        with pytest.raises(ValueError, match="c_initial_mg_mL"):
            simulate_precipitation(0.0, 1.0)

    def test_invalid_c_initial_negative(self):
        with pytest.raises(ValueError, match="c_initial_mg_mL"):
            simulate_precipitation(-1.0, 1.0)

    def test_invalid_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            simulate_precipitation(1.0, 0.0)

    def test_invalid_solubility_negative(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            simulate_precipitation(1.0, -0.5)

    def test_invalid_k_precip_zero(self):
        with pytest.raises(ValueError, match="k_precip_per_h"):
            simulate_precipitation(1.0, 0.5, k_precip_per_h=0.0)

    def test_invalid_polymer_inhibition_above_one(self):
        with pytest.raises(ValueError, match="polymer_inhibition"):
            simulate_precipitation(1.0, 0.5, polymer_inhibition=1.5)

    def test_invalid_polymer_inhibition_negative(self):
        with pytest.raises(ValueError, match="polymer_inhibition"):
            simulate_precipitation(1.0, 0.5, polymer_inhibition=-0.1)


# ---------------------------------------------------------------------------
# simulate_precipitation — results
# ---------------------------------------------------------------------------

class TestSimulatePrecipitationResults:
    def test_returns_precipitation_result(self):
        r = simulate_precipitation(5.0, 1.0)
        assert isinstance(r, PrecipitationResult)

    def test_frozen_dataclass(self):
        r = simulate_precipitation(5.0, 1.0)
        with pytest.raises((AttributeError, TypeError)):
            r.c_initial_mg_mL = 99.0  # type: ignore[misc]

    def test_no_precipitation_below_solubility(self):
        # C_initial <= solubility => no precipitation, C stays constant
        r = simulate_precipitation(0.5, 1.0, k_precip_per_h=10.0)
        # C should remain approximately constant (no driving force)
        assert r.c_dissolved[-1] == pytest.approx(0.5, rel=1e-3)

    def test_supersaturated_concentration_decreases(self):
        r = simulate_precipitation(5.0, 1.0, t_end_h=4.0)
        assert r.c_dissolved[-1] < r.c_initial_mg_mL

    def test_s_max_equals_ratio(self):
        r = simulate_precipitation(10.0, 2.0)
        assert r.s_max == pytest.approx(5.0, rel=1e-6)

    def test_induction_time_less_than_t_end(self):
        # With high k_precip, induction should be observed quickly
        r = simulate_precipitation(10.0, 1.0, k_precip_per_h=5.0, t_end_h=4.0)
        assert r.induction_time_h < 4.0

    def test_induction_time_at_t_end_if_no_precip(self):
        # No supersaturation, no precipitation
        r = simulate_precipitation(0.5, 1.0, t_end_h=2.0)
        assert r.induction_time_h == pytest.approx(2.0)

    def test_t50_less_than_t_end_for_fast_precip(self):
        r = simulate_precipitation(10.0, 1.0, k_precip_per_h=5.0, t_end_h=4.0)
        assert r.t50_precipitation_h < 4.0

    def test_f_remaining_at_2h_range(self):
        r = simulate_precipitation(5.0, 1.0)
        assert 0.0 <= r.f_remaining_at_2h <= 1.0

    def test_high_k_precip_lower_f_remaining(self):
        r_slow = simulate_precipitation(5.0, 1.0, k_precip_per_h=0.1, t_end_h=4.0)
        r_fast = simulate_precipitation(5.0, 1.0, k_precip_per_h=5.0, t_end_h=4.0)
        assert r_fast.f_remaining_at_2h < r_slow.f_remaining_at_2h

    def test_times_h_length_matches_c_dissolved(self):
        r = simulate_precipitation(5.0, 1.0, t_end_h=2.0, dt_h=0.1)
        assert len(r.times_h) == len(r.c_dissolved)

    def test_times_h_starts_at_zero(self):
        r = simulate_precipitation(5.0, 1.0, t_end_h=2.0)
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_h_ends_at_t_end(self):
        r = simulate_precipitation(5.0, 1.0, t_end_h=3.0, dt_h=0.01)
        assert r.times_h[-1] == pytest.approx(3.0, rel=1e-4)

    def test_c_dissolved_non_negative(self):
        r = simulate_precipitation(5.0, 1.0, k_precip_per_h=10.0, t_end_h=5.0)
        assert all(c >= 0.0 for c in r.c_dissolved)

    def test_polymer_inhibition_reduces_precipitation(self):
        r_no_poly = simulate_precipitation(5.0, 1.0, polymer_conc_mg_mL=0.0)
        r_with_poly = simulate_precipitation(
            5.0, 1.0, polymer_conc_mg_mL=5.0, polymer_inhibition=0.8
        )
        assert r_with_poly.f_remaining_at_2h > r_no_poly.f_remaining_at_2h

    def test_full_inhibition_no_precipitation(self):
        # polymer_inhibition=1.0 and high enough conc => k_eff = 0 => no precip
        r = simulate_precipitation(
            5.0, 1.0, polymer_conc_mg_mL=20.0, polymer_inhibition=1.0, t_end_h=4.0
        )
        # k_eff = k_precip * (1 - 1.0 * min(1, 20/10)) = 0 => no precipitation
        # C should remain constant at c_initial
        assert r.c_dissolved[-1] == pytest.approx(5.0, rel=1e-4)

    def test_notes_is_string(self):
        r = simulate_precipitation(5.0, 1.0)
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# compare_polymer_inhibition
# ---------------------------------------------------------------------------

class TestComparePolymerInhibition:
    def test_returns_list_of_results(self):
        polymers = [
            {"name": "HPMC", "conc_mg_mL": 5.0, "inhibition_factor": 0.7},
            {"name": "PVP", "conc_mg_mL": 3.0, "inhibition_factor": 0.4},
        ]
        results = compare_polymer_inhibition(10.0, 1.0, polymers)
        assert len(results) == 2
        assert all(isinstance(r, PrecipitationResult) for r in results)

    def test_sorted_by_f_remaining_descending(self):
        polymers = [
            {"name": "weak", "conc_mg_mL": 1.0, "inhibition_factor": 0.1},
            {"name": "strong", "conc_mg_mL": 10.0, "inhibition_factor": 0.9},
            {"name": "medium", "conc_mg_mL": 5.0, "inhibition_factor": 0.5},
        ]
        results = compare_polymer_inhibition(10.0, 1.0, polymers)
        fracs = [r.f_remaining_at_2h for r in results]
        assert fracs == sorted(fracs, reverse=True)

    def test_empty_polymer_list(self):
        results = compare_polymer_inhibition(5.0, 1.0, [])
        assert results == []

    def test_single_polymer(self):
        polymers = [{"name": "HPMC", "conc_mg_mL": 5.0, "inhibition_factor": 0.6}]
        results = compare_polymer_inhibition(5.0, 1.0, polymers)
        assert len(results) == 1
