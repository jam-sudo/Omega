"""Tests for biopharmaceutics/crystallization_kinetics.py — Phase 477."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.crystallization_kinetics import (
    CrystallizationResult,
    crystal_growth_rate,
    estimate_precipitation_risk,
    induction_time_estimate,
    nucleation_rate,
    simulate_crystallization,
)

# ---------------------------------------------------------------------------
# nucleation_rate tests
# ---------------------------------------------------------------------------


class TestNucleationRate:
    def test_s_equals_one_returns_zero(self):
        assert nucleation_rate(1.0) == 0.0

    def test_s_less_than_one_returns_zero(self):
        assert nucleation_rate(0.5) == 0.0

    def test_s_greater_than_one_positive_rate(self):
        rate = nucleation_rate(2.0)
        assert rate > 0.0

    def test_higher_s_faster_nucleation(self):
        """Higher supersaturation → faster nucleation rate."""
        rate_low = nucleation_rate(1.5)
        rate_high = nucleation_rate(5.0)
        assert rate_high > rate_low

    def test_higher_surface_energy_slower_nucleation(self):
        """Higher surface energy → higher energy barrier → slower nucleation."""
        rate_low_gamma = nucleation_rate(3.0, surface_energy_mJ_m2=2.0)
        rate_high_gamma = nucleation_rate(3.0, surface_energy_mJ_m2=20.0)
        assert rate_low_gamma > rate_high_gamma

    def test_temperature_effect(self):
        """Higher T reduces energy barrier → faster nucleation."""
        rate_cold = nucleation_rate(3.0, temperature_C=25.0)
        rate_warm = nucleation_rate(3.0, temperature_C=50.0)
        assert rate_warm > rate_cold

    def test_returns_float(self):
        rate = nucleation_rate(2.0)
        assert isinstance(rate, float)

    def test_invalid_s_zero_raises(self):
        with pytest.raises(ValueError, match="supersaturation_ratio must be > 0"):
            nucleation_rate(0.0)

    def test_invalid_s_negative_raises(self):
        with pytest.raises(ValueError, match="supersaturation_ratio must be > 0"):
            nucleation_rate(-1.0)

    def test_invalid_surface_energy_negative_raises(self):
        with pytest.raises(ValueError, match="surface_energy_mJ_m2 must be >= 0"):
            nucleation_rate(2.0, surface_energy_mJ_m2=-1.0)

    def test_zero_surface_energy_large_rate(self):
        """Zero surface energy → no barrier → very high nucleation rate."""
        rate = nucleation_rate(2.0, surface_energy_mJ_m2=0.0)
        assert rate > 0.0


# ---------------------------------------------------------------------------
# crystal_growth_rate tests
# ---------------------------------------------------------------------------


class TestCrystalGrowthRate:
    def test_s_equals_one_returns_zero(self):
        assert crystal_growth_rate(1.0) == 0.0

    def test_s_less_than_one_returns_zero(self):
        assert crystal_growth_rate(0.8) == 0.0

    def test_s_greater_than_one_positive_rate(self):
        assert crystal_growth_rate(2.0) > 0.0

    def test_higher_s_faster_growth(self):
        rate_low = crystal_growth_rate(1.5)
        rate_high = crystal_growth_rate(3.0)
        assert rate_high > rate_low

    def test_higher_kg_faster_growth(self):
        rate_low = crystal_growth_rate(2.0, kg=0.05)
        rate_high = crystal_growth_rate(2.0, kg=0.5)
        assert rate_high > rate_low

    def test_power_law_correctness(self):
        """RG = kg * (S-1)^g; at S=3, kg=1, g=2: RG = (3-1)^2 = 4."""
        rate = crystal_growth_rate(3.0, kg=1.0, g_exponent=2.0)
        assert abs(rate - 4.0) < 1e-9

    def test_invalid_s_raises(self):
        with pytest.raises(ValueError, match="supersaturation_ratio must be > 0"):
            crystal_growth_rate(0.0)

    def test_invalid_kg_negative_raises(self):
        with pytest.raises(ValueError, match="kg must be >= 0"):
            crystal_growth_rate(2.0, kg=-0.1)

    def test_invalid_g_negative_raises(self):
        with pytest.raises(ValueError, match="g_exponent must be >= 0"):
            crystal_growth_rate(2.0, g_exponent=-1.0)


# ---------------------------------------------------------------------------
# induction_time_estimate tests
# ---------------------------------------------------------------------------


class TestInductionTimeEstimate:
    def test_s_le_one_returns_large_value(self):
        t = induction_time_estimate(1.0)
        assert t >= 1.0e6

    def test_s_lt_one_returns_large_value(self):
        t = induction_time_estimate(0.5)
        assert t >= 1.0e6

    def test_s_gt_one_finite_induction_time(self):
        t = induction_time_estimate(3.0)
        assert math.isfinite(t)
        assert t > 0

    def test_higher_s_shorter_induction_time(self):
        """Higher supersaturation → nucleation faster → shorter induction time."""
        t_low = induction_time_estimate(2.0)
        t_high = induction_time_estimate(10.0)
        assert t_high < t_low

    def test_higher_surface_energy_longer_induction(self):
        """Higher surface energy → slower nucleation → longer induction time."""
        t_low_gamma = induction_time_estimate(3.0, surface_energy_mJ_m2=2.0)
        t_high_gamma = induction_time_estimate(3.0, surface_energy_mJ_m2=20.0)
        assert t_high_gamma > t_low_gamma

    def test_invalid_s_raises(self):
        with pytest.raises(ValueError):
            induction_time_estimate(0.0)

    def test_invalid_surface_energy_raises(self):
        with pytest.raises(ValueError):
            induction_time_estimate(2.0, surface_energy_mJ_m2=-5.0)


# ---------------------------------------------------------------------------
# simulate_crystallization tests
# ---------------------------------------------------------------------------


class TestSimulateCrystallization:
    def test_returns_crystallization_result(self):
        result = simulate_crystallization(
            initial_conc_mg_mL=1.0,
            equilibrium_solubility_mg_mL=0.1,
        )
        assert isinstance(result, CrystallizationResult)

    def test_no_crystallization_when_s_le_one(self):
        """When initial conc <= equilibrium, no crystallization."""
        result = simulate_crystallization(
            initial_conc_mg_mL=0.05,
            equilibrium_solubility_mg_mL=0.1,
        )
        # All crystallized should be zero
        assert all(c == 0.0 for c in result.conc_crystallized_mg_mL)
        assert math.isnan(result.induction_time_h)

    def test_dissolved_decreases_over_time(self):
        """Dissolved concentration should decrease from supersaturated state."""
        result = simulate_crystallization(
            initial_conc_mg_mL=5.0,
            equilibrium_solubility_mg_mL=0.5,
            t_end_h=2.0,
            kg=1.0,
        )
        assert result.conc_dissolved_mg_mL[-1] < result.conc_dissolved_mg_mL[0]

    def test_crystallized_increases_over_time(self):
        """Crystallized mass should increase from supersaturated state."""
        result = simulate_crystallization(
            initial_conc_mg_mL=5.0,
            equilibrium_solubility_mg_mL=0.5,
            t_end_h=2.0,
            kg=1.0,
        )
        assert result.conc_crystallized_mg_mL[-1] > result.conc_crystallized_mg_mL[0]

    def test_mass_conservation(self):
        """C_dissolved + C_crystallized ≈ initial_conc at all times."""
        result = simulate_crystallization(
            initial_conc_mg_mL=2.0,
            equilibrium_solubility_mg_mL=0.2,
            t_end_h=1.0,
            kg=0.5,
        )
        C0 = result.conc_dissolved_mg_mL[0] + result.conc_crystallized_mg_mL[0]
        for cd, cc in zip(result.conc_dissolved_mg_mL, result.conc_crystallized_mg_mL):
            total = cd + cc
            assert abs(total - C0) < 1e-6, f"Mass not conserved: {total} != {C0}"

    def test_equilibrium_approached(self):
        """With sufficient time and high kg, dissolved approaches C_eq."""
        result = simulate_crystallization(
            initial_conc_mg_mL=1.0,
            equilibrium_solubility_mg_mL=0.1,
            t_end_h=10.0,
            dt_h=0.01,
            kg=5.0,
        )
        final_conc = result.conc_dissolved_mg_mL[-1]
        assert final_conc <= 0.11  # within 10% of C_eq

    def test_supersaturation_starts_above_one(self):
        result = simulate_crystallization(
            initial_conc_mg_mL=2.0,
            equilibrium_solubility_mg_mL=0.5,
        )
        assert result.supersaturation[0] == pytest.approx(4.0)

    def test_supersaturation_does_not_go_below_one(self):
        """Dissolved concentration should not drop below equilibrium solubility."""
        result = simulate_crystallization(
            initial_conc_mg_mL=2.0,
            equilibrium_solubility_mg_mL=0.5,
            t_end_h=5.0,
            kg=2.0,
        )
        for cd in result.conc_dissolved_mg_mL:
            assert cd >= 0.5 - 1e-6

    def test_lists_same_length(self):
        result = simulate_crystallization(
            initial_conc_mg_mL=1.0,
            equilibrium_solubility_mg_mL=0.2,
        )
        n = len(result.times_h)
        assert len(result.conc_dissolved_mg_mL) == n
        assert len(result.conc_crystallized_mg_mL) == n
        assert len(result.supersaturation) == n

    def test_t90_crystallized_nan_when_not_reached(self):
        """With very low kg, 90% crystallization may not be reached."""
        result = simulate_crystallization(
            initial_conc_mg_mL=0.2,
            equilibrium_solubility_mg_mL=0.1,
            t_end_h=0.5,
            kg=0.001,
        )
        # With very slow growth, may not reach 90%
        # Just check it is either a valid float or nan
        assert math.isnan(result.t90_crystallized_h) or result.t90_crystallized_h >= 0

    def test_notes_string_not_empty(self):
        result = simulate_crystallization(
            initial_conc_mg_mL=1.0,
            equilibrium_solubility_mg_mL=0.1,
        )
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_invalid_initial_conc_raises(self):
        with pytest.raises(ValueError, match="initial_conc_mg_mL must be > 0"):
            simulate_crystallization(0.0, 0.1)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="equilibrium_solubility_mg_mL must be > 0"):
            simulate_crystallization(1.0, 0.0)

    def test_invalid_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h must be > 0"):
            simulate_crystallization(1.0, 0.1, t_end_h=0.0)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h must be > 0"):
            simulate_crystallization(1.0, 0.1, dt_h=0.0)

    def test_higher_kg_faster_crystallization(self):
        """Higher kg should lead to more crystallization in same time."""
        result_slow = simulate_crystallization(
            initial_conc_mg_mL=2.0,
            equilibrium_solubility_mg_mL=0.2,
            t_end_h=1.0,
            kg=0.01,
        )
        result_fast = simulate_crystallization(
            initial_conc_mg_mL=2.0,
            equilibrium_solubility_mg_mL=0.2,
            t_end_h=1.0,
            kg=2.0,
        )
        assert result_fast.conc_crystallized_mg_mL[-1] > result_slow.conc_crystallized_mg_mL[-1]


# ---------------------------------------------------------------------------
# estimate_precipitation_risk tests
# ---------------------------------------------------------------------------


class TestEstimatePrecipitationRisk:
    def test_high_auc_ratio_is_high_risk(self):
        """Very high AUC relative to solubility → 'high' risk."""
        risk = estimate_precipitation_risk(
            auc_supersaturated=100.0,  # mg/mL·h
            equilibrium_solubility_mg_mL=1.0,
            t_window_h=1.0,
        )
        assert risk == "high"

    def test_low_auc_ratio_is_low_risk(self):
        """AUC close to solubility → 'low' risk."""
        risk = estimate_precipitation_risk(
            auc_supersaturated=1.5,
            equilibrium_solubility_mg_mL=1.0,
            t_window_h=1.0,
        )
        assert risk == "low"

    def test_moderate_risk(self):
        """Mean S between 2 and 5 → 'moderate'."""
        risk = estimate_precipitation_risk(
            auc_supersaturated=3.0,
            equilibrium_solubility_mg_mL=1.0,
            t_window_h=1.0,
        )
        assert risk == "moderate"

    def test_boundary_low_moderate(self):
        """Exactly at threshold S=2 → 'moderate'."""
        risk = estimate_precipitation_risk(2.0, 1.0, 1.0)
        assert risk == "moderate"

    def test_boundary_moderate_high(self):
        """Exactly at threshold S=5 → 'high'."""
        risk = estimate_precipitation_risk(5.0, 1.0, 1.0)
        assert risk == "high"

    def test_zero_auc_is_low_risk(self):
        risk = estimate_precipitation_risk(0.0, 1.0, 1.0)
        assert risk == "low"

    def test_invalid_auc_negative_raises(self):
        with pytest.raises(ValueError, match="auc_supersaturated must be >= 0"):
            estimate_precipitation_risk(-1.0, 1.0, 1.0)

    def test_invalid_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="equilibrium_solubility_mg_mL must be > 0"):
            estimate_precipitation_risk(1.0, 0.0, 1.0)

    def test_invalid_t_window_raises(self):
        with pytest.raises(ValueError, match="t_window_h must be > 0"):
            estimate_precipitation_risk(1.0, 1.0, 0.0)

    def test_longer_window_reduces_risk(self):
        """Same AUC spread over longer time → lower mean S → lower risk."""
        risk_short = estimate_precipitation_risk(5.0, 1.0, 1.0)  # mean S=5 → high
        risk_long = estimate_precipitation_risk(5.0, 1.0, 5.0)  # mean S=1 → low
        assert risk_short in ("high", "moderate")
        assert risk_long == "low"
