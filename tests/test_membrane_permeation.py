"""Tests for core/membrane_permeation.py — Phase 475."""

from __future__ import annotations

import pytest

from omega_pbpk.core.membrane_permeation import (
    FranzDiffusionResult,
    calculate_permeability_coefficient,
    lag_time_analytical,
    simulate_franz_diffusion,
    steady_state_flux,
)

# ---------------------------------------------------------------------------
# lag_time_analytical
# ---------------------------------------------------------------------------


class TestLagTimeAnalytical:
    def test_formula_basic(self):
        """t_lag = L² / (6D)."""
        L_um = 400.0
        D = 1e-5  # cm²/h
        L_cm = L_um * 1e-4
        expected = L_cm**2 / (6.0 * D)
        result = lag_time_analytical(L_um, D)
        assert abs(result - expected) < 1e-12

    def test_thicker_membrane_longer_lag(self):
        D = 1e-5
        t_thin = lag_time_analytical(200.0, D)
        t_thick = lag_time_analytical(400.0, D)
        assert t_thick > t_thin

    def test_higher_D_shorter_lag(self):
        L = 400.0
        t_slow = lag_time_analytical(L, 1e-6)
        t_fast = lag_time_analytical(L, 1e-4)
        assert t_fast < t_slow

    def test_quadratic_scaling_with_thickness(self):
        """Doubling L → 4x lag time."""
        D = 1e-5
        t1 = lag_time_analytical(200.0, D)
        t2 = lag_time_analytical(400.0, D)
        assert abs(t2 / t1 - 4.0) < 1e-10

    def test_inverse_scaling_with_D(self):
        """Doubling D → half lag time."""
        L = 300.0
        t1 = lag_time_analytical(L, 1e-5)
        t2 = lag_time_analytical(L, 2e-5)
        assert abs(t1 / t2 - 2.0) < 1e-10

    def test_positive_thickness_required(self):
        with pytest.raises(ValueError):
            lag_time_analytical(0.0, 1e-5)

    def test_positive_D_required(self):
        with pytest.raises(ValueError):
            lag_time_analytical(400.0, 0.0)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError):
            lag_time_analytical(-10.0, 1e-5)


# ---------------------------------------------------------------------------
# steady_state_flux
# ---------------------------------------------------------------------------


class TestSteadyStateFlux:
    def test_linear_with_donor_concentration(self):
        J1 = steady_state_flux(1.0, 400.0, 1e-5, 1.0)
        J2 = steady_state_flux(2.0, 400.0, 1e-5, 1.0)
        assert abs(J2 / J1 - 2.0) < 1e-10

    def test_higher_D_higher_flux(self):
        J_slow = steady_state_flux(1.0, 400.0, 1e-6, 1.0)
        J_fast = steady_state_flux(1.0, 400.0, 1e-4, 1.0)
        assert J_fast > J_slow

    def test_higher_kp_higher_flux(self):
        J_low = steady_state_flux(1.0, 400.0, 1e-5, 1.0)
        J_high = steady_state_flux(1.0, 400.0, 1e-5, 5.0)
        assert J_high > J_low

    def test_linear_with_partition_coeff(self):
        J1 = steady_state_flux(1.0, 400.0, 1e-5, 1.0)
        J2 = steady_state_flux(1.0, 400.0, 1e-5, 3.0)
        assert abs(J2 / J1 - 3.0) < 1e-10

    def test_thicker_membrane_lower_flux(self):
        J_thin = steady_state_flux(1.0, 200.0, 1e-5, 1.0)
        J_thick = steady_state_flux(1.0, 400.0, 1e-5, 1.0)
        assert J_thin > J_thick

    def test_positive_donor_conc_required(self):
        with pytest.raises(ValueError):
            steady_state_flux(0.0, 400.0, 1e-5, 1.0)

    def test_positive_thickness_required(self):
        with pytest.raises(ValueError):
            steady_state_flux(1.0, 0.0, 1e-5, 1.0)

    def test_positive_D_required(self):
        with pytest.raises(ValueError):
            steady_state_flux(1.0, 400.0, 0.0, 1.0)

    def test_positive_kp_required(self):
        with pytest.raises(ValueError):
            steady_state_flux(1.0, 400.0, 1e-5, 0.0)


# ---------------------------------------------------------------------------
# calculate_permeability_coefficient
# ---------------------------------------------------------------------------


class TestCalculatePermeabilityCoefficient:
    def test_basic_calculation(self):
        papp = calculate_permeability_coefficient(0.01, 1.0)
        assert abs(papp - 0.01) < 1e-12

    def test_scales_with_flux(self):
        p1 = calculate_permeability_coefficient(0.005, 1.0)
        p2 = calculate_permeability_coefficient(0.010, 1.0)
        assert abs(p2 / p1 - 2.0) < 1e-10

    def test_inversely_proportional_to_concentration(self):
        p1 = calculate_permeability_coefficient(0.01, 1.0)
        p2 = calculate_permeability_coefficient(0.01, 2.0)
        assert abs(p1 / p2 - 2.0) < 1e-10

    def test_zero_donor_conc_raises(self):
        with pytest.raises(ValueError):
            calculate_permeability_coefficient(0.01, 0.0)

    def test_negative_flux_raises(self):
        with pytest.raises(ValueError):
            calculate_permeability_coefficient(-0.01, 1.0)

    def test_zero_flux_returns_zero(self):
        assert calculate_permeability_coefficient(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# simulate_franz_diffusion
# ---------------------------------------------------------------------------


class TestSimulateFranzDiffusion:
    def test_returns_dataclass(self):
        result = simulate_franz_diffusion("TestDrug", 10.0)
        assert isinstance(result, FranzDiffusionResult)

    def test_drug_name_stored(self):
        result = simulate_franz_diffusion("Ibuprofen", 5.0)
        assert result.drug_name == "Ibuprofen"

    def test_donor_decreases_over_time(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=24.0, dt_h=0.5)
        # Donor should generally decrease
        assert result.c_donor_mg_mL[-1] < result.c_donor_mg_mL[0]

    def test_acceptor_increases_over_time(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=24.0, dt_h=0.5)
        assert result.c_acceptor_mg_mL[-1] > result.c_acceptor_mg_mL[0]

    def test_cumulative_monotonically_increasing(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=24.0, dt_h=0.5)
        for i in range(1, len(result.cumulative_permeated_mg)):
            assert result.cumulative_permeated_mg[i] >= result.cumulative_permeated_mg[i - 1]

    def test_initial_cumulative_is_zero(self):
        result = simulate_franz_diffusion("Drug", 10.0)
        assert result.cumulative_permeated_mg[0] == 0.0

    def test_time_vector_starts_at_zero(self):
        result = simulate_franz_diffusion("Drug", 10.0)
        assert result.times_h[0] == 0.0

    def test_time_vector_ends_at_t_end(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=12.0, dt_h=0.5)
        assert abs(result.times_h[-1] - 12.0) < 1e-9

    def test_papp_positive(self):
        result = simulate_franz_diffusion("Drug", 10.0)
        assert result.papp_cm_h >= 0.0

    def test_lag_time_stored(self):
        result = simulate_franz_diffusion(
            "Drug", 10.0, membrane_thickness_um=400.0, diffusion_coeff_cm2_h=1e-5
        )
        expected_lag = lag_time_analytical(400.0, 1e-5)
        assert abs(result.lag_time_h - expected_lag) < 1e-10

    def test_thicker_membrane_longer_lag_in_result(self):
        r_thin = simulate_franz_diffusion("Drug", 10.0, membrane_thickness_um=200.0)
        r_thick = simulate_franz_diffusion("Drug", 10.0, membrane_thickness_um=400.0)
        assert r_thick.lag_time_h > r_thin.lag_time_h

    def test_higher_D_shorter_lag_in_result(self):
        r_slow = simulate_franz_diffusion("Drug", 10.0, diffusion_coeff_cm2_h=1e-6)
        r_fast = simulate_franz_diffusion("Drug", 10.0, diffusion_coeff_cm2_h=1e-4)
        assert r_fast.lag_time_h < r_slow.lag_time_h

    def test_higher_partition_faster_permeation(self):
        r_low = simulate_franz_diffusion("Drug", 10.0, partition_coeff=1.0, t_end_h=24.0)
        r_high = simulate_franz_diffusion("Drug", 10.0, partition_coeff=5.0, t_end_h=24.0)
        assert r_high.cumulative_permeated_mg[-1] > r_low.cumulative_permeated_mg[-1]

    def test_ss_flux_stored_correctly(self):
        D = 1e-5
        L = 400.0
        Kp = 2.0
        dose = 10.0
        vol = 5.0
        result = simulate_franz_diffusion(
            "Drug",
            dose,
            membrane_thickness_um=L,
            diffusion_coeff_cm2_h=D,
            partition_coeff=Kp,
            donor_volume_mL=vol,
        )
        c0 = dose / vol
        expected_ss = steady_state_flux(c0, L, D, Kp)
        assert abs(result.steady_state_flux_mg_cm2_h - expected_ss) < 1e-10

    def test_notes_not_empty(self):
        result = simulate_franz_diffusion("Drug", 10.0)
        assert len(result.notes) > 0

    def test_validation_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_franz_diffusion("Drug", -1.0)

    def test_validation_zero_thickness(self):
        with pytest.raises(ValueError):
            simulate_franz_diffusion("Drug", 10.0, membrane_thickness_um=0.0)

    def test_validation_negative_D(self):
        with pytest.raises(ValueError):
            simulate_franz_diffusion("Drug", 10.0, diffusion_coeff_cm2_h=-1e-5)

    def test_validation_zero_area(self):
        with pytest.raises(ValueError):
            simulate_franz_diffusion("Drug", 10.0, area_cm2=0.0)

    def test_higher_D_more_permeated(self):
        r_slow = simulate_franz_diffusion("Drug", 10.0, diffusion_coeff_cm2_h=1e-6, t_end_h=48.0)
        r_fast = simulate_franz_diffusion("Drug", 10.0, diffusion_coeff_cm2_h=1e-4, t_end_h=48.0)
        assert r_fast.cumulative_permeated_mg[-1] > r_slow.cumulative_permeated_mg[-1]

    def test_profile_lengths_consistent(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=10.0, dt_h=1.0)
        n = len(result.times_h)
        assert len(result.c_donor_mg_mL) == n
        assert len(result.c_acceptor_mg_mL) == n
        assert len(result.cumulative_permeated_mg) == n
        assert len(result.flux_mg_cm2_h) == n

    def test_flux_non_negative(self):
        result = simulate_franz_diffusion("Drug", 10.0, t_end_h=24.0, dt_h=0.5)
        for f in result.flux_mg_cm2_h:
            assert f >= 0.0

    def test_larger_area_more_permeated(self):
        r_small = simulate_franz_diffusion("Drug", 10.0, area_cm2=0.5, t_end_h=24.0)
        r_large = simulate_franz_diffusion("Drug", 10.0, area_cm2=2.0, t_end_h=24.0)
        assert r_large.cumulative_permeated_mg[-1] > r_small.cumulative_permeated_mg[-1]

    def test_initial_donor_concentration(self):
        dose = 10.0
        vol = 5.0
        result = simulate_franz_diffusion("Drug", dose, donor_volume_mL=vol)
        expected_c0 = dose / vol
        assert abs(result.c_donor_mg_mL[0] - expected_c0) < 1e-10

    def test_initial_acceptor_zero(self):
        result = simulate_franz_diffusion("Drug", 10.0)
        assert result.c_acceptor_mg_mL[0] == 0.0
