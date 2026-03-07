"""Tests for organ_pbpk — Phase 486."""

from __future__ import annotations

import pytest

from omega_pbpk.core.organ_pbpk import (
    OrganPBPKResult,
    calculate_organ_kp,
    organ_steady_state_conc,
    simulate_organ_pbpk,
)

# ---------------------------------------------------------------------------
# calculate_organ_kp
# ---------------------------------------------------------------------------


class TestCalculateOrganKp:
    def test_liver_kp_higher_logP(self):
        kp_low = calculate_organ_kp(logP=1.0, fu_plasma=0.1, organ="liver")
        kp_high = calculate_organ_kp(logP=4.0, fu_plasma=0.1, organ="liver")
        assert kp_high > kp_low

    def test_kidney_kp_higher_logP(self):
        kp_low = calculate_organ_kp(logP=1.0, fu_plasma=0.1, organ="kidney")
        kp_high = calculate_organ_kp(logP=4.0, fu_plasma=0.1, organ="kidney")
        assert kp_high > kp_low

    def test_lung_kp_higher_logP(self):
        kp_low = calculate_organ_kp(logP=1.0, fu_plasma=0.1, organ="lung")
        kp_high = calculate_organ_kp(logP=4.0, fu_plasma=0.1, organ="lung")
        assert kp_high > kp_low

    def test_muscle_kp_higher_logP(self):
        kp_low = calculate_organ_kp(logP=1.0, fu_plasma=0.1, organ="muscle")
        kp_high = calculate_organ_kp(logP=4.0, fu_plasma=0.1, organ="muscle")
        assert kp_high > kp_low

    def test_kp_liver_greater_than_muscle_same_logP(self):
        kp_liver = calculate_organ_kp(logP=3.0, fu_plasma=0.1, organ="liver")
        kp_muscle = calculate_organ_kp(logP=3.0, fu_plasma=0.1, organ="muscle")
        assert kp_liver > kp_muscle

    def test_kp_at_zero_logP_is_one(self):
        for organ in ("liver", "kidney", "lung", "muscle"):
            kp = calculate_organ_kp(logP=0.0, fu_plasma=0.5, organ=organ)
            assert abs(kp - 1.0) < 1e-9

    def test_invalid_organ_raises(self):
        with pytest.raises(ValueError):
            calculate_organ_kp(logP=2.0, fu_plasma=0.1, organ="brain")

    def test_kp_minimum_clamped(self):
        # Very negative logP should not give Kp < 0.1
        kp = calculate_organ_kp(logP=-100.0, fu_plasma=0.1, organ="liver")
        assert kp >= 0.1


# ---------------------------------------------------------------------------
# organ_steady_state_conc
# ---------------------------------------------------------------------------


class TestOrganSteadyStateConc:
    def test_proportional_to_kp(self):
        c1 = organ_steady_state_conc(c_plasma_ss=1.0, kp_organ=2.0)
        c2 = organ_steady_state_conc(c_plasma_ss=1.0, kp_organ=4.0)
        assert abs(c2 / c1 - 2.0) < 1e-9

    def test_proportional_to_plasma_conc(self):
        c1 = organ_steady_state_conc(c_plasma_ss=1.0, kp_organ=3.0)
        c2 = organ_steady_state_conc(c_plasma_ss=2.0, kp_organ=3.0)
        assert abs(c2 / c1 - 2.0) < 1e-9

    def test_zero_plasma_zero_organ(self):
        c = organ_steady_state_conc(c_plasma_ss=0.0, kp_organ=5.0)
        assert c == 0.0

    def test_negative_plasma_raises(self):
        with pytest.raises(ValueError):
            organ_steady_state_conc(c_plasma_ss=-1.0, kp_organ=2.0)

    def test_nonpositive_kp_raises(self):
        with pytest.raises(ValueError):
            organ_steady_state_conc(c_plasma_ss=1.0, kp_organ=0.0)


# ---------------------------------------------------------------------------
# simulate_organ_pbpk — basic
# ---------------------------------------------------------------------------


class TestSimulateOrganPBPK:
    def test_returns_correct_type(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        assert isinstance(result, OrganPBPKResult)

    def test_times_start_at_zero(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        assert result.times_h[0] == 0.0

    def test_times_end_at_t_end_h(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1

    def test_iv_immediate_cmax(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        # For IV, Cmax should occur at or near t=0
        cmax_idx = result.c_plasma_mg_L.index(result.cmax_plasma)
        assert cmax_idx < 5  # within first few steps

    def test_oral_delayed_cmax(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="oral")
        cmax_idx = result.c_plasma_mg_L.index(result.cmax_plasma)
        # Oral Cmax should not be at t=0
        assert cmax_idx > 0

    def test_higher_clint_lower_auc(self):
        r_low = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", cl_int_L_per_h=0.1)
        r_high = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", cl_int_L_per_h=5.0)
        assert r_high.auc_plasma < r_low.auc_plasma

    def test_double_dose_double_cmax(self):
        r1 = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        r2 = simulate_organ_pbpk("TestDrug", dose_mg=200.0, route="iv")
        assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01

    def test_double_dose_double_auc(self):
        r1 = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="oral")
        r2 = simulate_organ_pbpk("TestDrug", dose_mg=200.0, route="oral")
        ratio = r2.auc_plasma / r1.auc_plasma
        assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# simulate_organ_pbpk — tissue concentrations
# ---------------------------------------------------------------------------


class TestSimulateOrganConcentrations:
    def test_liver_higher_than_muscle_lipophilic(self):
        result = simulate_organ_pbpk(
            "LipophilicDrug", dose_mg=100.0, route="iv", logP=4.0, t_end_h=1.0
        )
        # Liver Kp >> Muscle Kp for high logP
        max_liver = max(result.c_liver_mg_kg)
        max_muscle = max(result.c_muscle_mg_kg)
        assert max_liver > max_muscle

    def test_kp_values_stored_correctly(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", logP=2.0)
        assert result.kp_liver > result.kp_kidney
        assert result.kp_kidney > result.kp_lung
        assert result.kp_lung > result.kp_muscle

    def test_concentrations_nonnegative(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=50.0, route="oral")
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)
        assert all(c >= 0.0 for c in result.c_liver_mg_kg)
        assert all(c >= 0.0 for c in result.c_kidney_mg_kg)
        assert all(c >= 0.0 for c in result.c_lung_mg_kg)
        assert all(c >= 0.0 for c in result.c_muscle_mg_kg)

    def test_auc_positive(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        assert result.auc_plasma > 0.0

    def test_cmax_positive(self):
        result = simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv")
        assert result.cmax_plasma > 0.0


# ---------------------------------------------------------------------------
# simulate_organ_pbpk — input validation
# ---------------------------------------------------------------------------


class TestSimulateOrganValidation:
    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="im")

    def test_nonpositive_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=0.0, route="iv")

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", cl_int_L_per_h=-1.0)

    def test_fu_plasma_out_of_range_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", fu_plasma=1.5)

    def test_zero_fu_plasma_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", fu_plasma=0.0)

    def test_nonpositive_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", t_end_h=0.0)

    def test_nonpositive_dt_raises(self):
        with pytest.raises(ValueError):
            simulate_organ_pbpk("TestDrug", dose_mg=100.0, route="iv", dt_h=0.0)
