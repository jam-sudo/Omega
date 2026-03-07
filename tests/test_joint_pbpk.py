"""Tests for cartilage/joint drug distribution PBPK model (Phase 884)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.joint_pbpk import (
    JointPKResult,
    compare_injection_volumes,
    simulate_joint_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> JointPKResult:
    params = dict(
        drug_name="triamcinolone",
        dose_mg=40.0,
        v_synovial_mL=3.0,
        k_sf_turnover_per_h=0.2,
        k_cart_uptake_per_h=0.05,
        k_cart_release_per_h=0.01,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
        t_end_h=72.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_joint_pk(**params)


# ---------------------------------------------------------------------------
# Result type and shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_returns_correct_type(self):
        r = _default()
        assert isinstance(r, JointPKResult)

    def test_times_match_synovial(self):
        r = _default(t_end_h=24.0, dt_h=0.2)
        assert len(r.times_h) == len(r.c_synovial_mg_L)

    def test_times_match_cartilage(self):
        r = _default(t_end_h=24.0, dt_h=0.2)
        assert len(r.times_h) == len(r.c_cartilage_mg_L)

    def test_times_match_plasma(self):
        r = _default(t_end_h=24.0, dt_h=0.2)
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        r = _default(t_end_h=48.0)
        assert r.times_h[-1] == pytest.approx(48.0)

    def test_drug_name_stored(self):
        r = _default(drug_name="cortisone")
        assert r.drug_name == "cortisone"

    def test_dose_stored(self):
        r = _default(dose_mg=20.0)
        assert r.dose_mg == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_synovial_initial_conc(self):
        """C_synovial(0) = dose_mg / v_synovial_L."""
        dose = 40.0
        vol_mL = 3.0
        v_L = vol_mL / 1000.0
        r = _default(dose_mg=dose, v_synovial_mL=vol_mL)
        assert r.c_synovial_mg_L[0] == pytest.approx(dose / v_L, rel=1e-4)

    def test_cartilage_zero_at_t0(self):
        r = _default()
        assert r.c_cartilage_mg_L[0] == pytest.approx(0.0)

    def test_plasma_zero_at_t0(self):
        r = _default()
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_synovial_declines_overall(self):
        r = _default()
        assert r.c_synovial_mg_L[0] > r.c_synovial_mg_L[-1]

    def test_cartilage_rises_then_falls(self):
        r = _default()
        cc = r.c_cartilage_mg_L
        assert r.cmax_cartilage_mg_L > cc[0]
        assert r.cmax_cartilage_mg_L > cc[-1]

    def test_auc_synovial_positive(self):
        r = _default()
        assert r.auc_synovial > 0.0

    def test_auc_cartilage_positive(self):
        r = _default()
        assert r.auc_cartilage > 0.0

    def test_cmax_synovial_at_t0(self):
        """Since all drug is in synovial fluid at t=0, cmax is at t=0."""
        r = _default()
        assert r.cmax_synovial_mg_L == pytest.approx(r.c_synovial_mg_L[0], rel=1e-4)
        assert r.tmax_synovial_h == pytest.approx(0.0, abs=0.2)


# ---------------------------------------------------------------------------
# Synovial half-life
# ---------------------------------------------------------------------------


class TestSynovialHalfLife:
    def test_synovial_t_half_formula(self):
        """t½ = ln(2) / (k_sf_turnover + k_cart_uptake)."""
        k_turn = 0.2
        k_uptk = 0.05
        r = _default(k_sf_turnover_per_h=k_turn, k_cart_uptake_per_h=k_uptk)
        expected = math.log(2) / (k_turn + k_uptk)
        assert r.synovial_t_half_h == pytest.approx(expected, rel=1e-6)

    def test_slower_turnover_longer_half_life(self):
        r_fast = _default(k_sf_turnover_per_h=1.0)
        r_slow = _default(k_sf_turnover_per_h=0.05)
        assert r_slow.synovial_t_half_h > r_fast.synovial_t_half_h


# ---------------------------------------------------------------------------
# Cartilage penetration
# ---------------------------------------------------------------------------


class TestCartilagePenetration:
    def test_cartilage_auc_ratio_between_0_and_positive(self):
        r = _default()
        assert r.cartilage_auc_ratio >= 0.0

    def test_higher_uptake_higher_cartilage_ratio(self):
        r_low = _default(k_cart_uptake_per_h=0.01)
        r_high = _default(k_cart_uptake_per_h=0.3)
        assert r_high.cartilage_auc_ratio > r_low.cartilage_auc_ratio


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_good_penetration_note(self):
        # Very high uptake relative to turnover → good penetration
        r = _default(k_cart_uptake_per_h=2.0, k_sf_turnover_per_h=0.01)
        assert "good" in r.notes.lower()

    def test_poor_penetration_note(self):
        # Very low uptake → poor penetration
        r = _default(k_cart_uptake_per_h=0.001, k_sf_turnover_per_h=2.0)
        assert "poor" in r.notes.lower() or "moderate" in r.notes.lower()


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


class TestLinearity:
    def test_double_dose_doubles_cmax_synovial(self):
        r1 = _default(dose_mg=20.0)
        r2 = _default(dose_mg=40.0)
        assert r2.cmax_synovial_mg_L == pytest.approx(r1.cmax_synovial_mg_L * 2.0, rel=1e-4)

    def test_double_dose_doubles_auc_synovial(self):
        r1 = _default(dose_mg=20.0)
        r2 = _default(dose_mg=40.0)
        assert r2.auc_synovial == pytest.approx(r1.auc_synovial * 2.0, rel=1e-3)

    def test_double_dose_doubles_auc_cartilage(self):
        r1 = _default(dose_mg=20.0)
        r2 = _default(dose_mg=40.0)
        assert r2.auc_cartilage == pytest.approx(r1.auc_cartilage * 2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# compare_injection_volumes
# ---------------------------------------------------------------------------


class TestCompareInjectionVolumes:
    def test_returns_three_results_by_default(self):
        results = compare_injection_volumes("hyaluronate", 20.0)
        assert len(results) == 3

    def test_sorted_by_auc_cartilage_descending(self):
        results = compare_injection_volumes("hyaluronate", 20.0)
        aucs = [r.auc_cartilage for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_custom_volumes(self):
        results = compare_injection_volumes("drug", 10.0, volumes_mL=[2.0, 4.0])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_joint_pk("x", -1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_joint_pk("x", 0.0)

    def test_zero_synovial_vol_raises(self):
        with pytest.raises(ValueError, match="v_synovial_mL"):
            simulate_joint_pk("x", 10.0, v_synovial_mL=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_joint_pk("x", 10.0, cl_sys_L_per_h=-1.0)
