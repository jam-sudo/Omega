"""Tests for biologic/mAb antibody PBPK model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.antibody_pbpk import (
    AntibodyPKResult,
    compare_sc_iv,
    simulate_antibody_pk,
)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_antibody_pk("mAb", dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_antibody_pk("mAb", dose_mg=-10)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_day"):
            simulate_antibody_pk("mAb", 100, cl_L_per_day=0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_day"):
            simulate_antibody_pk("mAb", 100, cl_L_per_day=-0.1)

    def test_vd_central_zero_raises(self):
        with pytest.raises(ValueError, match="vd_central_L"):
            simulate_antibody_pk("mAb", 100, vd_central_L=0)

    def test_vd_tissue_zero_raises(self):
        with pytest.raises(ValueError, match="vd_tissue_L"):
            simulate_antibody_pk("mAb", 100, vd_tissue_L=0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_antibody_pk("mAb", 100, route="oral")


# ---------------------------------------------------------------------------
# IV route
# ---------------------------------------------------------------------------


class TestIVRoute:
    def test_iv_initial_concentration(self):
        """C_c(0+) should be close to dose/Vc for IV bolus."""
        r = simulate_antibody_pk("mAb", 300, route="iv", vd_central_L=3.0)
        expected = 300 / 3.0  # 100 mg/L
        assert abs(r.cc_mg_L[0] - expected) < 1.0

    def test_iv_half_life_with_fcrn(self):
        """FcRn recycling should give t½ > 5 days for typical mAb."""
        r = simulate_antibody_pk(
            "mAb", 300, route="iv", fcrn_recycling=0.7, t_end_day=56, dt_day=0.1
        )
        assert r.t_half_day > 5.0

    def test_iv_bioavailability_is_one(self):
        r = simulate_antibody_pk("mAb", 300, route="iv")
        assert r.sc_bioavailability == 1.0

    def test_iv_tmax_at_zero(self):
        """For IV bolus, Cmax should be at t=0."""
        r = simulate_antibody_pk("mAb", 300, route="iv")
        assert r.tmax_day == 0.0

    def test_tissue_cmax_less_than_central(self):
        """Tissue Cmax should be lower than central Cmax for IV."""
        r = simulate_antibody_pk("mAb", 300, route="iv")
        tissue_cmax = max(r.ct_mg_L)
        assert tissue_cmax < r.cmax_mg_L


# ---------------------------------------------------------------------------
# SC route
# ---------------------------------------------------------------------------


class TestSCRoute:
    def test_sc_cmax_lower_than_iv(self):
        """SC Cmax should be lower than IV Cmax (same dose)."""
        sc = simulate_antibody_pk("mAb", 300, route="sc")
        iv = simulate_antibody_pk("mAb", 300, route="iv")
        assert sc.cmax_mg_L < iv.cmax_mg_L

    def test_sc_tmax_positive(self):
        """SC should have delayed peak (tmax > 0)."""
        r = simulate_antibody_pk("mAb", 300, route="sc")
        assert r.tmax_day > 0.0

    def test_sc_bioavailability_equals_f_sc(self):
        r = simulate_antibody_pk("mAb", 300, route="sc", f_sc=0.65)
        assert r.sc_bioavailability == 0.65

    def test_iv_auc_greater_than_sc(self):
        """IV AUC > SC AUC (same dose, SC has f_sc < 1)."""
        sc = simulate_antibody_pk("mAb", 300, route="sc", f_sc=0.7, t_end_day=56, dt_day=0.1)
        iv = simulate_antibody_pk("mAb", 300, route="iv", t_end_day=56, dt_day=0.1)
        assert iv.auc_0_t > sc.auc_0_t


# ---------------------------------------------------------------------------
# FcRn recycling
# ---------------------------------------------------------------------------


class TestFcRnRecycling:
    def test_higher_fcrn_longer_half_life(self):
        """Higher FcRn recycling should extend half-life."""
        r_low = simulate_antibody_pk(
            "mAb", 300, route="iv", fcrn_recycling=0.0, t_end_day=56, dt_day=0.1
        )
        r_high = simulate_antibody_pk(
            "mAb", 300, route="iv", fcrn_recycling=0.7, t_end_day=56, dt_day=0.1
        )
        assert r_high.t_half_day > r_low.t_half_day

    def test_fcrn_zero_shorter_half_life(self):
        """fcrn_recycling=0 should give shorter t½ than 0.7."""
        r0 = simulate_antibody_pk(
            "mAb", 300, route="iv", fcrn_recycling=0.0, t_end_day=56, dt_day=0.1
        )
        r7 = simulate_antibody_pk(
            "mAb", 300, route="iv", fcrn_recycling=0.7, t_end_day=56, dt_day=0.1
        )
        assert r0.t_half_day < r7.t_half_day


# ---------------------------------------------------------------------------
# Linear PK (dose proportionality without TMDD)
# ---------------------------------------------------------------------------


class TestLinearPK:
    def test_dose_proportional_cmax(self):
        """2x dose -> ~2x Cmax (no TMDD)."""
        r1 = simulate_antibody_pk("mAb", 100, route="iv")
        r2 = simulate_antibody_pk("mAb", 200, route="iv")
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_dose_proportional_auc(self):
        """2x dose -> ~2x AUC (no TMDD)."""
        r1 = simulate_antibody_pk("mAb", 100, route="iv", t_end_day=56, dt_day=0.1)
        r2 = simulate_antibody_pk("mAb", 200, route="iv", t_end_day=56, dt_day=0.1)
        ratio = r2.auc_0_t / r1.auc_0_t
        assert abs(ratio - 2.0) < 0.2


# ---------------------------------------------------------------------------
# TMDD
# ---------------------------------------------------------------------------


class TestTMDD:
    def test_tmdd_reduces_auc(self):
        """TMDD should reduce AUC compared to no-TMDD (same dose)."""
        r_no = simulate_antibody_pk("mAb", 300, route="iv", tmdd=False, t_end_day=56, dt_day=0.1)
        r_yes = simulate_antibody_pk("mAb", 300, route="iv", tmdd=True, t_end_day=56, dt_day=0.1)
        assert r_yes.auc_0_t < r_no.auc_0_t

    def test_tmdd_enabled_flag(self):
        r = simulate_antibody_pk("mAb", 300, tmdd=True)
        assert r.tmdd_enabled is True

    def test_tmdd_disabled_flag(self):
        r = simulate_antibody_pk("mAb", 300, tmdd=False)
        assert r.tmdd_enabled is False


# ---------------------------------------------------------------------------
# compare_sc_iv
# ---------------------------------------------------------------------------


class TestCompareSCIV:
    def test_returns_tuple_of_two(self):
        sc, iv = compare_sc_iv("mAb", 300)
        assert isinstance(sc, AntibodyPKResult)
        assert isinstance(iv, AntibodyPKResult)

    def test_routes_correct(self):
        sc, iv = compare_sc_iv("mAb", 300)
        assert sc.route == "sc"
        assert iv.route == "iv"


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_times_length(self):
        r = simulate_antibody_pk("mAb", 300, t_end_day=28, dt_day=0.05)
        expected_len = int(28 / 0.05) + 1
        assert len(r.times_day) == expected_len

    def test_notes_contains_drug_name_and_route(self):
        r = simulate_antibody_pk("trastuzumab", 300, route="iv")
        assert "trastuzumab" in r.notes
        assert "IV" in r.notes

    def test_all_concentrations_finite(self):
        r = simulate_antibody_pk("mAb", 300, route="iv")
        assert all(math.isfinite(c) for c in r.cc_mg_L)
        assert all(math.isfinite(c) for c in r.ct_mg_L)

    def test_no_nan_in_key_metrics(self):
        r = simulate_antibody_pk("mAb", 300, route="iv", t_end_day=56, dt_day=0.1)
        assert math.isfinite(r.cmax_mg_L)
        assert math.isfinite(r.auc_0_t)
        assert math.isfinite(r.t_half_day)

    def test_frozen_dataclass(self):
        r = simulate_antibody_pk("mAb", 300)
        with pytest.raises(AttributeError):
            r.dose_mg = 999  # type: ignore[misc]
