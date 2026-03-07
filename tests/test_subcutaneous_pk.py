"""Tests for core.subcutaneous_pk (Phase 322)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.subcutaneous_pk import (
    SCPKResult,
    compare_sc_iv,
    sc_bioavailability,
    simulate_sc_pk,
)


# ---------------------------------------------------------------------------
# simulate_sc_pk — validation
# ---------------------------------------------------------------------------

class TestSimulateScPKValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_sc_pk("drug", dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_sc_pk("drug", dose_mg=-10.0)

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            simulate_sc_pk("drug", dose_mg=100.0, mw_kDa=0.0)

    def test_invalid_ka_cap_zero(self):
        with pytest.raises(ValueError, match="ka_capillary"):
            simulate_sc_pk("drug", dose_mg=100.0, ka_capillary_per_h=0.0)

    def test_invalid_ka_lymph_zero(self):
        with pytest.raises(ValueError, match="ka_lymph"):
            simulate_sc_pk("drug", dose_mg=100.0, ka_lymph_per_h=0.0)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_sc_pk("drug", dose_mg=100.0, cl_L_per_h=0.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_sc_pk("drug", dose_mg=100.0, vd_L=0.0)

    def test_invalid_f_lymph_above_one(self):
        with pytest.raises(ValueError, match="f_lymph"):
            simulate_sc_pk("drug", dose_mg=100.0, f_lymph=1.5)

    def test_invalid_f_lymph_negative(self):
        with pytest.raises(ValueError, match="f_lymph"):
            simulate_sc_pk("drug", dose_mg=100.0, f_lymph=-0.1)


# ---------------------------------------------------------------------------
# simulate_sc_pk — results
# ---------------------------------------------------------------------------

class TestSimulateScPKResults:
    def test_returns_scpk_result(self):
        r = simulate_sc_pk("insulin", dose_mg=1.0, mw_kDa=5.8)
        assert isinstance(r, SCPKResult)

    def test_frozen_dataclass(self):
        r = simulate_sc_pk("insulin", dose_mg=1.0)
        with pytest.raises((AttributeError, TypeError)):
            r.dose_mg = 99.0  # type: ignore[misc]

    def test_small_molecule_low_f_lymph(self):
        # MW=0.5 kDa => f_lymph ~ sigmoid(-2.5)*0.5 ~ small
        r = simulate_sc_pk("small_drug", dose_mg=100.0, mw_kDa=0.5)
        assert r.f_lymph < 0.1

    def test_large_molecule_high_f_lymph(self):
        # MW=20 kDa => f_lymph ~ sigmoid(17)*0.5 ~ 0.5
        r = simulate_sc_pk("large_biologic", dose_mg=10.0, mw_kDa=20.0)
        assert r.f_lymph > 0.4

    def test_explicit_f_lymph_used(self):
        r = simulate_sc_pk("drug", dose_mg=100.0, f_lymph=0.3)
        assert r.f_lymph == pytest.approx(0.3)

    def test_cmax_positive(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert r.cmax > 0.0

    def test_tmax_within_simulation(self):
        r = simulate_sc_pk("drug", dose_mg=100.0, t_end_h=24.0)
        assert 0.0 <= r.tmax_h <= 24.0

    def test_auc_positive(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert r.auc > 0.0

    def test_times_h_length_matches_c_plasma(self):
        r = simulate_sc_pk("drug", dose_mg=100.0, t_end_h=12.0, dt_h=0.5)
        assert len(r.times_h) == len(r.c_plasma)

    def test_times_h_starts_at_zero(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert r.times_h[0] == pytest.approx(0.0)

    def test_c_plasma_starts_at_zero(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert r.c_plasma[0] == pytest.approx(0.0)

    def test_c_plasma_non_negative(self):
        r = simulate_sc_pk("drug", dose_mg=100.0, t_end_h=48.0)
        assert all(c >= 0.0 for c in r.c_plasma)

    def test_f_via_routes_sum_to_one(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert r.f_via_capillary + r.f_via_lymph == pytest.approx(1.0, abs=1e-4)

    def test_higher_dose_higher_cmax(self):
        r1 = simulate_sc_pk("drug", dose_mg=50.0)
        r2 = simulate_sc_pk("drug", dose_mg=200.0)
        assert r2.cmax > r1.cmax

    def test_drug_name_preserved(self):
        r = simulate_sc_pk("adalimumab", dose_mg=40.0, mw_kDa=148.0)
        assert r.drug_name == "adalimumab"

    def test_notes_is_string(self):
        r = simulate_sc_pk("drug", dose_mg=100.0)
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_large_molecule_slower_absorption(self):
        # Large molecule with low ka_lymph => slower Tmax
        r_small = simulate_sc_pk("small", dose_mg=100.0, mw_kDa=0.3, ka_capillary_per_h=1.0)
        r_large = simulate_sc_pk(
            "large",
            dose_mg=100.0,
            mw_kDa=150.0,
            ka_capillary_per_h=0.1,
            ka_lymph_per_h=0.02,
        )
        # Large molecule should have later Tmax or lower Cmax
        assert r_large.tmax_h >= r_small.tmax_h or r_large.cmax <= r_small.cmax


# ---------------------------------------------------------------------------
# sc_bioavailability
# ---------------------------------------------------------------------------

class TestScBioavailability:
    def test_small_hydrophilic_high_f(self):
        # Small MW, low logP => high SC F
        f = sc_bioavailability(mw_kDa=0.3, logP=0.5, solubility_mg_mL=10.0)
        assert f > 0.8

    def test_large_molecule_lower_f(self):
        # Large MW => lower SC F due to proteolytic degradation
        f_small = sc_bioavailability(mw_kDa=0.3, logP=1.0, solubility_mg_mL=1.0)
        f_large = sc_bioavailability(mw_kDa=150.0, logP=1.0, solubility_mg_mL=1.0)
        assert f_large < f_small

    def test_lipophilic_penalty(self):
        # High logP => lower F
        f_hydrophilic = sc_bioavailability(mw_kDa=0.5, logP=0.5, solubility_mg_mL=10.0)
        f_lipophilic = sc_bioavailability(mw_kDa=0.5, logP=8.0, solubility_mg_mL=0.01)
        assert f_lipophilic < f_hydrophilic

    def test_clamped_to_range(self):
        # All valid inputs should give [0.1, 0.99]
        f1 = sc_bioavailability(mw_kDa=0.1, logP=-2.0, solubility_mg_mL=100.0)
        f2 = sc_bioavailability(mw_kDa=1000.0, logP=10.0, solubility_mg_mL=0.001)
        assert 0.1 <= f1 <= 0.99
        assert 0.1 <= f2 <= 0.99

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            sc_bioavailability(mw_kDa=0.0, logP=1.0, solubility_mg_mL=1.0)

    def test_invalid_solubility_negative(self):
        with pytest.raises(ValueError, match="solubility"):
            sc_bioavailability(mw_kDa=1.0, logP=1.0, solubility_mg_mL=-0.1)


# ---------------------------------------------------------------------------
# compare_sc_iv
# ---------------------------------------------------------------------------

class TestCompareScIv:
    def test_returns_dict(self):
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        for key in ("auc_sc", "auc_iv", "auc_ratio", "cmax_sc", "cmax_iv", "cmax_ratio",
                    "tmax_sc_h", "f_lymph"):
            assert key in result

    def test_iv_auc_analytical(self):
        # IV AUC = dose / CL
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        assert result["auc_iv"] == pytest.approx(20.0, rel=1e-6)  # 100/5

    def test_iv_cmax_analytical(self):
        # IV Cmax = dose / Vd
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        assert result["cmax_iv"] == pytest.approx(2.0, rel=1e-6)  # 100/50

    def test_auc_ratio_less_than_one(self):
        # SC AUC should be <= IV AUC (bioavailability <= 1)
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        assert result["auc_ratio"] <= 1.05  # allow small numerical overshoot

    def test_cmax_sc_less_than_iv(self):
        # SC Cmax should be less than IV Cmax (slower absorption)
        result = compare_sc_iv("drug", dose_mg=100.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)
        assert result["cmax_sc"] < result["cmax_iv"]

    def test_invalid_dose(self):
        with pytest.raises(ValueError):
            compare_sc_iv("drug", dose_mg=0.0, mw_kDa=0.5, cl_L_per_h=5.0, vd_L=50.0)

    def test_drug_name_in_result(self):
        result = compare_sc_iv("myDrug", dose_mg=50.0, mw_kDa=1.0, cl_L_per_h=2.0, vd_L=20.0)
        assert result["drug_name"] == "myDrug"
