"""Tests for Phase 872 — Skin Appendage Drug Distribution."""

import pytest

from omega_pbpk.core.skin_appendage_pk_p872 import (
    SkinAppendagePKResult,
    simulate_skin_appendage_pk,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


class TestSkinAppendagePKResult:
    def test_dataclass_fields(self):
        r = SkinAppendagePKResult(drug_name="X", dose_mg=1.0, appendage="nail")
        assert r.drug_name == "X"
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_plasma_mg_L, list)
        assert isinstance(r.c_appendage_mg_g, list)

    def test_default_values(self):
        r = SkinAppendagePKResult(drug_name="X", dose_mg=1.0, appendage="nail")
        assert r.kp_appendage == 0.0
        assert r.cmax_appendage_mg_g == 0.0


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": 0})

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": -1})

    def test_cl_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_skin_appendage_pk(**{**DEFAULTS, "cl_sys_L_per_h": 0})

    def test_vd_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_skin_appendage_pk(**{**DEFAULTS, "vd_sys_L": 0})

    def test_invalid_appendage(self):
        with pytest.raises(ValueError):
            simulate_skin_appendage_pk(**{**DEFAULTS, "appendage": "liver"})


class TestAllAppendages:
    @pytest.mark.parametrize("app", ["hair_follicle", "sebaceous_gland", "sweat_gland", "nail"])
    def test_appendage_accepted(self, app):
        r = simulate_skin_appendage_pk(**DEFAULTS, appendage=app)
        assert isinstance(r, SkinAppendagePKResult)
        assert r.appendage == app

    @pytest.mark.parametrize("app", ["hair_follicle", "sebaceous_gland", "sweat_gland", "nail"])
    def test_cmax_positive(self, app):
        r = simulate_skin_appendage_pk(**DEFAULTS, appendage=app)
        assert r.cmax_appendage_mg_g > 0

    @pytest.mark.parametrize("app", ["hair_follicle", "sebaceous_gland", "sweat_gland", "nail"])
    def test_auc_positive(self, app):
        r = simulate_skin_appendage_pk(**DEFAULTS, appendage=app)
        assert r.auc_appendage_mg_h_per_g > 0


class TestKpValues:
    def test_sebaceous_highest_kp_lipophilic(self):
        """Sebaceous gland should have highest Kp for lipophilic drugs."""
        results = {}
        for app in ["hair_follicle", "sebaceous_gland", "sweat_gland", "nail"]:
            r = simulate_skin_appendage_pk(**{**DEFAULTS, "logP": 3.0}, appendage=app)
            results[app] = r.kp_appendage
        assert results["sebaceous_gland"] == max(results.values())

    def test_sweat_gland_prefers_hydrophilic(self):
        """Sweat gland should have relatively higher Kp for hydrophilic drugs."""
        r_hydro = simulate_skin_appendage_pk(**{**DEFAULTS, "logP": -1.0}, appendage="sweat_gland")
        r_lipo = simulate_skin_appendage_pk(**{**DEFAULTS, "logP": 3.0}, appendage="sweat_gland")
        assert r_hydro.kp_appendage > r_lipo.kp_appendage


class TestNailSlowest:
    def test_nail_longest_half_life(self):
        """Nail should have the longest appendage half-life (slowest)."""
        half_lives = {}
        for app in ["hair_follicle", "sebaceous_gland", "sweat_gland", "nail"]:
            r = simulate_skin_appendage_pk(**DEFAULTS, appendage=app)
            half_lives[app] = r.t_half_appendage_h
        assert half_lives["nail"] == max(half_lives.values())

    def test_nail_t_half_positive(self):
        r = simulate_skin_appendage_pk(**DEFAULTS, appendage="nail")
        assert r.t_half_appendage_h > 0


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": 100.0})
        r2 = simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": 200.0})
        ratio = r2.cmax_appendage_mg_g / r1.cmax_appendage_mg_g
        assert abs(ratio - 2.0) < 0.2

    def test_double_dose_doubles_auc(self):
        r1 = simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": 100.0})
        r2 = simulate_skin_appendage_pk(**{**DEFAULTS, "dose_mg": 200.0})
        ratio = r2.auc_appendage_mg_h_per_g / r1.auc_appendage_mg_h_per_g
        assert abs(ratio - 2.0) < 0.2


class TestTimeSeries:
    def test_times_start_at_zero(self):
        r = simulate_skin_appendage_pk(**DEFAULTS)
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = simulate_skin_appendage_pk(**DEFAULTS, t_end_h=48.0)
        assert abs(r.times_h[-1] - 48.0) < 0.1

    def test_plasma_rises_then_falls(self):
        r = simulate_skin_appendage_pk(**DEFAULTS)
        cmax_idx = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
        assert cmax_idx > 0
        assert cmax_idx < len(r.c_plasma_mg_L) - 1


class TestAppendageToPlasmaRatio:
    def test_ratio_positive(self):
        r = simulate_skin_appendage_pk(**DEFAULTS)
        assert r.appendage_to_plasma_ratio > 0

    def test_ratio_calculated_correctly(self):
        r = simulate_skin_appendage_pk(**DEFAULTS)
        expected = (
            r.auc_appendage_mg_h_per_g / r.auc_plasma_mg_h_per_L
            if r.auc_plasma_mg_h_per_L > 0
            else 0.0
        )
        assert abs(r.appendage_to_plasma_ratio - expected) < 1e-6


class TestNotes:
    def test_notes_contain_drug_name(self):
        r = simulate_skin_appendage_pk(**DEFAULTS)
        assert "TestDrug" in r.notes

    def test_notes_contain_appendage(self):
        r = simulate_skin_appendage_pk(**DEFAULTS, appendage="nail")
        assert "nail" in r.notes
