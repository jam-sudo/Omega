"""Tests for Phase 870 — Ascitic Fluid Drug Distribution."""

import pytest

from omega_pbpk.core.ascitic_fluid_pk_p870 import (
    AsciticFluidPKResult,
    simulate_ascitic_fluid_pk,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    fu_plasma=0.5,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_ascitic_fluid_pk(**params)


# --- Return type and structure ---


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, AsciticFluidPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list) and len(r.times_h) > 1

    def test_plasma_list_length(self):
        r = _run()
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_ascites_list_length(self):
        r = _run()
        assert len(r.c_ascites_mg_L) == len(r.times_h)


# --- Non-negative concentrations ---


class TestNonNegative:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_ascites_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_ascites_mg_L)


# --- Ascites volume ---


class TestAscitesVolume:
    def test_default_volume(self):
        r = _run()
        assert r.ascites_volume_L == 5.0

    def test_custom_volume(self):
        r = _run(ascites_volume_L=10.0)
        assert r.ascites_volume_L == 10.0


# --- fu_plasma effect ---


class TestFuPlasmaEffect:
    def test_higher_fu_higher_ascites(self):
        r_low = _run(fu_plasma=0.1)
        r_high = _run(fu_plasma=0.9)
        assert r_high.auc_ascites_mg_h_per_L > r_low.auc_ascites_mg_h_per_L


# --- Molecular weight effect ---


class TestMWEffect:
    def test_higher_mw_lower_penetration(self):
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=2000.0)
        assert r_low.auc_ascites_mg_h_per_L > r_high.auc_ascites_mg_h_per_L


# --- Half-life ---


class TestHalfLife:
    def test_positive(self):
        r = _run()
        assert r.t_half_ascites_h > 0

    def test_expected_value(self):
        import math

        r = _run()
        expected = math.log(2) / 0.02
        assert r.t_half_ascites_h == pytest.approx(expected, rel=0.01)


# --- Ascites-to-plasma ratio ---


class TestAscitesPlasmaRatio:
    def test_positive(self):
        r = _run()
        assert r.ascites_to_plasma_ratio > 0


# --- Dose adjustment factor ---


class TestDoseAdjustment:
    def test_greater_than_one(self):
        r = _run()
        assert r.dose_adjustment_factor > 1.0

    def test_expected_value(self):
        r = _run(vd_sys_L=50.0, ascites_volume_L=5.0)
        assert r.dose_adjustment_factor == pytest.approx(55.0 / 50.0, rel=0.01)

    def test_larger_ascites_higher_adjustment(self):
        r_small = _run(ascites_volume_L=2.0)
        r_large = _run(ascites_volume_L=10.0)
        assert r_large.dose_adjustment_factor > r_small.dose_adjustment_factor


# --- Dose linearity ---


class TestDoseLinearity:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_plasma_mg_L == pytest.approx(2 * r1.cmax_plasma_mg_L, rel=0.05)

    def test_double_dose_doubles_cmax_ascites(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_ascites_mg_L == pytest.approx(2 * r1.cmax_ascites_mg_L, rel=0.05)

    def test_double_dose_doubles_auc_plasma(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.auc_plasma_mg_h_per_L == pytest.approx(2 * r1.auc_plasma_mg_h_per_L, rel=0.05)


# --- Validation errors ---


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_zero_cl(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_fu_zero(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=0)

    def test_fu_above_one(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=1.5)

    def test_ascites_volume_zero(self):
        with pytest.raises(ValueError):
            _run(ascites_volume_L=0)

    def test_negative_ascites_volume(self):
        with pytest.raises(ValueError):
            _run(ascites_volume_L=-5)


# --- Notes ---


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_notes_contains_drug_name(self):
        r = _run()
        assert "TestDrug" in r.notes
