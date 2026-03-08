"""Tests for Phase 868 — Interstitial Fluid Drug Distribution."""

import pytest

from omega_pbpk.core.interstitial_fluid_pk_p868 import (
    InterstitialFluidPKResult,
    simulate_interstitial_fluid_pk,
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
    return simulate_interstitial_fluid_pk(**params)


# --- Return type and structure ---


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, InterstitialFluidPKResult)

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

    def test_isf_list_length(self):
        r = _run()
        assert len(r.c_isf_mg_L) == len(r.times_h)


# --- Non-negative concentrations ---


class TestNonNegative:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_isf_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_isf_mg_L)


# --- ISF volume ---


class TestISFVolume:
    def test_default_volume(self):
        r = _run()
        assert r.isf_volume_L == 12.0

    def test_custom_volume(self):
        r = _run(isf_volume_L=15.0)
        assert r.isf_volume_L == 15.0


# --- fu_plasma effect ---


class TestFuPlasmaEffect:
    def test_higher_fu_higher_isf(self):
        r_low = _run(fu_plasma=0.1)
        r_high = _run(fu_plasma=0.9)
        assert r_high.auc_isf_mg_h_per_L > r_low.auc_isf_mg_h_per_L


# --- Capillary permeability ---


class TestCapillaryPermeability:
    def test_positive(self):
        r = _run()
        assert r.capillary_permeability > 0

    def test_higher_mw_lower_permeability(self):
        r_low_mw = _run(mw_Da=200.0)
        r_high_mw = _run(mw_Da=2000.0)
        assert r_low_mw.capillary_permeability > r_high_mw.capillary_permeability

    def test_in_range(self):
        r = _run()
        assert 0.1 <= r.capillary_permeability <= 5.0


# --- kp_isf ---


class TestKpISF:
    def test_in_range(self):
        r = _run()
        assert 0.1 <= r.kp_isf <= 2.0

    def test_higher_fu_higher_kp(self):
        r_low = _run(fu_plasma=0.1, logP=2.0)
        r_high = _run(fu_plasma=0.9, logP=2.0)
        assert r_high.kp_isf > r_low.kp_isf


# --- Equilibrium time ---


class TestEquilibrium:
    def test_positive(self):
        r = _run()
        assert r.t_equilibrium_h > 0

    def test_bounded(self):
        r = _run()
        assert 0.1 <= r.t_equilibrium_h <= 24.0


# --- ISF-to-plasma ratio ---


class TestISFPlasmaRatio:
    def test_positive(self):
        r = _run()
        assert r.isf_to_plasma_ratio > 0


# --- Dose linearity ---


class TestDoseLinearity:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_plasma_mg_L == pytest.approx(2 * r1.cmax_plasma_mg_L, rel=0.05)

    def test_double_dose_doubles_cmax_isf(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_isf_mg_L == pytest.approx(2 * r1.cmax_isf_mg_L, rel=0.05)


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

    def test_isf_volume_zero(self):
        with pytest.raises(ValueError):
            _run(isf_volume_L=0)


# --- Notes ---


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0
