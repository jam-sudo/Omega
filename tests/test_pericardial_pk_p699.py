"""Tests for Phase 699 — Pericardial Fluid Drug Distribution."""

import math

import pytest

from omega_pbpk.core.pericardial_pk_p699 import (
    PericardialPKResult,
    simulate_pericardial_pk,
)

BASE = {
    "drug_name": "TestDrug",
    "dose_mg": 100.0,
    "logP": 2.0,
    "mw_Da": 400.0,
    "fu_plasma": 0.3,
    "cl_sys_L_per_h": 5.0,
    "vd_sys_L": 50.0,
}


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_pericardial_pk(**params)


class TestReturnType:
    def test_returns_result_type(self):
        r = _run()
        assert isinstance(r, PericardialPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_c_pericardial_is_list(self):
        r = _run()
        assert isinstance(r.c_pericardial_mg_L, list)
        assert len(r.c_pericardial_mg_L) == len(r.times_h)


class TestConcentrations:
    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_non_negative_pericardial(self):
        r = _run()
        assert all(c >= 0 for c in r.c_pericardial_mg_L)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_pericardial_positive(self):
        r = _run()
        assert r.cmax_pericardial_mg_L > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_pericardial_positive(self):
        r = _run()
        assert r.auc_pericardial_mg_h_per_L > 0


class TestPharmacokinetics:
    def test_t_half_pericardial(self):
        r = _run()
        expected = math.log(2) / 0.5
        assert abs(r.t_half_pericardial_h - expected) < 0.01

    def test_pericardial_to_plasma_ratio_positive(self):
        r = _run()
        assert r.pericardial_to_plasma_ratio > 0

    def test_higher_fu_higher_pericardial_exposure(self):
        r_low = _run(fu_plasma=0.1)
        r_high = _run(fu_plasma=0.9)
        assert r_high.auc_pericardial_mg_h_per_L > r_low.auc_pericardial_mg_h_per_L

    def test_dose_linearity_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_dose_linearity_pericardial_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_pericardial_mg_L / r1.cmax_pericardial_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_pericardial_volume_stored(self):
        r = _run(pericardial_volume_mL=20.0)
        assert r.pericardial_volume_mL == 20.0

    def test_default_pericardial_volume(self):
        r = _run()
        assert r.pericardial_volume_mL == 15.0


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10)

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_fu_zero(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=0.0)

    def test_fu_above_one(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=1.5)

    def test_pericardial_volume_zero(self):
        with pytest.raises(ValueError):
            _run(pericardial_volume_mL=0)

    def test_negative_pericardial_volume(self):
        with pytest.raises(ValueError):
            _run(pericardial_volume_mL=-5)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
