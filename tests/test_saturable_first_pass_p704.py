"""Tests for Phase 704 — Saturable First-Pass Metabolism."""

import pytest

from omega_pbpk.core.saturable_first_pass_p704 import (
    SaturableFirstPassResult,
    simulate_saturable_first_pass,
)


def _default():
    return simulate_saturable_first_pass(
        drug_name="Propranolol",
        dose_mg=80.0,
        vmax_mg_per_h=500.0,
        km_mg_L=10.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=250.0,
    )


def _high_dose():
    return simulate_saturable_first_pass(
        drug_name="Propranolol",
        dose_mg=800.0,
        vmax_mg_per_h=500.0,
        km_mg_L=10.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=250.0,
        t_end_h=48.0,
    )


def _low_dose():
    return simulate_saturable_first_pass(
        drug_name="Propranolol",
        dose_mg=5.0,
        vmax_mg_per_h=500.0,
        km_mg_L=10.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=250.0,
    )


class TestReturnType:
    def test_returns_result(self):
        r = _default()
        assert isinstance(r, SaturableFirstPassResult)

    def test_drug_name(self):
        r = _default()
        assert r.drug_name == "Propranolol"

    def test_dose_mg(self):
        r = _default()
        assert r.dose_mg == 80.0

    def test_times_is_list(self):
        r = _default()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_plasma_is_list(self):
        r = _default()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)


class TestConcentrations:
    def test_non_negative_plasma(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_mg_L > 0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_mg_h_per_L > 0

    def test_tmax_positive(self):
        r = _default()
        assert r.tmax_h > 0


class TestBioavailability:
    def test_f_bioavailable_in_range(self):
        r = _default()
        assert 0 < r.f_bioavailable <= 1.0

    def test_high_dose_higher_f(self):
        r_high = _high_dose()
        r_low = _low_dose()
        assert r_high.f_bioavailable > r_low.f_bioavailable

    def test_low_dose_lower_f(self):
        r_low = simulate_saturable_first_pass(
            drug_name="Drug",
            dose_mg=10.0,
            vmax_mg_per_h=50.0,
            km_mg_L=5.0,
            cl_sys_L_per_h=3.0,
            vd_sys_L=100.0,
            t_end_h=48.0,
        )
        r_high = simulate_saturable_first_pass(
            drug_name="Drug",
            dose_mg=500.0,
            vmax_mg_per_h=50.0,
            km_mg_L=5.0,
            cl_sys_L_per_h=3.0,
            vd_sys_L=100.0,
            t_end_h=48.0,
        )
        assert r_low.f_bioavailable < r_high.f_bioavailable


class TestNonlinearity:
    def test_nonlinearity_index_in_range(self):
        r = _default()
        assert 0.0 <= r.nonlinearity_index <= 1.0

    def test_high_dose_higher_nonlinearity(self):
        r_high = _high_dose()
        r_low = _low_dose()
        assert r_high.nonlinearity_index > r_low.nonlinearity_index

    def test_dose_normalized_auc_increases_with_dose(self):
        r_high = _high_dose()
        r_low = _low_dose()
        assert r_high.dose_normalized_auc > r_low.dose_normalized_auc

    def test_superlinear_auc(self):
        r1 = simulate_saturable_first_pass(
            drug_name="Drug",
            dose_mg=100.0,
            vmax_mg_per_h=200.0,
            km_mg_L=5.0,
            cl_sys_L_per_h=3.0,
            vd_sys_L=100.0,
            t_end_h=48.0,
        )
        r2 = simulate_saturable_first_pass(
            drug_name="Drug",
            dose_mg=200.0,
            vmax_mg_per_h=200.0,
            km_mg_L=5.0,
            cl_sys_L_per_h=3.0,
            vd_sys_L=100.0,
            t_end_h=48.0,
        )
        # 2x dose should give more than 2x AUC due to saturation
        assert r2.auc_mg_h_per_L > 2.0 * r1.auc_mg_h_per_L


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", 0, 500.0, 10.0, 5.0, 250.0)

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", -10, 500.0, 10.0, 5.0, 250.0)

    def test_vmax_zero(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", 80, 0, 10.0, 5.0, 250.0)

    def test_km_zero(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", 80, 500.0, 0, 5.0, 250.0)

    def test_cl_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", 80, 500.0, 10.0, 0, 250.0)

    def test_vd_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_saturable_first_pass("X", 80, 500.0, 10.0, 5.0, 0)


class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
