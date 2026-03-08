"""Tests for Phase 703 — Adipose Tissue Sequestration."""

import pytest

from omega_pbpk.core.adipose_sequestration_p703 import (
    AdiposeSequestrationResult,
    simulate_adipose_sequestration,
)


def _default():
    return simulate_adipose_sequestration(
        drug_name="Amiodarone",
        dose_mg=200.0,
        logP=7.6,
        mw_Da=645.0,
        cl_sys_L_per_h=0.6,
        vd_sys_L=60.0,
    )


def _low_logp():
    return simulate_adipose_sequestration(
        drug_name="LowLogP",
        dose_mg=200.0,
        logP=1.0,
        mw_Da=300.0,
        cl_sys_L_per_h=0.6,
        vd_sys_L=60.0,
    )


class TestReturnType:
    def test_returns_result(self):
        r = _default()
        assert isinstance(r, AdiposeSequestrationResult)

    def test_drug_name(self):
        r = _default()
        assert r.drug_name == "Amiodarone"

    def test_dose_mg(self):
        r = _default()
        assert r.dose_mg == 200.0

    def test_n_doses(self):
        r = _default()
        assert r.n_doses == 1

    def test_times_is_list(self):
        r = _default()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_plasma_is_list(self):
        r = _default()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_c_adipose_is_list(self):
        r = _default()
        assert isinstance(r.c_adipose_mg_g, list)
        assert len(r.c_adipose_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_non_negative_plasma(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_non_negative_adipose(self):
        r = _default()
        assert all(c >= 0 for c in r.c_adipose_mg_g)

    def test_cmax_plasma_positive(self):
        r = _default()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_adipose_positive(self):
        r = _default()
        assert r.cmax_adipose_mg_g > 0


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _default()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_adipose_positive(self):
        r = _default()
        assert r.auc_adipose_mg_h_per_g > 0


class TestKpAndWashout:
    def test_kp_in_range(self):
        r = _default()
        assert 0.5 <= r.kp_adipose <= 50.0

    def test_kp_low_logp_in_range(self):
        r = _low_logp()
        assert 0.5 <= r.kp_adipose <= 50.0

    def test_higher_logp_higher_kp(self):
        r_high = _default()  # logP=7.6
        r_low = _low_logp()  # logP=1.0
        assert r_high.kp_adipose > r_low.kp_adipose

    def test_higher_logp_slower_washout(self):
        r_high = _default()
        r_low = _low_logp()
        assert r_high.washout_t50_h > r_low.washout_t50_h

    def test_washout_t50_positive(self):
        r = _default()
        assert r.washout_t50_h > 0


class TestMultipleDosing:
    def test_n_doses_3_higher_adipose_cmax(self):
        r1 = simulate_adipose_sequestration(
            drug_name="TestDrug",
            dose_mg=100.0,
            logP=4.0,
            mw_Da=400.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=50.0,
            n_doses=1,
            t_end_h=120.0,
        )
        r3 = simulate_adipose_sequestration(
            drug_name="TestDrug",
            dose_mg=100.0,
            logP=4.0,
            mw_Da=400.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=50.0,
            n_doses=3,
            dosing_interval_h=24.0,
            t_end_h=120.0,
        )
        assert r3.cmax_adipose_mg_g > r1.cmax_adipose_mg_g

    def test_accumulation_ratio_gte_1(self):
        r = simulate_adipose_sequestration(
            drug_name="TestDrug",
            dose_mg=100.0,
            logP=4.0,
            mw_Da=400.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=50.0,
            n_doses=3,
            dosing_interval_h=24.0,
            t_end_h=120.0,
        )
        assert r.accumulation_ratio >= 1.0

    def test_single_dose_accumulation_ratio_1(self):
        r = _default()
        assert r.accumulation_ratio == 1.0


class TestDoseProportionality:
    def test_double_dose_double_cmax(self):
        r1 = simulate_adipose_sequestration(
            drug_name="Drug",
            dose_mg=100.0,
            logP=3.0,
            mw_Da=300.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=50.0,
            n_doses=1,
        )
        r2 = simulate_adipose_sequestration(
            drug_name="Drug",
            dose_mg=200.0,
            logP=3.0,
            mw_Da=300.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=50.0,
            n_doses=1,
        )
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.9 < ratio < 2.1


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 0, 3.0, 300.0, 1.0, 50.0)

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", -10, 3.0, 300.0, 1.0, 50.0)

    def test_cl_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 100, 3.0, 300.0, 0, 50.0)

    def test_vd_sys_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 100, 3.0, 300.0, 1.0, 0)

    def test_fat_mass_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 100, 3.0, 300.0, 1.0, 50.0, fat_mass_kg=0)

    def test_n_doses_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 100, 3.0, 300.0, 1.0, 50.0, n_doses=0)

    def test_dosing_interval_zero(self):
        with pytest.raises(ValueError):
            simulate_adipose_sequestration("X", 100, 3.0, 300.0, 1.0, 50.0, dosing_interval_h=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
