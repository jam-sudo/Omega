"""Tests for Phase 859 — Urinary Bladder Drug Concentration."""

import math
import pytest

from omega_pbpk.core.bladder_pk_p859 import BladderPKResult, simulate_bladder_pk

# Default parameters for a typical drug
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    fu_plasma=0.3,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_bladder_pk(**params)


class TestReturnType:
    def test_returns_bladder_pk_result(self):
        r = _run()
        assert isinstance(r, BladderPKResult)

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

    def test_c_urine_is_list(self):
        r = _run()
        assert isinstance(r.c_urine_mg_L, list)
        assert len(r.c_urine_mg_L) == len(r.times_h)

    def test_c_bladder_wall_is_list(self):
        r = _run()
        assert isinstance(r.c_bladder_wall_mg_g, list)
        assert len(r.c_bladder_wall_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_urine_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_urine_mg_L)

    def test_bladder_wall_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_bladder_wall_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_urine_positive(self):
        r = _run()
        assert r.cmax_urine_mg_L > 0

    def test_cmax_bladder_wall_positive(self):
        r = _run()
        assert r.cmax_bladder_wall_mg_g > 0


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_urine_positive(self):
        r = _run()
        assert r.auc_urine_mg_h_per_L > 0


class TestUrineExcretion:
    def test_urine_excretion_pct_default(self):
        r = _run()
        assert r.urine_excretion_pct == pytest.approx(30.0)

    def test_urine_excretion_pct_custom(self):
        r = _run(fe_urine=0.7)
        assert r.urine_excretion_pct == pytest.approx(70.0)


class TestKpBladderWall:
    def test_kp_in_range(self):
        r = _run()
        assert 0.1 <= r.kp_bladder_wall <= 5.0

    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.0)
        r_high = _run(logP=4.0)
        assert r_high.kp_bladder_wall > r_low.kp_bladder_wall

    def test_negative_logP_kp(self):
        r = _run(logP=-2.0)
        assert r.kp_bladder_wall >= 0.1


class TestBladderWallProportional:
    def test_bladder_wall_proportional_to_urine(self):
        r = _run()
        kp = r.kp_bladder_wall
        for cu, cbw in zip(r.c_urine_mg_L, r.c_bladder_wall_mg_g):
            expected = cu * kp * 0.001
            assert cbw == pytest.approx(expected, abs=1e-12)


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert ratio == pytest.approx(2.0, rel=0.05)


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose"):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose"):
            _run(dose_mg=-10)

    def test_zero_clearance(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_fe_urine_too_high(self):
        with pytest.raises(ValueError, match="fe_urine"):
            _run(fe_urine=1.5)

    def test_fe_urine_negative(self):
        with pytest.raises(ValueError, match="fe_urine"):
            _run(fe_urine=-0.1)

    def test_zero_urine_volume(self):
        with pytest.raises(ValueError, match="urine_volume"):
            _run(urine_volume_L_per_h=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
