"""Tests for Phase 697 — Drug Concentration in Tears."""

import pytest

from omega_pbpk.core.tear_fluid_pk_p697 import TearFluidPKResult, simulate_tear_fluid_pk

BASE = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    fu_plasma=0.5,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**kw):
    params = {**BASE, **kw}
    return simulate_tear_fluid_pk(**params)


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, TearFluidPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_route_default(self):
        r = _run()
        assert r.route == "systemic"

    def test_times_list(self):
        r = _run()
        assert isinstance(r.times_h, list) and len(r.times_h) > 1

    def test_plasma_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list) and len(r.c_plasma_mg_L) == len(r.times_h)

    def test_tear_list(self):
        r = _run()
        assert isinstance(r.c_tear_mg_L, list) and len(r.c_tear_mg_L) == len(r.times_h)


class TestSystemicRoute:
    def test_plasma_nonzero(self):
        r = _run(route="systemic")
        assert max(r.c_plasma_mg_L) > 0

    def test_tear_follows_plasma(self):
        r = _run(route="systemic")
        assert r.cmax_tear_mg_L > 0

    def test_cmax_plasma_positive(self):
        r = _run(route="systemic")
        assert r.cmax_plasma_mg_L > 0

    def test_auc_plasma_positive(self):
        r = _run(route="systemic")
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_tear_positive(self):
        r = _run(route="systemic")
        assert r.auc_tear_mg_h_per_L > 0

    def test_tear_to_plasma_ratio_positive(self):
        r = _run(route="systemic")
        assert r.tear_to_plasma_ratio > 0


class TestTopicalRoute:
    def test_tear_starts_high(self):
        r = _run(route="topical")
        assert r.c_tear_mg_L[0] > 1000  # very high initial

    def test_tear_decays(self):
        r = _run(route="topical")
        assert r.c_tear_mg_L[-1] < r.c_tear_mg_L[0]

    def test_plasma_all_zero(self):
        r = _run(route="topical")
        assert all(c == 0.0 for c in r.c_plasma_mg_L)

    def test_cmax_tear_positive(self):
        r = _run(route="topical")
        assert r.cmax_tear_mg_L > 0

    def test_auc_tear_positive_topical(self):
        r = _run(route="topical")
        assert r.auc_tear_mg_h_per_L > 0


class TestNonNegative:
    def test_plasma_nonneg(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_tear_nonneg(self):
        r = _run()
        assert all(c >= 0 for c in r.c_tear_mg_L)


class TestParameters:
    def test_k_drainage(self):
        r = _run()
        assert r.k_drainage_per_h == 30.0

    def test_tear_volume(self):
        r = _run()
        assert r.tear_volume_mL == 0.007

    def test_higher_fu_higher_tear(self):
        r1 = _run(fu_plasma=0.1, route="systemic")
        r2 = _run(fu_plasma=0.9, route="systemic")
        assert r2.auc_tear_mg_h_per_L > r1.auc_tear_mg_h_per_L

    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=10.0, route="systemic")
        r2 = _run(dose_mg=20.0, route="systemic")
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.9 < ratio < 2.1


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

    def test_bad_fu(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=0.0)

    def test_bad_route(self):
        with pytest.raises(ValueError):
            _run(route="inhalation")

    def test_bad_tear_volume(self):
        with pytest.raises(ValueError):
            _run(tear_volume_mL=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0
