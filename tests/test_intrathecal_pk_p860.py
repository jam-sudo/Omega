"""Tests for Phase 860 — Intrathecal Drug Distribution."""

import math

import pytest

from omega_pbpk.core.intrathecal_pk_p860 import IntrathecalPKResult, simulate_intrathecal_pk

# Default parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=1.5,
    mw_Da=350.0,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_intrathecal_pk(**params)


class TestReturnType:
    def test_returns_intrathecal_pk_result(self):
        r = _run()
        assert isinstance(r, IntrathecalPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_injection_site_default(self):
        r = _run()
        assert r.injection_site == "lumbar"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_csf_lumbar_mg_L) == n
        assert len(r.c_csf_cistern_mg_L) == n
        assert len(r.c_plasma_mg_L) == n


class TestLumbarInjection:
    def test_lumbar_starts_high(self):
        r = _run(injection_site="lumbar")
        assert r.c_csf_lumbar_mg_L[0] > 0

    def test_cistern_starts_zero(self):
        r = _run(injection_site="lumbar")
        assert r.c_csf_cistern_mg_L[0] == 0.0

    def test_cistern_rises_later(self):
        r = _run(injection_site="lumbar")
        # Cistern should rise above zero at some point
        assert max(r.c_csf_cistern_mg_L) > 0


class TestCisternalInjection:
    def test_cistern_starts_high(self):
        r = _run(injection_site="cisternal")
        assert r.c_csf_cistern_mg_L[0] > 0

    def test_lumbar_starts_zero(self):
        r = _run(injection_site="cisternal")
        assert r.c_csf_lumbar_mg_L[0] == 0.0


class TestConcentrations:
    def test_lumbar_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_csf_lumbar_mg_L)

    def test_cistern_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_csf_cistern_mg_L)

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_cmax_lumbar_positive(self):
        r = _run()
        assert r.cmax_lumbar_mg_L > 0

    def test_cmax_cistern_positive(self):
        r = _run()
        assert r.cmax_cistern_mg_L > 0

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0


class TestHalfLife:
    def test_t_half_csf_positive(self):
        r = _run()
        assert r.t_half_csf_h > 0

    def test_t_half_csf_value(self):
        r = _run()
        expected = math.log(2) / (0.3 + 0.05)
        assert r.t_half_csf_h == pytest.approx(expected, rel=0.01)


class TestSystemicExposure:
    def test_systemic_exposure_in_range(self):
        r = _run()
        assert 0 <= r.systemic_exposure_pct <= 100

    def test_plasma_much_lower_than_csf(self):
        r = _run()
        # Plasma exposure should be very low compared to CSF
        assert r.cmax_plasma_mg_L < r.cmax_lumbar_mg_L


class TestAUC:
    def test_auc_lumbar_positive(self):
        r = _run()
        assert r.auc_lumbar_mg_h_per_L > 0

    def test_auc_cistern_positive(self):
        r = _run()
        assert r.auc_cistern_mg_h_per_L > 0


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        ratio = r2.cmax_lumbar_mg_L / r1.cmax_lumbar_mg_L
        assert ratio == pytest.approx(2.0, rel=0.05)


class TestValidation:
    def test_bad_injection_site(self):
        with pytest.raises(ValueError, match="injection_site"):
            _run(injection_site="epidural")

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose"):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose"):
            _run(dose_mg=-5)

    def test_zero_clearance(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_zero_csf_flow(self):
        with pytest.raises(ValueError, match="csf_bulk_flow"):
            _run(csf_bulk_flow_mL_per_h=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
