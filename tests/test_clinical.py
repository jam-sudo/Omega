"""Tests for clinical modules: DDI report and NCA."""
import math
import numpy as np
import pytest


class TestDDIReport:
    def test_no_inhibition_r1_is_one(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        inh = DDIInhibitor(name="safe", cmax_uM=1.0)  # no Ki values set
        r = assess_ddi_risk(inh)
        assert r.R1_3a4 == pytest.approx(1.0)

    def test_strong_inhibitor_flags(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        inh = DDIInhibitor(name="ketoconazole", cmax_uM=5.0, ki_3a4_uM=0.01)
        r = assess_ddi_risk(inh)
        assert r.R1_3a4 >= 2.0
        assert any("3A4" in f for f in r.flags)

    def test_r1_formula(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        inh = DDIInhibitor(name="test", cmax_uM=2.0, ki_3a4_uM=2.0)
        r = assess_ddi_risk(inh)
        assert r.R1_3a4 == pytest.approx(2.0, rel=0.01)

    def test_no_flags_gives_no_study_recommendation(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        # No Ki values set means no CYP inhibition; cmax very low, no induction
        inh = DDIInhibitor(name="low_risk", cmax_uM=0.01)
        r = assess_ddi_risk(inh)
        assert not r.flags
        assert r.recommendation == "No clinical DDI study warranted"

    def test_mbi_r2_no_inhibition(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        inh = DDIInhibitor(name="test", cmax_uM=1.0, kinact_3a4_per_h=0.0)
        r = assess_ddi_risk(inh)
        assert r.R2_3a4 == pytest.approx(1.0)

    def test_format_report_returns_string(self):
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk, format_report
        inh = DDIInhibitor(name="test", cmax_uM=1.0, ki_3a4_uM=0.5)
        r = assess_ddi_risk(inh)
        s = format_report(r)
        assert isinstance(s, str)
        assert "DDI Risk" in s


class TestNCA:
    def _make_iv_pk(self, dose=100.0, cl=10.0, vd=50.0, n=50):
        """Generate synthetic IV PK profile."""
        kel = cl / vd
        t = np.linspace(0, 24, n)
        c = (dose / vd) * np.exp(-kel * t)
        return t, c, dose

    def test_nca_cmax_equals_c0_for_iv(self):
        from omega_pbpk.clinical.nca import run_nca
        t, c, dose = self._make_iv_pk()
        r = run_nca(t, c, dose)
        assert r.cmax_mg_L == pytest.approx(c[0], rel=0.05)

    def test_nca_auc_close_to_dose_over_cl(self):
        from omega_pbpk.clinical.nca import run_nca
        cl = 10.0
        t, c, dose = self._make_iv_pk(cl=cl)
        r = run_nca(t, c, dose)
        assert r.auc0inf_mg_h_L == pytest.approx(dose / cl, rel=0.05)

    def test_nca_t_half_correct(self):
        from omega_pbpk.clinical.nca import run_nca
        vd, cl = 50.0, 10.0
        t, c, dose = self._make_iv_pk(cl=cl, vd=vd)
        expected_t_half = 0.693 * vd / cl
        r = run_nca(t, c, dose)
        assert r.t_half_h == pytest.approx(expected_t_half, rel=0.1)

    def test_nca_cl_close_to_true(self):
        from omega_pbpk.clinical.nca import run_nca
        cl = 10.0
        t, c, dose = self._make_iv_pk(cl=cl)
        r = run_nca(t, c, dose)
        assert r.cl_L_per_h == pytest.approx(cl, rel=0.1)

    def test_nca_r_squared_high_for_mono_exponential(self):
        from omega_pbpk.clinical.nca import run_nca
        t, c, dose = self._make_iv_pk()
        r = run_nca(t, c, dose)
        assert r.r_squared >= 0.99

    def test_nca_tmax_at_zero_for_iv(self):
        from omega_pbpk.clinical.nca import run_nca
        t, c, dose = self._make_iv_pk()
        r = run_nca(t, c, dose)
        assert r.tmax_h == pytest.approx(0.0, abs=0.5)
