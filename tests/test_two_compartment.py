"""Tests for two-compartment PK model (Phase 61)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.core.two_compartment import (
    TwoCompartmentResult,
    _alpha_beta,
    fit_biexponential,
    simulate_two_compartment,
)


class TestAlphaBeta:
    def test_alpha_greater_than_beta(self):
        alpha, beta = _alpha_beta(k10=0.5, k12=0.3, k21=0.2)
        assert alpha > beta

    def test_one_compartment_limit(self):
        alpha, beta = _alpha_beta(k10=0.5, k12=0.0, k21=0.0)
        assert abs(alpha - 0.5) < 1e-6

    def test_positive_rates(self):
        alpha, beta = _alpha_beta(k10=1.0, k12=0.5, k21=0.3)
        assert alpha > 0
        assert beta > 0


class TestSimulateTwoCompartmentIV:
    def _sim(self, **kw):
        defaults = dict(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_central_L=10.0,
            vd_peripheral_L=20.0,
            q_L_per_h=3.0,
            route="iv",
            t_end_h=24.0,
            dt_h=0.05,
        )
        defaults.update(kw)
        return simulate_two_compartment(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._sim(), TwoCompartmentResult)

    def test_iv_cmax_at_t0(self):
        r = self._sim()
        assert r.tmax_h < 1.0

    def test_auc_0_inf_geq_auc_0_t(self):
        r = self._sim()
        assert r.auc_0_inf >= r.auc_0_t - 1e-6

    def test_thalf_alpha_less_than_thalf_beta(self):
        r = self._sim()
        assert r.thalf_alpha_h < r.thalf_beta_h

    def test_vd_ss_equals_central_plus_peripheral(self):
        r = self._sim()
        assert abs(r.vd_ss_L - (r.vd_central_L + r.vd_peripheral_L)) < 1e-6

    def test_vd_ss_value(self):
        r = self._sim()
        assert abs(r.vd_ss_L - 30.0) < 1e-6

    def test_cl_preserved(self):
        r = self._sim(cl_L_per_h=5.0)
        assert abs(r.cl_L_per_h - 5.0) < 1e-6

    def test_dose_scales_cmax(self):
        r1 = self._sim(dose_mg=100.0)
        r2 = self._sim(dose_mg=200.0)
        assert abs(r2.cmax / r1.cmax - 2.0) < 0.05

    def test_dose_scales_auc(self):
        r1 = self._sim(dose_mg=100.0)
        r2 = self._sim(dose_mg=200.0)
        assert abs(r2.auc_0_t / r1.auc_0_t - 2.0) < 0.05

    def test_concentration_decreasing_at_end(self):
        r = self._sim()
        cp = r.cp_central
        assert cp[-1] < cp[0]

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            self._sim(dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            self._sim(cl_L_per_h=0.0)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValueError):
            self._sim(vd_central_L=0.0)

    def test_negative_q_raises(self):
        with pytest.raises(ValueError):
            self._sim(q_L_per_h=-1.0)


class TestSimulateTwoCompartmentOral:
    def _sim_oral(self, **kw):
        defaults = dict(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_central_L=10.0,
            vd_peripheral_L=20.0,
            q_L_per_h=3.0,
            ka_per_h=1.5,
            route="oral",
            t_end_h=24.0,
            dt_h=0.05,
            f=1.0,
        )
        defaults.update(kw)
        return simulate_two_compartment(**defaults)

    def test_oral_tmax_after_t0(self):
        r = self._sim_oral()
        assert r.tmax_h > 0.0

    def test_oral_cmax_positive(self):
        r = self._sim_oral()
        assert r.cmax > 0.0

    def test_oral_bioavailability_reduces_auc(self):
        r_f1 = self._sim_oral(f=1.0)
        r_f05 = self._sim_oral(f=0.5)
        assert r_f05.auc_0_t < r_f1.auc_0_t * 0.9


class TestFitBiexponential:
    def _generate_biexp(self, A=10.0, alpha=1.0, B=5.0, beta=0.1, n=30):
        t = np.linspace(0.1, 24.0, n)
        c = A * np.exp(-alpha * t) + B * np.exp(-beta * t)
        return t.tolist(), c.tolist()

    def test_returns_dict_with_keys(self):
        t, c = self._generate_biexp()
        result = fit_biexponential(t, c)
        for key in ("A", "alpha", "B", "beta", "thalf_alpha_h", "thalf_beta_h"):
            assert key in result

    def test_beta_less_than_alpha(self):
        t, c = self._generate_biexp(alpha=1.5, beta=0.1)
        result = fit_biexponential(t, c)
        assert result["alpha"] > result["beta"]

    def test_thalf_beta_reasonable(self):
        t, c = self._generate_biexp(beta=0.1)
        result = fit_biexponential(t, c)
        assert 1.0 < result["thalf_beta_h"] < 50.0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="4"):
            fit_biexponential([1.0, 2.0, 3.0], [10.0, 5.0, 2.0])
