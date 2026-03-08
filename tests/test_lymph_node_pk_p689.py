"""Tests for Phase 689 — Lymph Node Drug Concentration."""

import pytest

from omega_pbpk.core.lymph_node_pk_p689 import LymphNodePKResult, simulate_lymph_node_pk

BASE = dict(
    drug_name="TestDrug", dose_mg=100.0, logP=2.0, mw_Da=350.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_lymph_node_pk(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, LymphNodePKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_c_lymph_node_is_list(self):
        r = _run()
        assert isinstance(r.c_lymph_node_mg_g, list)

    def test_kp_is_float(self):
        r = _run()
        assert isinstance(r.kp_lymph_node, float)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_lymph_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_lymph_node_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_lymph_positive(self):
        r = _run()
        assert r.cmax_lymph_node_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_lymph_positive(self):
        r = _run()
        assert r.auc_lymph_node_mg_h_per_g > 0

    def test_lists_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_lymph_node_mg_g)


class TestPartitioning:
    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=4.0)
        assert r_high.kp_lymph_node > r_low.kp_lymph_node

    def test_kp_in_range(self):
        r = _run()
        assert 0.2 <= r.kp_lymph_node <= 15.0

    def test_kp_lower_bound(self):
        r = _run(logP=-3.0, mw_Da=2000.0)
        assert r.kp_lymph_node >= 0.2

    def test_kp_upper_bound(self):
        r = _run(logP=6.0, mw_Da=100.0)
        assert r.kp_lymph_node <= 15.0


class TestPKProperties:
    def test_t_half_lymph_positive(self):
        r = _run()
        assert r.t_half_lymph_h > 0

    def test_lymph_to_plasma_ratio_positive(self):
        r = _run()
        assert r.lymph_to_plasma_ratio > 0

    def test_dose_linearity_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.15

    def test_dose_linearity_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert abs(ratio - 2.0) < 0.15


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

    def test_zero_lymph_mass(self):
        with pytest.raises(ValueError):
            _run(lymph_node_mass_g=0)
