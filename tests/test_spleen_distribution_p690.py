"""Tests for Phase 690 — Spleen Drug Distribution."""

import pytest

from omega_pbpk.core.spleen_distribution_p690 import (
    SpleenDistributionResult,
    simulate_spleen_distribution,
)

BASE = dict(
    drug_name="TestDrug", dose_mg=100.0, logP=2.0, mw_Da=350.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_spleen_distribution(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, SpleenDistributionResult)

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

    def test_c_spleen_is_list(self):
        r = _run()
        assert isinstance(r.c_spleen_mg_g, list)

    def test_kp_is_float(self):
        r = _run()
        assert isinstance(r.kp_spleen, float)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_spleen_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_spleen_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_spleen_positive(self):
        r = _run()
        assert r.cmax_spleen_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_spleen_positive(self):
        r = _run()
        assert r.auc_spleen_mg_h_per_g > 0

    def test_lists_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_spleen_mg_g)


class TestPartitioning:
    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=4.0)
        assert r_high.kp_spleen > r_low.kp_spleen

    def test_kp_in_range(self):
        r = _run()
        assert 0.2 <= r.kp_spleen <= 12.0

    def test_kp_lower_bound(self):
        r = _run(logP=-5.0, mw_Da=2000.0)
        assert r.kp_spleen >= 0.2

    def test_kp_upper_bound(self):
        r = _run(logP=6.0, mw_Da=100.0)
        assert r.kp_spleen <= 12.0


class TestPKProperties:
    def test_t_half_spleen_positive(self):
        r = _run()
        assert r.t_half_spleen_h > 0

    def test_spleen_to_plasma_ratio_positive(self):
        r = _run()
        assert r.spleen_to_plasma_ratio > 0

    def test_red_pulp_fraction(self):
        r = _run()
        assert r.red_pulp_fraction == 0.75

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

    def test_zero_spleen_weight(self):
        with pytest.raises(ValueError):
            _run(spleen_weight_g=0)
