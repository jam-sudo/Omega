"""Tests for Phase 687 — Pancreatic Drug Distribution."""

import pytest

from omega_pbpk.core.pancreatic_distribution_p687 import (
    PancreaticDistributionResult,
    simulate_pancreatic_distribution,
)

# Default test parameters
_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_pancreatic_distribution(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, PancreaticDistributionResult)

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

    def test_c_pancreas_is_list(self):
        r = _run()
        assert isinstance(r.c_pancreas_mg_g, list)
        assert len(r.c_pancreas_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_pancreas_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_pancreas_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_pancreas_positive(self):
        r = _run()
        assert r.cmax_pancreas_mg_g > 0


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_pancreas_positive(self):
        r = _run()
        assert r.auc_pancreas_mg_h_per_g > 0


class TestKpPancreas:
    def test_kp_in_range(self):
        r = _run()
        assert 0.1 <= r.kp_pancreas <= 10.0

    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.0)
        r_high = _run(logP=4.0)
        assert r_high.kp_pancreas > r_low.kp_pancreas

    def test_kp_low_bound(self):
        r = _run(logP=-5.0, mw_Da=1000.0)
        assert r.kp_pancreas >= 0.1

    def test_kp_upper_bound(self):
        r = _run(logP=10.0, mw_Da=100.0)
        assert r.kp_pancreas <= 10.0


class TestHalfLife:
    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_pancreas_h > 0


class TestRatios:
    def test_pancreas_to_plasma_ratio_positive(self):
        r = _run()
        assert r.pancreas_to_plasma_ratio > 0


class TestAcinarVsIslet:
    def test_acinar_for_low_kp(self):
        r = _run(logP=-1.0, mw_Da=500.0)
        assert r.acinar_vs_islet == "acinar"

    def test_islet_for_high_kp(self):
        r = _run(logP=5.0, mw_Da=100.0)
        assert r.acinar_vs_islet == "islet"

    def test_valid_values(self):
        r = _run()
        assert r.acinar_vs_islet in ("acinar", "islet")


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_zero_pancreas_weight(self):
        with pytest.raises(ValueError, match="pancreas_weight"):
            _run(pancreas_weight_g=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
