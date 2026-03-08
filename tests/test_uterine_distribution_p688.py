"""Tests for Phase 688 — Uterine Drug Distribution."""

import pytest

from omega_pbpk.core.uterine_distribution_p688 import (
    UterineDistributionResult,
    simulate_uterine_distribution,
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
    return simulate_uterine_distribution(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, UterineDistributionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_cycle_phase_default(self):
        r = _run()
        assert r.cycle_phase == "follicular"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_c_uterus_is_list(self):
        r = _run()
        assert isinstance(r.c_uterus_mg_g, list)
        assert len(r.c_uterus_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_uterus_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_uterus_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_uterus_positive(self):
        r = _run()
        assert r.cmax_uterus_mg_g > 0


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_uterus_positive(self):
        r = _run()
        assert r.auc_uterus_mg_h_per_g > 0


class TestCyclePhases:
    def test_pregnant_higher_cmax_than_follicular(self):
        r_fol = _run(cycle_phase="follicular")
        r_preg = _run(cycle_phase="pregnant")
        assert r_preg.cmax_uterus_mg_g > r_fol.cmax_uterus_mg_g

    def test_luteal_higher_kp_than_follicular(self):
        r_fol = _run(cycle_phase="follicular")
        r_lut = _run(cycle_phase="luteal")
        assert r_lut.kp_uterus > r_fol.kp_uterus


class TestKpUterus:
    def test_kp_in_range(self):
        r = _run()
        assert 0.1 <= r.kp_uterus <= 8.0

    def test_kp_low_bound(self):
        r = _run(logP=-5.0, mw_Da=1000.0)
        assert r.kp_uterus >= 0.1

    def test_kp_upper_bound(self):
        r = _run(logP=10.0, mw_Da=100.0)
        assert r.kp_uterus <= 8.0


class TestHalfLife:
    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_uterus_h > 0


class TestRatios:
    def test_uterus_to_plasma_ratio_positive(self):
        r = _run()
        assert r.uterus_to_plasma_ratio > 0


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

    def test_zero_uterus_weight(self):
        with pytest.raises(ValueError, match="uterus_weight"):
            _run(uterus_weight_g=0)

    def test_invalid_cycle_phase(self):
        with pytest.raises(ValueError, match="cycle_phase"):
            _run(cycle_phase="invalid")


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
