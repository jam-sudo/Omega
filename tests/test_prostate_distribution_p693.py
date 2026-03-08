"""Tests for Phase 693 — Prostate Drug Distribution."""

import math

import pytest

from omega_pbpk.core.prostate_distribution_p693 import (
    ProstateDistributionResult,
    simulate_prostate_distribution,
)

# Default test parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=350.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_prostate_distribution(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, ProstateDistributionResult)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Tamsulosin")
        assert r.drug_name == "Tamsulosin"

    def test_dose_preserved(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_plasma_conc_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_prostate_conc_is_list(self):
        r = _run()
        assert isinstance(r.c_prostate_mg_g, list)
        assert len(r.c_prostate_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_prostate_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_prostate_mg_g)

    def test_plasma_starts_at_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0

    def test_prostate_starts_at_zero(self):
        r = _run()
        assert r.c_prostate_mg_g[0] == 0.0


class TestCmaxAndAUC:
    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_prostate_positive(self):
        r = _run()
        assert r.cmax_prostate_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_prostate_positive(self):
        r = _run()
        assert r.auc_prostate_mg_h_per_g > 0


class TestKpAndPartitioning:
    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        assert r_high.kp_prostate > r_low.kp_prostate

    def test_kp_within_bounds(self):
        r = _run()
        assert 0.1 <= r.kp_prostate <= 10.0

    def test_kp_lower_bound(self):
        r = _run(logP=-5.0, mw_Da=1000.0)
        assert r.kp_prostate >= 0.1

    def test_kp_upper_bound(self):
        r = _run(logP=10.0, mw_Da=100.0)
        assert r.kp_prostate <= 10.0


class TestHalfLife:
    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_prostate_h > 0

    def test_t_half_is_finite(self):
        r = _run()
        assert math.isfinite(r.t_half_prostate_h)


class TestRatios:
    def test_prostate_to_plasma_ratio_positive(self):
        r = _run()
        assert r.prostate_to_plasma_ratio > 0


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_prostate_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_prostate_mg_g / r1.cmax_prostate_mg_g
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=-1.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=-1.0)

    def test_negative_prostate_weight_raises(self):
        with pytest.raises(ValueError, match="prostate_weight"):
            _run(prostate_weight_g=-1.0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
