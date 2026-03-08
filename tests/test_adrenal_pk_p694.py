"""Tests for Phase 694 — Adrenal Gland Drug Accumulation."""

import math

import pytest

from omega_pbpk.core.adrenal_pk_p694 import (
    AdrenalPKResult,
    simulate_adrenal_pk,
)

# Default test parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_adrenal_pk(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, AdrenalPKResult)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Hydrocortisone")
        assert r.drug_name == "Hydrocortisone"

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

    def test_adrenal_conc_is_list(self):
        r = _run()
        assert isinstance(r.c_adrenal_mg_g, list)
        assert len(r.c_adrenal_mg_g) == len(r.times_h)


class TestConcentrations:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_adrenal_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_adrenal_mg_g)

    def test_plasma_starts_at_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0

    def test_adrenal_starts_at_zero(self):
        r = _run()
        assert r.c_adrenal_mg_g[0] == 0.0


class TestCmaxAndAUC:
    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_adrenal_positive(self):
        r = _run()
        assert r.cmax_adrenal_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_adrenal_positive(self):
        r = _run()
        assert r.auc_adrenal_mg_h_per_g > 0


class TestKpAndPartitioning:
    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        assert r_high.kp_adrenal > r_low.kp_adrenal

    def test_kp_within_bounds(self):
        r = _run()
        assert 0.3 <= r.kp_adrenal <= 20.0

    def test_kp_lower_bound(self):
        r = _run(logP=-5.0, mw_Da=1000.0)
        assert r.kp_adrenal >= 0.3

    def test_kp_upper_bound(self):
        r = _run(logP=10.0, mw_Da=100.0)
        assert r.kp_adrenal <= 20.0


class TestCortexVsMedulla:
    def test_high_logP_cortex(self):
        r = _run(logP=3.0)
        assert r.cortex_vs_medulla == "cortex"

    def test_low_logP_medulla(self):
        r = _run(logP=0.5)
        assert r.cortex_vs_medulla == "medulla"

    def test_boundary_logP_medulla(self):
        r = _run(logP=1.5)
        assert r.cortex_vs_medulla == "medulla"


class TestHalfLife:
    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_adrenal_h > 0

    def test_t_half_is_finite(self):
        r = _run()
        assert math.isfinite(r.t_half_adrenal_h)


class TestRatios:
    def test_adrenal_to_plasma_ratio_positive(self):
        r = _run()
        assert r.adrenal_to_plasma_ratio > 0


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_adrenal_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_adrenal_mg_g / r1.cmax_adrenal_mg_g
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

    def test_negative_adrenal_weight_raises(self):
        with pytest.raises(ValueError, match="adrenal_weight"):
            _run(adrenal_weight_g=-1.0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
