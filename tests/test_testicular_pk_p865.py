"""Tests for Phase 865 — Testicular Drug Distribution PK."""

import pytest

from omega_pbpk.core.testicular_pk_p865 import TesticularPKResult, simulate_testicular_pk

# Default parameters for convenience
DEFAULTS = dict(
    drug_name="TestDrug", dose_mg=100.0, logP=2.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_testicular_pk(**params)


class TestReturnType:
    def test_returns_testicular_pk_result(self):
        r = _run()
        assert isinstance(r, TesticularPKResult)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Cisplatin")
        assert r.drug_name == "Cisplatin"

    def test_dose_preserved(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0


class TestListOutputs:
    def test_times_non_empty(self):
        r = _run()
        assert len(r.times_h) > 0

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_testis_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_testis_mg_g)

    def test_lists_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_testis_mg_g)


class TestCmaxAUC:
    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_testis_positive(self):
        r = _run()
        assert r.cmax_testis_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_testis_positive(self):
        r = _run()
        assert r.auc_testis_mg_h_per_g > 0


class TestKpTestis:
    def test_kp_in_range(self):
        r = _run()
        assert 0.05 <= r.kp_testis <= 5.0

    def test_higher_logp_higher_kp(self):
        r_low = _run(logP=0.0)
        r_high = _run(logP=3.0)
        assert r_high.kp_testis > r_low.kp_testis

    def test_kp_lower_bound(self):
        r = _run(logP=-3.0, mw_Da=800.0)
        assert r.kp_testis >= 0.05

    def test_kp_upper_bound(self):
        r = _run(logP=5.0, mw_Da=100.0)
        assert r.kp_testis <= 5.0


class TestBTBPenetration:
    def test_btb_penetration_valid_string(self):
        r = _run()
        assert r.btb_penetration in ("high", "moderate", "low")

    def test_high_mw_low_penetration(self):
        r = _run(logP=0.5, mw_Da=900.0)
        assert r.btb_penetration == "low"

    def test_high_logp_better_penetration(self):
        r_low = _run(logP=-1.0, mw_Da=300.0)
        r_high = _run(logP=4.0, mw_Da=300.0)
        # High logP should give at least as good penetration
        order = {"low": 0, "moderate": 1, "high": 2}
        assert order[r_high.btb_penetration] >= order[r_low.btb_penetration]


class TestDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestTestisToPlasmaRatio:
    def test_ratio_positive(self):
        r = _run()
        assert r.testis_to_plasma_ratio > 0

    def test_t_half_testis_positive(self):
        r = _run()
        assert r.t_half_testis_h > 0


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0.0)

    def test_negative_clearance_raises(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=-1.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0.0)

    def test_zero_testis_weight_raises(self):
        with pytest.raises(ValueError):
            _run(testis_weight_g=0.0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
