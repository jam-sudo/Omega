"""Tests for Phase 866 — Breast Tissue Drug Distribution PK."""

import pytest

from omega_pbpk.core.breast_tissue_pk_p866 import BreastTissuePKResult, simulate_breast_tissue_pk

DEFAULTS = dict(
    drug_name="TestDrug", dose_mg=100.0, logP=2.0, mw_Da=300.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_breast_tissue_pk(**params)


class TestReturnType:
    def test_returns_breast_tissue_pk_result(self):
        r = _run()
        assert isinstance(r, BreastTissuePKResult)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Tamoxifen")
        assert r.drug_name == "Tamoxifen"

    def test_dose_preserved(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_lactation_preserved(self):
        r = _run(lactation=True)
        assert r.lactation is True


class TestListOutputs:
    def test_times_non_empty(self):
        r = _run()
        assert len(r.times_h) > 0

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_breast_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_breast_mg_g)

    def test_milk_non_negative(self):
        r = _run(lactation=True)
        assert all(c >= 0 for c in r.c_milk_mg_L)

    def test_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_breast_mg_g) == n
        assert len(r.c_milk_mg_L) == n


class TestCmaxAUC:
    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_breast_positive(self):
        r = _run()
        assert r.cmax_breast_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_breast_positive(self):
        r = _run()
        assert r.auc_breast_mg_h_per_g > 0


class TestKpBreast:
    def test_kp_in_range(self):
        r = _run()
        assert 0.2 <= r.kp_breast <= 10.0

    def test_higher_logp_higher_kp(self):
        r_low = _run(logP=-1.0)
        r_high = _run(logP=3.0)
        assert r_high.kp_breast > r_low.kp_breast

    def test_kp_lower_bound(self):
        r = _run(logP=-5.0, mw_Da=800.0)
        assert r.kp_breast >= 0.2

    def test_kp_upper_bound(self):
        r = _run(logP=8.0, mw_Da=100.0)
        assert r.kp_breast <= 10.0


class TestLactation:
    def test_lactation_milk_positive(self):
        r = _run(lactation=True)
        assert r.cmax_milk_mg_L > 0
        assert any(c > 0 for c in r.c_milk_mg_L)

    def test_no_lactation_milk_zero(self):
        r = _run(lactation=False)
        assert all(c == 0 for c in r.c_milk_mg_L)
        assert r.milk_to_plasma_ratio == 0.0

    def test_milk_to_plasma_ratio_positive_when_lactating(self):
        r = _run(lactation=True)
        assert r.milk_to_plasma_ratio > 0

    def test_milk_to_plasma_ratio_zero_when_not_lactating(self):
        r = _run(lactation=False)
        assert r.milk_to_plasma_ratio == 0.0


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

    def test_zero_breast_weight_raises(self):
        with pytest.raises(ValueError):
            _run(breast_weight_g=0.0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0

    def test_t_half_breast_positive(self):
        r = _run()
        assert r.t_half_breast_h > 0
