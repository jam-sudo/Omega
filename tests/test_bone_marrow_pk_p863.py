"""Tests for Phase 863 — Bone Marrow Drug Distribution."""

import pytest

from omega_pbpk.core.bone_marrow_pk_p863 import BoneMarrowPKResult, simulate_bone_marrow_pk

BASE = dict(
    drug_name="TestDrug", dose_mg=100.0, logP=2.0, mw_Da=350.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_bone_marrow_pk(**params)


class TestBoneMarrowPKBasic:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, BoneMarrowPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_non_empty(self):
        r = _run()
        assert len(r.times_h) > 0

    def test_times_start_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_marrow_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_marrow_mg_g)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_marrow_positive(self):
        r = _run()
        assert r.cmax_marrow_mg_g > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_marrow_positive(self):
        r = _run()
        assert r.auc_marrow_mg_h_per_g > 0

    def test_kp_marrow_positive(self):
        r = _run()
        assert r.kp_marrow > 0

    def test_t_half_marrow_positive(self):
        r = _run()
        assert r.t_half_marrow_h > 0

    def test_myelotoxicity_index_positive(self):
        r = _run()
        assert r.myelotoxicity_index > 0

    def test_marrow_to_plasma_ratio_positive(self):
        r = _run()
        assert r.marrow_to_plasma_ratio > 0

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0


class TestBoneMarrowPKDoseLinearity:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.8 < ratio < 2.2

    def test_double_dose_doubles_cmax_marrow(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_marrow_mg_g / r1.cmax_marrow_mg_g
        assert 1.8 < ratio < 2.2


class TestBoneMarrowPKPartitioning:
    def test_kp_bounded(self):
        r = _run()
        assert 0.2 <= r.kp_marrow <= 12.0

    def test_high_logP_higher_kp(self):
        r_low = _run(logP=0.0)
        r_high = _run(logP=4.0)
        assert r_high.kp_marrow > r_low.kp_marrow

    def test_high_mw_lower_kp(self):
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=800.0)
        assert r_low.kp_marrow >= r_high.kp_marrow

    def test_kp_min_clamp(self):
        r = _run(logP=-3.0, mw_Da=1500.0)
        assert r.kp_marrow >= 0.2

    def test_kp_max_clamp(self):
        r = _run(logP=10.0, mw_Da=100.0)
        assert r.kp_marrow <= 12.0


class TestBoneMarrowPKValidation:
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

    def test_zero_marrow_weight(self):
        with pytest.raises(ValueError):
            _run(marrow_weight_g=0)


class TestBoneMarrowPKListLengths:
    def test_list_lengths_match(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_marrow_mg_g)
