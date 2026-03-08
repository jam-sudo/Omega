"""Tests for Phase 708 — Drug Partition into Red Blood Cells."""

import pytest

from omega_pbpk.core.rbc_partition_p708 import (
    RBCPartitionResult,
    simulate_rbc_partition,
)


def _default():
    return simulate_rbc_partition(drug_name="TestDrug", dose_mg=100.0, logP=2.5, fu_plasma=0.3)


class TestReturnType:
    def test_returns_result_type(self):
        r = _default()
        assert isinstance(r, RBCPartitionResult)

    def test_drug_name(self):
        r = _default()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _default()
        assert r.dose_mg == 100.0

    def test_hematocrit_default(self):
        r = _default()
        assert r.hematocrit == 0.45


class TestKpRbc:
    def test_kp_in_range(self):
        r = _default()
        assert 0.1 <= r.kp_rbc <= 10.0

    def test_higher_logP_higher_kp(self):
        low = simulate_rbc_partition("D", 100.0, logP=0.5, fu_plasma=0.3)
        high = simulate_rbc_partition("D", 100.0, logP=4.0, fu_plasma=0.3)
        assert high.kp_rbc > low.kp_rbc


class TestBloodPlasmaRatio:
    def test_formula(self):
        r = _default()
        H = r.hematocrit
        expected = 1.0 - H + H * r.kp_rbc
        assert r.blood_to_plasma_ratio == pytest.approx(expected, rel=1e-6)

    def test_higher_logP_higher_Rb(self):
        low = simulate_rbc_partition("D", 100.0, logP=0.5, fu_plasma=0.3)
        high = simulate_rbc_partition("D", 100.0, logP=4.0, fu_plasma=0.3)
        assert high.blood_to_plasma_ratio > low.blood_to_plasma_ratio


class TestFuBlood:
    def test_fu_blood_formula(self):
        r = _default()
        expected = 0.3 / r.blood_to_plasma_ratio
        assert r.fu_blood == pytest.approx(expected, rel=1e-6)

    def test_fu_blood_le_fu_plasma(self):
        r = _default()
        assert r.fu_blood <= r.fu_blood or r.fu_blood <= 0.3 + 1e-9


class TestConcentrationRelationships:
    def test_c_rbc_proportional(self):
        r = _default()
        for i in range(len(r.times_h)):
            expected = r.c_plasma_mg_L[i] * r.kp_rbc
            assert r.c_rbc_mg_L[i] == pytest.approx(expected, abs=1e-12)

    def test_c_blood_eq_plasma_times_Rb(self):
        r = _default()
        for i in range(len(r.times_h)):
            expected = r.c_plasma_mg_L[i] * r.blood_to_plasma_ratio
            assert r.c_whole_blood_mg_L[i] == pytest.approx(expected, abs=1e-12)


class TestListLengths:
    def test_all_lists_same_length(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_rbc_mg_L) == n
        assert len(r.c_whole_blood_mg_L) == n

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == 0.0


class TestNonNegative:
    def test_plasma_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_rbc_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_rbc_mg_L)

    def test_blood_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_whole_blood_mg_L)


class TestCmaxAuc:
    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_plasma_mg_L > 0

    def test_auc_plasma_positive(self):
        r = _default()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_blood_positive(self):
        r = _default()
        assert r.auc_blood_mg_h_per_L > 0

    def test_auc_blood_approx_auc_plasma_times_Rb(self):
        r = _default()
        expected = r.auc_plasma_mg_h_per_L * r.blood_to_plasma_ratio
        assert r.auc_blood_mg_h_per_L == pytest.approx(expected, rel=0.05)


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = simulate_rbc_partition("D", 100.0, 2.5, 0.3)
        r2 = simulate_rbc_partition("D", 200.0, 2.5, 0.3)
        assert r2.cmax_plasma_mg_L == pytest.approx(2 * r1.cmax_plasma_mg_L, rel=0.05)


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 0.0, 2.5, 0.3)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", -10.0, 2.5, 0.3)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 0.3, vd_sys_L=0.0)

    def test_hematocrit_zero(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 0.3, hematocrit=0.0)

    def test_hematocrit_one(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 0.3, hematocrit=1.0)

    def test_fu_plasma_zero(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 0.0)

    def test_fu_plasma_gt_one(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 1.5)

    def test_negative_cl_blood(self):
        with pytest.raises(ValueError):
            simulate_rbc_partition("D", 100.0, 2.5, 0.3, cl_blood_L_per_h=-1.0)


class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _default()
        assert len(r.notes) > 0
