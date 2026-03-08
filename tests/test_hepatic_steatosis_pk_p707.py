"""Tests for Phase 707 — Hepatic Steatosis PK Impact."""

import pytest

from omega_pbpk.clinical.hepatic_steatosis_pk_p707 import (
    HepaticSteatosisPKResult,
    simulate_hepatic_steatosis_pk,
)


def _default():
    return simulate_hepatic_steatosis_pk(
        drug_name="TestDrug", dose_mg=100.0, logP=2.5, cl_sys_L_per_h=5.0, vd_sys_L=50.0
    )


class TestReturnType:
    def test_returns_result_type(self):
        r = _default()
        assert isinstance(r, HepaticSteatosisPKResult)

    def test_drug_name(self):
        r = _default()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _default()
        assert r.dose_mg == 100.0

    def test_steatosis_grade(self):
        r = _default()
        assert r.steatosis_grade == "moderate"


class TestListLengths:
    def test_all_lists_same_length(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.c_plasma_normal_mg_L) == n
        assert len(r.c_plasma_steatosis_mg_L) == n

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == 0.0

    def test_times_monotonic(self):
        r = _default()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] > r.times_h[i - 1]


class TestConcentrations:
    def test_normal_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_normal_mg_L)

    def test_steatosis_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_steatosis_mg_L)

    def test_cmax_normal_positive(self):
        r = _default()
        assert r.cmax_normal_mg_L > 0

    def test_cmax_steatosis_positive(self):
        r = _default()
        assert r.cmax_steatosis_mg_L > 0


class TestSteatosisEffect:
    def test_steatosis_auc_ge_normal(self):
        r = _default()
        assert r.auc_steatosis_mg_h_per_L >= r.auc_normal_mg_h_per_L

    def test_severe_auc_gt_mild(self):
        mild = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="mild")
        severe = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="severe")
        assert severe.auc_steatosis_mg_h_per_L > mild.auc_steatosis_mg_h_per_L

    def test_cl_ratio_lt_one(self):
        r = _default()
        assert r.cl_ratio < 1.0

    def test_cl_ratio_severe_lt_moderate_lt_mild(self):
        mild = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="mild")
        mod = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="moderate")
        sev = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="severe")
        assert sev.cl_ratio < mod.cl_ratio < mild.cl_ratio

    def test_vd_ratio_ge_one_lipophilic(self):
        r = _default()
        assert r.vd_ratio >= 1.0

    def test_vd_ratio_one_for_hydrophilic(self):
        r = simulate_hepatic_steatosis_pk("D", 100.0, logP=-1.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0)
        assert r.vd_ratio == pytest.approx(1.0)


class TestHepaticFatPct:
    def test_mild(self):
        r = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="mild")
        assert r.hepatic_fat_pct == 10.0

    def test_moderate(self):
        r = _default()
        assert r.hepatic_fat_pct == 30.0

    def test_severe(self):
        r = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="severe")
        assert r.hepatic_fat_pct == 60.0


class TestDoseAdjustment:
    def test_factor_in_range(self):
        r = _default()
        assert 0.5 <= r.dose_adjustment_factor <= 1.0

    def test_factor_le_one(self):
        r = _default()
        assert r.dose_adjustment_factor <= 1.0


class TestDoseLinearity:
    def test_double_dose_double_cmax_normal(self):
        r1 = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0)
        r2 = simulate_hepatic_steatosis_pk("D", 200.0, 2.5, 5.0, 50.0)
        assert r2.cmax_normal_mg_L == pytest.approx(2 * r1.cmax_normal_mg_L, rel=0.05)

    def test_double_dose_double_cmax_steatosis(self):
        r1 = simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0)
        r2 = simulate_hepatic_steatosis_pk("D", 200.0, 2.5, 5.0, 50.0)
        assert r2.cmax_steatosis_mg_L == pytest.approx(2 * r1.cmax_steatosis_mg_L, rel=0.05)


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            simulate_hepatic_steatosis_pk("D", 0.0, 2.5, 5.0, 50.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_hepatic_steatosis_pk("D", -10.0, 2.5, 5.0, 50.0)

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 0.0, 50.0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 0.0)

    def test_invalid_grade(self):
        with pytest.raises(ValueError):
            simulate_hepatic_steatosis_pk("D", 100.0, 2.5, 5.0, 50.0, steatosis_grade="extreme")


class TestNotes:
    def test_notes_is_string(self):
        r = _default()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _default()
        assert len(r.notes) > 0
