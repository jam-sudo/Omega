"""Tests for RBC distribution model (phase 656)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.rbc_distribution_p656 import (
    RBCDistributionResult,
    simulate_rbc_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kw):
    defaults = dict(
        drug_name="drugX",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=400.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    defaults.update(kw)
    return simulate_rbc_distribution(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default_result()
        assert isinstance(r, RBCDistributionResult)

    def test_drug_name_preserved(self):
        r = _default_result(drug_name="TestDrug")
        assert r.drug_name == "TestDrug"

    def test_dose_preserved(self):
        r = _default_result(dose_mg=200.0)
        assert r.dose_mg == 200.0


# ---------------------------------------------------------------------------
# List lengths and non-negative
# ---------------------------------------------------------------------------


class TestListProperties:
    def test_list_lengths_equal(self):
        r = _default_result()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_rbc_mg_L) == n
        assert len(r.c_whole_blood_mg_L) == n

    def test_non_negative_plasma(self):
        r = _default_result()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_non_negative_rbc(self):
        r = _default_result()
        assert all(c >= 0 for c in r.c_rbc_mg_L)

    def test_non_negative_blood(self):
        r = _default_result()
        assert all(c >= 0 for c in r.c_whole_blood_mg_L)

    def test_times_monotonic(self):
        r = _default_result()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] >= r.times_h[i - 1]


# ---------------------------------------------------------------------------
# Kp_RBC and Rb/p
# ---------------------------------------------------------------------------


class TestPartitioning:
    def test_higher_logP_higher_kp(self):
        r_low = _default_result(logP=0.5)
        r_high = _default_result(logP=3.0)
        assert r_high.kp_rbc > r_low.kp_rbc

    def test_kp_rbc_in_range(self):
        r = _default_result()
        assert 0.1 <= r.kp_rbc <= 10.0

    def test_rb_p_positive(self):
        r = _default_result()
        assert r.rb_p_ratio > 0

    def test_rb_p_ge_03(self):
        r = _default_result()
        assert r.rb_p_ratio >= 0.3


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _default_result()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_rbc_positive(self):
        r = _default_result()
        assert r.auc_rbc_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# RBC sequestration
# ---------------------------------------------------------------------------


class TestSequestration:
    def test_high_logP_high_sequestration(self):
        r_low = _default_result(logP=0.5)
        r_high = _default_result(logP=3.5)
        assert r_high.rbc_sequestration_pct > r_low.rbc_sequestration_pct

    def test_seq_in_range(self):
        r = _default_result()
        assert 0 <= r.rbc_sequestration_pct <= 100


# ---------------------------------------------------------------------------
# Whole blood
# ---------------------------------------------------------------------------


class TestWholeBlood:
    def test_blood_between_plasma_and_rbc(self):
        r = _default_result()
        for i in range(len(r.times_h)):
            low = min(r.c_plasma_mg_L[i], r.c_rbc_mg_L[i])
            high = max(r.c_plasma_mg_L[i], r.c_rbc_mg_L[i])
            assert r.c_whole_blood_mg_L[i] >= low - 1e-9
            assert r.c_whole_blood_mg_L[i] <= high + 1e-9


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_double_cmax(self):
        r1 = _default_result(dose_mg=100.0)
        r2 = _default_result(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_positive(self):
        r = _default_result()
        assert r.t_half_blood_h > 0

    def test_t_half_matches_kel(self):
        r = _default_result(cl_sys_L_per_h=5.0, vd_sys_L=50.0)
        expected = math.log(2) / (5.0 / 50.0)
        assert abs(r.t_half_blood_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Hematocrit edge case
# ---------------------------------------------------------------------------


class TestHematocritEdge:
    def test_hematocrit_zero_rbp_near_one(self):
        r = _default_result(hematocrit=0.0)
        assert abs(r.rb_p_ratio - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=-10)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_sys_L_per_h=0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(vd_sys_L=0)

    def test_hematocrit_one_raises(self):
        with pytest.raises(ValueError):
            _default_result(hematocrit=1.0)

    def test_hematocrit_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(hematocrit=-0.1)
