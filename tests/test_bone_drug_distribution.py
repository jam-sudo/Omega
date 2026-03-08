"""Tests for Phase 501: Bone Drug Distribution PK Model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bone_drug_distribution import (
    BoneDrugDistributionResult,
    compare_bone_affinity,
    simulate_bone_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    cl_L_per_h=2.0,
    vd_plasma_L=5.0,
    k_bone_uptake=0.05,
    k_bone_release=0.001,
)


def default_sim(**kwargs):
    params = {**DEFAULT_PARAMS, **kwargs}
    return simulate_bone_pk(**params)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        res = default_sim()
        assert isinstance(res, BoneDrugDistributionResult)

    def test_has_drug_name(self):
        res = default_sim(drug_name="Alendronate")
        assert res.drug_name == "Alendronate"

    def test_has_dose_mg(self):
        res = default_sim(dose_mg=20.0)
        assert res.dose_mg == 20.0

    def test_has_logP(self):
        res = default_sim(logP=1.5)
        assert res.logP == 1.5

    def test_has_times_h_list(self):
        res = default_sim()
        assert isinstance(res.times_h, list)
        assert len(res.times_h) > 1

    def test_has_c_plasma_mg_L_list(self):
        res = default_sim()
        assert isinstance(res.c_plasma_mg_L, list)
        assert len(res.c_plasma_mg_L) == len(res.times_h)

    def test_has_c_bone_mg_L_list(self):
        res = default_sim()
        assert isinstance(res.c_bone_mg_L, list)
        assert len(res.c_bone_mg_L) == len(res.times_h)

    def test_has_all_scalar_fields(self):
        res = default_sim()
        for field_name in [
            "cmax_plasma",
            "tmax_plasma_h",
            "auc_plasma",
            "cmax_bone",
            "tmax_bone_h",
            "auc_bone",
            "kp_bone",
            "bone_retention_pct",
            "t_half_plasma_h",
        ]:
            val = getattr(res, field_name)
            assert isinstance(val, float), f"{field_name} should be float"

    def test_has_notes_str(self):
        res = default_sim()
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_plasma_starts_at_dose_over_vd(self):
        res = default_sim(dose_mg=10.0, vd_plasma_L=5.0)
        # c_plasma[0] = dose/Vd_plasma = 10/5 = 2.0 mg/L
        assert abs(res.c_plasma_mg_L[0] - 2.0) < 1e-9

    def test_bone_starts_at_zero(self):
        res = default_sim()
        assert res.c_bone_mg_L[0] == 0.0

    def test_time_starts_at_zero(self):
        res = default_sim()
        assert res.times_h[0] == 0.0

    def test_cmax_plasma_equals_initial_concentration(self):
        # For IV bolus plasma concentration starts at max and declines
        res = default_sim()
        assert res.cmax_plasma == pytest.approx(res.c_plasma_mg_L[0], rel=1e-6)
        assert res.tmax_plasma_h == pytest.approx(0.0, abs=0.15)


# ---------------------------------------------------------------------------
# Bone compartment dynamics
# ---------------------------------------------------------------------------


class TestBoneDynamics:
    def test_bone_rises_from_zero(self):
        res = default_sim()
        # Bone should rise from 0
        assert res.c_bone_mg_L[1] > 0.0

    def test_bone_has_positive_cmax(self):
        res = default_sim()
        assert res.cmax_bone > 0.0

    def test_bone_tmax_after_plasma_tmax(self):
        res = default_sim()
        assert res.tmax_bone_h > res.tmax_plasma_h

    def test_slow_release_drug_elevated_bone_at_end(self):
        # Very slow release: bisphosphonate-like
        res = simulate_bone_pk(
            drug_name="Bisphosphonate",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            k_bone_uptake=0.1,
            k_bone_release=0.0001,
            t_end_h=72.0,
        )
        # Bone retention should be substantial
        assert res.bone_retention_pct > 10.0

    def test_fast_release_lower_bone_retention(self):
        slow = default_sim(k_bone_release=0.001)
        fast = default_sim(k_bone_release=0.1)
        assert slow.bone_retention_pct > fast.bone_retention_pct


# ---------------------------------------------------------------------------
# Higher uptake → higher bone AUC
# ---------------------------------------------------------------------------


class TestUptakeEffect:
    def test_higher_uptake_higher_bone_auc(self):
        low = default_sim(k_bone_uptake=0.02)
        high = default_sim(k_bone_uptake=0.10)
        assert high.auc_bone > low.auc_bone

    def test_higher_release_lower_bone_auc(self):
        slow = default_sim(k_bone_release=0.001)
        fast = default_sim(k_bone_release=0.05)
        assert slow.auc_bone > fast.auc_bone


# ---------------------------------------------------------------------------
# KP bone
# ---------------------------------------------------------------------------


class TestKpBone:
    def test_kp_bone_equals_ratio(self):
        res = simulate_bone_pk(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=1.0,
            vd_plasma_L=5.0,
            k_bone_uptake=0.05,
            k_bone_release=0.002,
        )
        expected = 0.05 / 0.002
        assert abs(res.kp_bone - expected) < 1e-9

    def test_higher_uptake_higher_kp(self):
        low = default_sim(k_bone_uptake=0.02)
        high = default_sim(k_bone_uptake=0.10)
        assert high.kp_bone > low.kp_bone


# ---------------------------------------------------------------------------
# Bone retention
# ---------------------------------------------------------------------------


class TestBoneRetention:
    def test_retention_between_0_and_100(self):
        res = default_sim()
        assert 0.0 <= res.bone_retention_pct <= 100.0

    def test_slow_release_high_retention(self):
        res = simulate_bone_pk(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            k_bone_uptake=0.2,
            k_bone_release=0.0001,
            t_end_h=72.0,
        )
        assert res.bone_retention_pct > 20.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_double_plasma_auc(self):
        res1 = default_sim(dose_mg=10.0)
        res2 = default_sim(dose_mg=20.0)
        ratio = res2.auc_plasma / res1.auc_plasma
        assert abs(ratio - 2.0) < 0.05

    def test_double_dose_double_bone_auc(self):
        res1 = default_sim(dose_mg=10.0)
        res2 = default_sim(dose_mg=20.0)
        ratio = res2.auc_bone / res1.auc_bone
        assert abs(ratio - 2.0) < 0.05

    def test_double_dose_double_cmax_plasma(self):
        res1 = default_sim(dose_mg=10.0)
        res2 = default_sim(dose_mg=20.0)
        ratio = res2.cmax_plasma / res1.cmax_plasma
        assert abs(ratio - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bone_pk("Drug", 0.0, 2.0, 5.0, 0.05, 0.001)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bone_pk("Drug", -1.0, 2.0, 5.0, 0.05, 0.001)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_bone_pk("Drug", 10.0, 0.0, 5.0, 0.05, 0.001)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            simulate_bone_pk("Drug", 10.0, 2.0, 0.0, 0.05, 0.001)

    def test_zero_uptake_raises(self):
        with pytest.raises(ValueError, match="k_bone_uptake"):
            simulate_bone_pk("Drug", 10.0, 2.0, 5.0, 0.0, 0.001)

    def test_zero_release_raises(self):
        with pytest.raises(ValueError, match="k_bone_release"):
            simulate_bone_pk("Drug", 10.0, 2.0, 5.0, 0.05, 0.0)


# ---------------------------------------------------------------------------
# Compare function
# ---------------------------------------------------------------------------


class TestCompareFunction:
    def test_compare_returns_list(self):
        result = compare_bone_affinity(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            uptake_rates=[0.01, 0.05, 0.10],
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_compare_sorted_by_auc_bone_descending(self):
        result = compare_bone_affinity(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            uptake_rates=[0.01, 0.05, 0.10],
        )
        aucs = [r.auc_bone for r in result]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_returns_correct_type_elements(self):
        result = compare_bone_affinity(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            uptake_rates=[0.05],
        )
        assert isinstance(result[0], BoneDrugDistributionResult)

    def test_compare_single_rate(self):
        result = compare_bone_affinity(
            drug_name="Drug",
            dose_mg=10.0,
            cl_L_per_h=2.0,
            vd_plasma_L=5.0,
            uptake_rates=[0.05],
        )
        assert len(result) == 1

    def test_t_half_plasma_positive(self):
        res = default_sim()
        assert res.t_half_plasma_h > 0.0

    def test_auc_plasma_positive(self):
        res = default_sim()
        assert res.auc_plasma > 0.0

    def test_auc_bone_positive(self):
        res = default_sim()
        assert res.auc_bone > 0.0
