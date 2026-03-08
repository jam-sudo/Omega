"""Tests for Phase 1113 — Brain microdialysis PK model."""

import pytest

from omega_pbpk.core.brain_microdialysis_pk import (
    BrainMicrodialysisResult,
    simulate_brain_microdialysis,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        ps_mL_per_min=1.0,
        kp_uu=0.3,
        fu_plasma=0.1,
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_brain_microdialysis(**params)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


class TestResultType:
    def test_returns_correct_type(self):
        res = default_result()
        assert isinstance(res, BrainMicrodialysisResult)

    def test_drug_name_preserved(self):
        res = simulate_brain_microdialysis("Morphine", 5.0)
        assert res.drug_name == "Morphine"

    def test_dose_preserved(self):
        res = default_result(dose_mg=20.0)
        assert res.dose_mg == 20.0

    def test_kp_uu_preserved(self):
        res = default_result(kp_uu=0.5)
        assert res.kp_uu == 0.5

    def test_times_list_nonempty(self):
        res = default_result()
        assert len(res.times_h) > 0

    def test_times_starts_at_zero(self):
        res = default_result()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_times_ends_near_t_end(self):
        res = default_result(t_end_h=8.0)
        assert res.times_h[-1] == pytest.approx(8.0, abs=0.1)

    def test_c_plasma_length_matches_times(self):
        res = default_result()
        assert len(res.c_plasma_mg_L) == len(res.times_h)

    def test_c_ecf_length_matches_times(self):
        res = default_result()
        assert len(res.c_ecf_mg_L) == len(res.times_h)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_plasma_at_t0_equals_dose_over_vd(self):
        dose = 10.0
        vd = 10.0
        res = simulate_brain_microdialysis("Drug", dose, vd_plasma_L=vd)
        assert res.c_plasma_mg_L[0] == pytest.approx(dose / vd, rel=1e-6)

    def test_c_ecf_at_t0_is_zero(self):
        res = default_result()
        assert res.c_ecf_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_dose_scaling(self):
        res1 = simulate_brain_microdialysis("D", 10.0, vd_plasma_L=10.0)
        res2 = simulate_brain_microdialysis("D", 20.0, vd_plasma_L=10.0)
        assert res2.c_plasma_mg_L[0] == pytest.approx(2 * res1.c_plasma_mg_L[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Dynamic behaviour tests
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_c_plasma_declines_monotonically(self):
        res = default_result()
        for i in range(1, len(res.c_plasma_mg_L)):
            assert res.c_plasma_mg_L[i] <= res.c_plasma_mg_L[i - 1] + 1e-12

    def test_c_ecf_rises_then_falls(self):
        res = default_result()
        # ECF should have a tmax that is not at t=0 and not at t_end
        assert res.tmax_ecf_h > 0.0
        assert res.tmax_ecf_h < res.times_h[-1]

    def test_c_ecf_rises_initially(self):
        res = default_result()
        # ECF at step 5 should be higher than at step 0
        assert res.c_ecf_mg_L[5] > res.c_ecf_mg_L[0]

    def test_c_ecf_falls_at_end(self):
        res = default_result()
        # Last ECF value should be less than max
        assert res.c_ecf_mg_L[-1] < res.cmax_ecf_mg_L


# ---------------------------------------------------------------------------
# AUC and ratio tests
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_ecf_positive(self):
        res = default_result()
        assert res.auc_ecf_mg_h_per_L > 0

    def test_auc_plasma_positive(self):
        res = default_result()
        assert res.auc_plasma_mg_h_per_L > 0

    def test_ecf_to_plasma_auc_ratio_positive(self):
        res = default_result()
        assert res.ecf_to_plasma_auc_ratio > 0

    def test_ecf_to_plasma_ratio_consistent(self):
        res = default_result()
        expected = res.auc_ecf_mg_h_per_L / res.auc_plasma_mg_h_per_L
        assert res.ecf_to_plasma_auc_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Parameter sensitivity tests
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    def test_higher_ps_increases_ecf_auc(self):
        res_low = simulate_brain_microdialysis("D", 10.0, ps_mL_per_min=0.5)
        res_high = simulate_brain_microdialysis("D", 10.0, ps_mL_per_min=5.0)
        assert res_high.auc_ecf_mg_h_per_L > res_low.auc_ecf_mg_h_per_L

    def test_higher_kp_uu_increases_ecf_auc(self):
        res_low = simulate_brain_microdialysis("D", 10.0, kp_uu=0.1)
        res_high = simulate_brain_microdialysis("D", 10.0, kp_uu=0.8)
        assert res_high.auc_ecf_mg_h_per_L > res_low.auc_ecf_mg_h_per_L

    def test_zero_ps_gives_zero_ecf(self):
        res = simulate_brain_microdialysis("D", 10.0, ps_mL_per_min=0.0)
        assert res.auc_ecf_mg_h_per_L == pytest.approx(0.0, abs=1e-9)
        assert res.cmax_ecf_mg_L == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Penetration class tests
# ---------------------------------------------------------------------------


class TestPenetrationClass:
    def test_poor_penetration_class(self):
        # Very low PS → low ratio → "poor"
        res = simulate_brain_microdialysis("D", 10.0, ps_mL_per_min=0.01, kp_uu=0.05)
        assert res.brain_penetration_class == "poor"

    def test_good_penetration_class(self):
        # High PS, high kp_uu → high ratio → "good"
        res = simulate_brain_microdialysis(
            "D",
            10.0,
            ps_mL_per_min=20.0,
            kp_uu=1.5,
            fu_plasma=0.8,
            cl_sys_L_per_h=2.0,
            vd_plasma_L=5.0,
        )
        assert res.brain_penetration_class == "good"

    def test_penetration_class_consistent_with_ratio(self):
        res = default_result()
        ratio = res.ecf_to_plasma_auc_ratio
        if ratio < 0.1:
            assert res.brain_penetration_class == "poor"
        elif ratio <= 0.3:
            assert res.brain_penetration_class == "moderate"
        else:
            assert res.brain_penetration_class == "good"


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_nonempty(self):
        res = default_result()
        assert len(res.notes) > 0

    def test_notes_contains_kp_uu(self):
        res = default_result(kp_uu=0.42)
        assert "0.42" in res.notes or "Kp_uu=0.42" in res.notes


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_brain_microdialysis("D", 0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_brain_microdialysis("D", -5.0)

    def test_cl_sys_zero_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_brain_microdialysis("D", 10.0, cl_sys_L_per_h=0.0)

    def test_vd_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="vd_plasma"):
            simulate_brain_microdialysis("D", 10.0, vd_plasma_L=0.0)

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_brain_microdialysis("D", 10.0, fu_plasma=0.0)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_brain_microdialysis("D", 10.0, fu_plasma=1.5)

    def test_ps_negative_raises(self):
        with pytest.raises(ValueError, match="ps_mL_per_min"):
            simulate_brain_microdialysis("D", 10.0, ps_mL_per_min=-1.0)

    def test_kp_uu_zero_raises(self):
        with pytest.raises(ValueError, match="kp_uu"):
            simulate_brain_microdialysis("D", 10.0, kp_uu=0.0)
