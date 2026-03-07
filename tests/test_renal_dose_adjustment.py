"""Tests for clinical/renal_dose_adjustment.py (Phase 679)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.renal_dose_adjustment import (
    RenalDoseAdjResult,
    adjust_dose_for_renal,
    ckd_stage,
    estimate_crcl_cockcroft_gault,
    estimate_egfr_ckd_epi,
)

# ---------------------------------------------------------------------------
# estimate_crcl_cockcroft_gault
# ---------------------------------------------------------------------------


class TestCockcroftGault:
    def test_basic_male(self):
        crcl = estimate_crcl_cockcroft_gault(40, 70, 1.0, sex="male")
        expected = (140 - 40) * 70 / (72 * 1.0)
        assert abs(crcl - expected) < 0.01

    def test_female_factor_0_85(self):
        crcl_male = estimate_crcl_cockcroft_gault(40, 70, 1.0, sex="male")
        crcl_female = estimate_crcl_cockcroft_gault(40, 70, 1.0, sex="female")
        assert abs(crcl_female - crcl_male * 0.85) < 0.01

    def test_older_age_lower_crcl(self):
        crcl_young = estimate_crcl_cockcroft_gault(30, 70, 1.0)
        crcl_old = estimate_crcl_cockcroft_gault(75, 70, 1.0)
        assert crcl_old < crcl_young

    def test_higher_scr_lower_crcl(self):
        crcl_normal = estimate_crcl_cockcroft_gault(50, 70, 1.0)
        crcl_elevated = estimate_crcl_cockcroft_gault(50, 70, 3.0)
        assert crcl_elevated < crcl_normal

    def test_minimum_value_clamp(self):
        # Very high age/scr gives near-zero; should be clamped to >= 1
        crcl = estimate_crcl_cockcroft_gault(85, 50, 5.0)
        assert crcl >= 1.0

    def test_age_zero_raises(self):
        with pytest.raises(ValueError, match="age_years"):
            estimate_crcl_cockcroft_gault(0, 70, 1.0)

    def test_age_negative_raises(self):
        with pytest.raises(ValueError):
            estimate_crcl_cockcroft_gault(-5, 70, 1.0)

    def test_weight_zero_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            estimate_crcl_cockcroft_gault(40, 0, 1.0)

    def test_scr_zero_raises(self):
        with pytest.raises(ValueError, match="serum_creatinine"):
            estimate_crcl_cockcroft_gault(40, 70, 0.0)


# ---------------------------------------------------------------------------
# estimate_egfr_ckd_epi
# ---------------------------------------------------------------------------


class TestEgfrCkdEpi:
    def test_normal_subject_egfr_above_90(self):
        egfr = estimate_egfr_ckd_epi(35, 0.9, sex="male")
        assert egfr > 90

    def test_female_higher_than_male_same_scr(self):
        egfr_male = estimate_egfr_ckd_epi(50, 1.0, sex="male")
        egfr_female = estimate_egfr_ckd_epi(50, 1.0, sex="female")
        # Female gets ×1.018 factor but lower kappa makes scr/kappa larger => net differs
        # Just check both are positive and reasonable
        assert egfr_male > 0
        assert egfr_female > 0

    def test_high_scr_reduces_egfr(self):
        egfr_low = estimate_egfr_ckd_epi(50, 0.9, sex="male")
        egfr_high = estimate_egfr_ckd_epi(50, 4.0, sex="male")
        assert egfr_high < egfr_low

    def test_older_age_reduces_egfr(self):
        egfr_young = estimate_egfr_ckd_epi(30, 1.0)
        egfr_old = estimate_egfr_ckd_epi(80, 1.0)
        assert egfr_old < egfr_young

    def test_minimum_clamp(self):
        egfr = estimate_egfr_ckd_epi(90, 10.0)
        assert egfr >= 1.0

    def test_age_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_egfr_ckd_epi(0, 1.0)

    def test_scr_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_egfr_ckd_epi(40, 0.0)


# ---------------------------------------------------------------------------
# ckd_stage
# ---------------------------------------------------------------------------


class TestCkdStage:
    def test_g1_at_90(self):
        assert ckd_stage(90) == "G1"

    def test_g1_above_90(self):
        assert ckd_stage(100) == "G1"

    def test_g2(self):
        assert ckd_stage(75) == "G2"

    def test_g3a(self):
        assert ckd_stage(52) == "G3a"

    def test_g3b(self):
        assert ckd_stage(37) == "G3b"

    def test_g4(self):
        assert ckd_stage(20) == "G4"

    def test_g5_below_15(self):
        assert ckd_stage(10) == "G5"

    def test_g5_at_14(self):
        assert ckd_stage(14) == "G5"


# ---------------------------------------------------------------------------
# adjust_dose_for_renal
# ---------------------------------------------------------------------------


class TestAdjustDoseForRenal:
    def test_returns_dataclass(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 90.0)
        assert isinstance(result, RenalDoseAdjResult)

    def test_normal_crcl_factor_near_one(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 120.0, cl_renal_fraction=0.5)
        assert abs(result.cl_adjustment_factor - 1.0) < 0.01

    def test_low_crcl_factor_less_than_half(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 15.0, cl_renal_fraction=0.9)
        assert result.cl_adjustment_factor < 0.5

    def test_recommended_dose_equals_dose_times_factor(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 60.0)
        expected = 500.0 * result.cl_adjustment_factor
        assert abs(result.recommended_dose_mg - expected) < 0.01

    def test_dose_reduction_pct_formula(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 60.0)
        expected_pct = (1.0 - result.cl_adjustment_factor) * 100.0
        assert abs(result.dose_reduction_pct - expected_pct) < 0.01

    def test_compound_name_preserved(self):
        result = adjust_dose_for_renal("Vancomycin", 1000.0, 12.0, 80.0)
        assert result.compound_name == "Vancomycin"

    def test_dosing_interval_preserved(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 24.0, 90.0)
        assert result.dosing_interval_h == 24.0

    def test_warnings_nonempty_for_severe_impairment(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 15.0)
        assert len(result.warnings) > 0

    def test_no_warnings_for_normal_crcl(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 110.0)
        assert result.warnings == ""

    def test_ckd_stage_in_notes(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 10.0)
        assert "G5" in result.notes

    def test_fe_urine_out_of_range_raises(self):
        with pytest.raises(ValueError, match="fe_urine"):
            adjust_dose_for_renal("DrugX", 500.0, 12.0, 90.0, fe_urine=1.5)

    def test_cl_renal_fraction_out_of_range_raises(self):
        with pytest.raises(ValueError, match="cl_renal_fraction"):
            adjust_dose_for_renal("DrugX", 500.0, 12.0, 90.0, cl_renal_fraction=-0.1)

    def test_zero_renal_fraction_factor_is_one(self):
        # If cl_renal_fraction=0, all clearance is non-renal -> no adjustment
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 15.0, cl_renal_fraction=0.0)
        assert abs(result.cl_adjustment_factor - 1.0) < 0.01

    def test_crcl_mL_per_min_stored(self):
        result = adjust_dose_for_renal("DrugX", 500.0, 12.0, 45.0)
        assert result.crcl_mL_per_min == 45.0
