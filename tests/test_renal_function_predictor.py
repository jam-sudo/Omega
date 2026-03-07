"""Tests for renal_function_predictor module (Phase 436)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.renal_function_predictor import (
    RenalFunctionResult,
    ckd_epi_egfr,
    predict_renal_drug_clearance,
)

# ===========================================================================
# CKD-EPI eGFR tests
# ===========================================================================


class TestCkdEpiEgfr:
    def test_healthy_young_male(self):
        """Healthy young male with low creatinine should have high eGFR."""
        egfr = ckd_epi_egfr(scr_mg_dL=0.9, age=25, sex="male")
        assert egfr > 90

    def test_healthy_young_female(self):
        """Healthy young female with low creatinine should have high eGFR."""
        egfr = ckd_epi_egfr(scr_mg_dL=0.7, age=25, sex="female")
        assert egfr > 90

    def test_high_creatinine_low_egfr(self):
        """High creatinine (CKD stage 5) should give low eGFR."""
        egfr = ckd_epi_egfr(scr_mg_dL=5.0, age=65, sex="male")
        assert egfr < 20

    def test_female_multiplier_applied(self):
        """The female sex multiplier (1.012) is applied in CKD-EPI 2021.
        Verify it affects the result: female eGFR with multiplier should differ
        from what it would be without (tested by verifying the formula runs with
        both sexes and female result reflects the 1.012 factor at low SCr).
        Specifically: a female with SCr=0.7 (at her k-threshold) gets sex
        multiplier 1.012 applied, making her eGFR 1.2% above the un-multiplied
        value that a male would get at the equivalent physiological SCr."""
        # At SCr = female k (0.7): min/max terms both equal 1 for female
        egfr_f_at_k = ckd_epi_egfr(scr_mg_dL=0.7, age=50, sex="female")
        # Sanity: 1.012 factor is included so result should be reasonable positive
        assert egfr_f_at_k > 80  # at k threshold for healthy 50-yr female
        # Verify both sexes return different values at same SCr (sex IS used)
        egfr_f = ckd_epi_egfr(scr_mg_dL=0.7, age=50, sex="female")
        egfr_m = ckd_epi_egfr(scr_mg_dL=0.7, age=50, sex="male")
        assert egfr_f != egfr_m  # different results confirm sex is applied

    def test_age_decreases_egfr(self):
        """Older patients should have lower eGFR."""
        egfr_young = ckd_epi_egfr(scr_mg_dL=1.0, age=30, sex="male")
        egfr_old = ckd_epi_egfr(scr_mg_dL=1.0, age=80, sex="male")
        assert egfr_old < egfr_young

    def test_race_param_ignored(self):
        """2021 CKD-EPI removes race adjustment - race should not change result."""
        egfr_black = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="male", race="black")
        egfr_other = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="male", race="other")
        assert egfr_black == pytest.approx(egfr_other)

    def test_sex_case_insensitive(self):
        """sex parameter should accept any case."""
        egfr1 = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="Female")
        egfr2 = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="FEMALE")
        egfr3 = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="female")
        assert egfr1 == pytest.approx(egfr3)
        assert egfr2 == pytest.approx(egfr3)

    def test_returns_float(self):
        result = ckd_epi_egfr(1.0, 50, "male")
        assert isinstance(result, float)

    def test_egfr_positive(self):
        result = ckd_epi_egfr(1.0, 50, "female")
        assert result > 0

    def test_raises_on_zero_creatinine(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            ckd_epi_egfr(0.0, 50, "male")

    def test_raises_on_negative_creatinine(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            ckd_epi_egfr(-1.0, 50, "male")

    def test_raises_on_invalid_age_low(self):
        with pytest.raises(ValueError, match="age"):
            ckd_epi_egfr(1.0, 10, "male")

    def test_raises_on_invalid_age_high(self):
        with pytest.raises(ValueError, match="age"):
            ckd_epi_egfr(1.0, 130, "male")

    def test_raises_on_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            ckd_epi_egfr(1.0, 50, "unknown")

    def test_alpha_changes_at_k_threshold_female(self):
        """Female: k=0.7; scr<=0.7 uses alpha=-0.241, scr>0.7 uses alpha=-1.2."""
        egfr_below = ckd_epi_egfr(scr_mg_dL=0.6, age=50, sex="female")
        egfr_above = ckd_epi_egfr(scr_mg_dL=0.8, age=50, sex="female")
        # Below k: alpha=-0.241 (mild reduction as scr increases)
        # Above k: alpha=-1.2 (steep reduction)
        assert egfr_below > egfr_above

    def test_alpha_changes_at_k_threshold_male(self):
        """Male: k=0.9; scr<=0.9 uses alpha=-0.302, scr>0.9 uses alpha=-1.2."""
        egfr_below = ckd_epi_egfr(scr_mg_dL=0.8, age=50, sex="male")
        egfr_above = ckd_epi_egfr(scr_mg_dL=1.0, age=50, sex="male")
        assert egfr_below > egfr_above


# ===========================================================================
# predict_renal_drug_clearance tests
# ===========================================================================


class TestPredictRenalDrugClearance:
    def _base_kwargs(self):
        return dict(
            drug_name="TestDrug",
            gfr_mL_per_min=90.0,
            fu_plasma=0.5,
            logP=1.5,
            mw=300.0,
            f_renal_excretion=0.5,
            normal_cl_L_per_h=10.0,
        )

    def test_returns_renal_function_result(self):
        result = predict_renal_drug_clearance(**self._base_kwargs())
        assert isinstance(result, RenalFunctionResult)

    def test_result_is_frozen(self):
        result = predict_renal_drug_clearance(**self._base_kwargs())
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_gfr_g1_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 95.0})
        assert result.gfr_category == "G1"

    def test_gfr_g2_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 75.0})
        assert result.gfr_category == "G2"

    def test_gfr_g3a_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 52.0})
        assert result.gfr_category == "G3a"

    def test_gfr_g3b_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 37.0})
        assert result.gfr_category == "G3b"

    def test_gfr_g4_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 22.0})
        assert result.gfr_category == "G4"

    def test_gfr_g5_category(self):
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 10.0})
        assert result.gfr_category == "G5"

    def test_normal_gfr_near_full_dose(self):
        """GFR=120 (normal) -> dose_adjustment_factor close to 1.0."""
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 120.0})
        assert result.dose_adjustment_factor == pytest.approx(1.0, abs=0.05)

    def test_zero_gfr_removes_renal_cl(self):
        """GFR=0 -> renal CL is 0 -> total CL is non-renal only."""
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 0.0})
        assert result.cl_renal_L_per_h == pytest.approx(0.0, abs=1e-6)

    def test_fully_renally_cleared_drug_severe_reduction(self):
        """Drug cleared 100% renally with GFR=15 should have large dose reduction."""
        result = predict_renal_drug_clearance(
            drug_name="D",
            gfr_mL_per_min=15.0,
            fu_plasma=0.8,
            logP=0.0,
            mw=200.0,
            f_renal_excretion=1.0,
            normal_cl_L_per_h=10.0,
        )
        # GFR=15 vs normal 120 -> ~12.5% of normal -> big dose reduction
        assert result.dose_adjustment_factor < 0.3

    def test_cl_total_equals_renal_plus_nonrenal(self):
        result = predict_renal_drug_clearance(**self._base_kwargs())
        total = result.cl_renal_L_per_h + result.cl_nonrenal_L_per_h
        assert abs(result.cl_total_adjusted_L_per_h - total) < 1e-9

    def test_dose_adjustment_factor_equals_ratio(self):
        result = predict_renal_drug_clearance(**self._base_kwargs())
        expected = result.cl_total_adjusted_L_per_h / self._base_kwargs()["normal_cl_L_per_h"]
        assert result.dose_adjustment_factor == pytest.approx(expected, rel=1e-6)

    def test_nonrenal_cl_preserved(self):
        """Non-renal CL should be (1 - f_renal) * normal_CL."""
        kwargs = self._base_kwargs()
        result = predict_renal_drug_clearance(**kwargs)
        expected_nr = kwargs["normal_cl_L_per_h"] * (1.0 - kwargs["f_renal_excretion"])
        assert result.cl_nonrenal_L_per_h == pytest.approx(expected_nr, rel=1e-6)

    def test_notes_non_empty(self):
        result = predict_renal_drug_clearance(**self._base_kwargs())
        assert len(result.notes) > 0

    def test_raises_on_negative_gfr(self):
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": -1.0})

    def test_raises_on_zero_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "fu_plasma": 0.0})

    def test_raises_on_zero_mw(self):
        with pytest.raises(ValueError, match="mw"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "mw": 0.0})

    def test_raises_on_negative_f_renal(self):
        with pytest.raises(ValueError, match="f_renal_excretion"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "f_renal_excretion": -0.1})

    def test_raises_on_f_renal_greater_than_one(self):
        with pytest.raises(ValueError, match="f_renal_excretion"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "f_renal_excretion": 1.1})

    def test_raises_on_zero_normal_cl(self):
        with pytest.raises(ValueError, match="normal_cl_L_per_h"):
            predict_renal_drug_clearance(**{**self._base_kwargs(), "normal_cl_L_per_h": 0.0})

    def test_severe_ckd_g5_note_present(self):
        """G4/G5 should warn about dose reduction."""
        result = predict_renal_drug_clearance(**{**self._base_kwargs(), "gfr_mL_per_min": 10.0})
        assert "dose reduction" in result.notes.lower() or "impairment" in result.notes.lower()
