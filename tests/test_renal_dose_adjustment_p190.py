"""Tests for Phase 190: renal_dose_adjustment_p190."""

import pytest

from omega_pbpk.clinical.renal_dose_adjustment_p190 import (
    RenalDoseResult,
    adjust_renal_dose,
    screen_renal_impairment,
)

# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 90.0)
        assert isinstance(result, RenalDoseResult)

    def test_drug_name_preserved(self):
        result = adjust_renal_dose("Metformin", 500.0, 1.0, 90.0)
        assert result.drug_name == "Metformin"

    def test_egfr_preserved(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 45.0)
        assert result.egfr_mL_min == 45.0

    def test_fe_renal_preserved(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.7, 60.0)
        assert result.fe_renal == 0.7

    def test_normal_dose_preserved(self):
        result = adjust_renal_dose("DrugA", 250.0, 0.5, 90.0)
        assert result.normal_dose_mg == 250.0

    def test_has_notes_field(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 90.0)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Normal eGFR (90 mL/min) → adjusted dose ≈ normal dose
# ---------------------------------------------------------------------------


class TestNormalEgfr:
    def test_normal_egfr_cl_ratio_is_1(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 90.0)
        assert abs(result.cl_ratio - 1.0) < 1e-9

    def test_normal_egfr_adjusted_dose_equals_normal(self):
        result = adjust_renal_dose("DrugA", 200.0, 0.9, 90.0)
        assert abs(result.adjusted_dose_mg - 200.0) < 1e-9

    def test_normal_egfr_stage_G1(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 90.0)
        assert result.egfr_stage == "G1"

    def test_above_normal_egfr_stage_G1(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 120.0)
        assert result.egfr_stage == "G1"


# ---------------------------------------------------------------------------
# Low eGFR → reduced dose
# ---------------------------------------------------------------------------


class TestLowEgfr:
    def test_low_egfr_reduces_dose(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 30.0)
        assert result.adjusted_dose_mg < 100.0

    def test_very_low_egfr_severely_reduces_dose(self):
        result = adjust_renal_dose("DrugA", 100.0, 1.0, 10.0)
        # cl_ratio = 1.0 * (10/90) + 0 ≈ 0.111
        expected_cl = 10.0 / 90.0
        assert abs(result.cl_ratio - expected_cl) < 1e-9
        assert result.adjusted_dose_mg < 20.0

    def test_low_egfr_extends_interval(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 30.0, dosing_interval_h=24.0)
        assert result.adjusted_interval_h > 24.0

    def test_zero_egfr_fe_renal_1_cl_ratio_0(self):
        result = adjust_renal_dose("DrugA", 100.0, 1.0, 0.0)
        assert result.cl_ratio == 0.0


# ---------------------------------------------------------------------------
# fe_renal = 0 → no dose adjustment needed
# ---------------------------------------------------------------------------


class TestFeRenalZero:
    def test_fe_renal_zero_cl_ratio_is_1(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.0, 10.0)
        assert abs(result.cl_ratio - 1.0) < 1e-9

    def test_fe_renal_zero_adjusted_dose_unchanged(self):
        result = adjust_renal_dose("DrugA", 200.0, 0.0, 5.0)
        assert abs(result.adjusted_dose_mg - 200.0) < 1e-9

    def test_fe_renal_zero_no_dose_reduction_recommended(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.0, 10.0)
        assert result.dose_reduction_recommended is False


# ---------------------------------------------------------------------------
# eGFR stages assigned correctly
# ---------------------------------------------------------------------------


class TestEgfrStages:
    @pytest.mark.parametrize(
        "egfr,expected_stage",
        [
            (100.0, "G1"),
            (90.0, "G1"),
            (75.0, "G2"),
            (60.0, "G2"),
            (50.0, "G3a"),
            (45.0, "G3a"),
            (35.0, "G3b"),
            (30.0, "G3b"),
            (20.0, "G4"),
            (15.0, "G4"),
            (10.0, "G5"),
            (0.0, "G5"),
        ],
    )
    def test_stage(self, egfr, expected_stage):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, egfr)
        assert result.egfr_stage == expected_stage, (
            f"eGFR={egfr} should be {expected_stage}, got {result.egfr_stage}"
        )


# ---------------------------------------------------------------------------
# dose_reduction_recommended logic
# ---------------------------------------------------------------------------


class TestDoseReductionRecommended:
    def test_fe_renal_gt_0p5_and_egfr_lt_60_recommended(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.6, 50.0)
        assert result.dose_reduction_recommended is True

    def test_fe_renal_le_0p5_not_recommended(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 50.0)
        assert result.dose_reduction_recommended is False

    def test_egfr_ge_60_not_recommended_even_high_fe(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.9, 60.0)
        assert result.dose_reduction_recommended is False

    def test_fe_renal_exactly_0p5_not_recommended(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.5, 30.0)
        assert result.dose_reduction_recommended is False

    def test_dose_adjustment_pct_positive_when_reduced(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 30.0)
        assert result.dose_adjustment_pct > 0.0

    def test_dose_adjustment_pct_zero_at_normal_egfr(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.8, 90.0)
        assert abs(result.dose_adjustment_pct) < 1e-9


# ---------------------------------------------------------------------------
# cl_ratio calculation
# ---------------------------------------------------------------------------


class TestClRatio:
    def test_cl_ratio_formula(self):
        fe = 0.7
        egfr = 45.0
        result = adjust_renal_dose("DrugA", 100.0, fe, egfr)
        expected = fe * (egfr / 90.0) + (1.0 - fe)
        assert abs(result.cl_ratio - expected) < 1e-9

    def test_cl_ratio_1_when_fe_0(self):
        result = adjust_renal_dose("DrugA", 100.0, 0.0, 10.0)
        assert abs(result.cl_ratio - 1.0) < 1e-9

    def test_cl_ratio_proportional_to_egfr_when_fe_1(self):
        result = adjust_renal_dose("DrugA", 100.0, 1.0, 45.0)
        expected = 45.0 / 90.0
        assert abs(result.cl_ratio - expected) < 1e-9


# ---------------------------------------------------------------------------
# screen_renal_impairment
# ---------------------------------------------------------------------------


class TestScreenRenalImpairment:
    def test_returns_correct_count(self):
        egfr_values = [90.0, 60.0, 45.0, 30.0, 15.0]
        results = screen_renal_impairment("DrugB", 100.0, 0.8, 24.0, egfr_values)
        assert len(results) == 5

    def test_returns_list_of_dataclasses(self):
        egfr_values = [90.0, 60.0, 30.0]
        results = screen_renal_impairment("DrugB", 100.0, 0.8, 24.0, egfr_values)
        for r in results:
            assert isinstance(r, RenalDoseResult)

    def test_ordering_preserved(self):
        egfr_values = [90.0, 60.0, 30.0, 15.0, 5.0]
        results = screen_renal_impairment("DrugB", 100.0, 0.8, 24.0, egfr_values)
        for i, (r, egfr) in enumerate(zip(results, egfr_values)):
            assert r.egfr_mL_min == egfr, f"Index {i}: expected eGFR {egfr}, got {r.egfr_mL_min}"

    def test_doses_decrease_with_egfr_for_high_fe(self):
        egfr_values = [90.0, 60.0, 30.0, 10.0]
        results = screen_renal_impairment("DrugB", 100.0, 0.9, 24.0, egfr_values)
        doses = [r.adjusted_dose_mg for r in results]
        for i in range(len(doses) - 1):
            assert doses[i] >= doses[i + 1], (
                f"Expected dose to decrease with eGFR, but dose[{i}]={doses[i]} < dose[{i + 1}]={doses[i + 1]}"
            )

    def test_empty_egfr_list(self):
        results = screen_renal_impairment("DrugB", 100.0, 0.8, 24.0, [])
        assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            adjust_renal_dose("DrugX", 0.0, 0.5, 90.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            adjust_renal_dose("DrugX", -100.0, 0.5, 90.0)

    def test_fe_renal_below_0_raises(self):
        with pytest.raises(ValueError):
            adjust_renal_dose("DrugX", 100.0, -0.1, 90.0)

    def test_fe_renal_above_1_raises(self):
        with pytest.raises(ValueError):
            adjust_renal_dose("DrugX", 100.0, 1.1, 90.0)

    def test_egfr_negative_raises(self):
        with pytest.raises(ValueError):
            adjust_renal_dose("DrugX", 100.0, 0.5, -5.0)

    def test_valid_boundary_fe_0(self):
        result = adjust_renal_dose("DrugX", 100.0, 0.0, 90.0)
        assert isinstance(result, RenalDoseResult)

    def test_valid_boundary_fe_1(self):
        result = adjust_renal_dose("DrugX", 100.0, 1.0, 90.0)
        assert isinstance(result, RenalDoseResult)

    def test_valid_boundary_egfr_0(self):
        result = adjust_renal_dose("DrugX", 100.0, 0.5, 0.0)
        assert isinstance(result, RenalDoseResult)
