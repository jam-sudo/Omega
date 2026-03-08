"""Tests for renal_impairment_dosing module (Phase 205 + Phase 385)."""

import math

import pytest

from omega_pbpk.clinical.renal_impairment_dosing import (
    RenalDosingResult,
    RenalImpairmentDoseResult,
    adjust_dose_for_ckd,
    calculate_renal_dose,
    ckd_dose_table,
    screen_ckd_stages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**overrides):
    """Return a result with sensible defaults, overriding specific fields."""
    kwargs = dict(
        drug_name="TestDrug",
        gfr_mL_per_min=60.0,
        fe_renal=0.5,
        cl_normal_L_per_h=5.0,
        vd_L=50.0,
        dose_normal_mg=500.0,
        interval_normal_h=24.0,
        dose_strategy="dose_reduction",
    )
    kwargs.update(overrides)
    return adjust_dose_for_ckd(**kwargs)


# ---------------------------------------------------------------------------
# CKD stage assignment
# ---------------------------------------------------------------------------


class TestCKDStageAssignment:
    def test_gfr_95_is_stage1(self):
        r = adjust_dose_for_ckd("Drug", 95.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 1"

    def test_gfr_90_is_stage1(self):
        r = adjust_dose_for_ckd("Drug", 90.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 1"

    def test_gfr_75_is_stage2(self):
        r = adjust_dose_for_ckd("Drug", 75.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 2"

    def test_gfr_60_is_stage2(self):
        r = adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 2"

    def test_gfr_52_is_stage3a(self):
        r = adjust_dose_for_ckd("Drug", 52.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 3A"

    def test_gfr_37_is_stage3b(self):
        r = adjust_dose_for_ckd("Drug", 37.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 3B"

    def test_gfr_22_is_stage4(self):
        r = adjust_dose_for_ckd("Drug", 22.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 4"

    def test_gfr_10_is_stage5(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 5 (ESRD)"

    def test_gfr_0_is_stage5(self):
        r = adjust_dose_for_ckd("Drug", 0.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 5 (ESRD)"


# ---------------------------------------------------------------------------
# Clearance calculation (Giusti-Hayton)
# ---------------------------------------------------------------------------


class TestClearanceAdjustment:
    def test_fe_renal_zero_cl_unchanged(self):
        """fe_renal=0: non-renally cleared drug — CL should be unchanged."""
        r = adjust_dose_for_ckd("Drug", 10.0, 0.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0, rel_tol=1e-9)

    def test_fe_renal_one_gfr_zero_cl_floored(self):
        """fe_renal=1, GFR=0: CL should be floored at 5% of normal."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0 * 0.05, rel_tol=1e-9)

    def test_fe_renal_one_gfr_normal_cl_unchanged(self):
        """fe_renal=1, GFR=120 (normal): CL factor = 1."""
        r = adjust_dose_for_ckd("Drug", 120.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0, rel_tol=1e-9)

    def test_cl_adjusted_lower_than_normal_for_ckd(self):
        r = adjust_dose_for_ckd("Drug", 30.0, 0.8, 5.0, 50.0, 500.0)
        assert r.cl_adjusted_L_per_h < r.cl_normal_L_per_h

    def test_cl_adjusted_is_floored(self):
        """Extreme CKD + high fe_renal: CL floor at 5%."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 10.0, 100.0, 1000.0)
        assert r.cl_adjusted_L_per_h >= 10.0 * 0.05 - 1e-9


# ---------------------------------------------------------------------------
# Dose strategies
# ---------------------------------------------------------------------------


class TestDoseStrategies:
    def test_dose_reduction_changes_dose_not_interval(self):
        r = adjust_dose_for_ckd(
            "Drug",
            30.0,
            0.7,
            5.0,
            50.0,
            500.0,
            interval_normal_h=24.0,
            dose_strategy="dose_reduction",
        )
        assert r.interval_adjusted_h == 24.0
        assert r.dose_adjusted_mg < r.dose_normal_mg

    def test_interval_extension_changes_interval_not_dose(self):
        r = adjust_dose_for_ckd(
            "Drug",
            30.0,
            0.7,
            5.0,
            50.0,
            500.0,
            interval_normal_h=24.0,
            dose_strategy="interval_extension",
        )
        assert r.dose_adjusted_mg == 500.0
        assert r.interval_adjusted_h > 24.0

    def test_dose_reduction_pct_correct(self):
        """With fe_renal=1, GFR=60: adj_factor = 1 - 1*(1-60/120) = 0.5."""
        r = adjust_dose_for_ckd("Drug", 60.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.dose_reduction_pct, 50.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Half-life and accumulation risk
# ---------------------------------------------------------------------------


class TestHalfLifeAndAccumulation:
    def test_t_half_normal_formula(self):
        """t_half = 0.693 * Vd / CL."""
        r = adjust_dose_for_ckd("Drug", 120.0, 0.0, 5.0, 50.0, 500.0)
        expected = 0.693 / (5.0 / 50.0)
        assert math.isclose(r.t_half_normal_h, expected, rel_tol=1e-9)

    def test_t_half_ckd_longer_than_normal(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.9, 5.0, 50.0, 500.0)
        assert r.t_half_adjusted_h > r.t_half_normal_h

    def test_accumulation_risk_high_when_long_half_life(self):
        """Force high accumulation: very long t_half vs short interval."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 5.0, 5000.0, 500.0, interval_normal_h=24.0)
        assert r.accumulation_risk == "high"

    def test_accumulation_risk_low_for_normal_gfr(self):
        """At GFR=120, no CKD adjustment needed → t_half should be short."""
        r = adjust_dose_for_ckd("Drug", 120.0, 0.5, 5.0, 10.0, 500.0, interval_normal_h=24.0)
        # t_half = 0.693 / (5/10) = ~1.4 h << 24 h
        assert r.accumulation_risk == "low"


# ---------------------------------------------------------------------------
# Monitoring messages
# ---------------------------------------------------------------------------


class TestMonitoring:
    def test_tdm_for_severe_ckd(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.5, 5.0, 50.0, 500.0)
        assert "TDM" in r.monitoring

    def test_monitor_renal_for_moderate_ckd(self):
        r = adjust_dose_for_ckd("Drug", 45.0, 0.5, 5.0, 50.0, 500.0)
        assert "Monitor" in r.monitoring

    def test_routine_for_stage1(self):
        r = adjust_dose_for_ckd("Drug", 95.0, 0.5, 5.0, 50.0, 500.0)
        assert r.monitoring == "Routine monitoring"


# ---------------------------------------------------------------------------
# ckd_dose_table
# ---------------------------------------------------------------------------


class TestCKDDoseTable:
    def test_returns_six_entries(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        assert len(table) == 6

    def test_all_entries_are_results(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        assert all(isinstance(r, RenalImpairmentDoseResult) for r in table)

    def test_gfr_values_decreasing(self):
        """Table should go from highest (Stage 1) to lowest GFR (Stage 5)."""
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        gfrs = [r.gfr_mL_per_min for r in table]
        assert gfrs == sorted(gfrs, reverse=True)

    def test_dose_decreasing_with_gfr(self):
        """For renally-cleared drug, lower GFR → lower adjusted dose."""
        table = ckd_dose_table("Drug", 0.9, 5.0, 50.0, 500.0, dose_strategy="dose_reduction")
        doses = [r.dose_adjusted_mg for r in table]
        assert doses == sorted(doses, reverse=True)

    def test_kwargs_forwarded(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0, dose_strategy="interval_extension")
        # All doses should equal normal dose when strategy is interval_extension
        assert all(math.isclose(r.dose_adjusted_mg, 500.0) for r in table)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr"):
            adjust_dose_for_ckd("Drug", -1.0, 0.5, 5.0, 50.0, 500.0)

    def test_fe_renal_gt1_raises(self):
        with pytest.raises(ValueError, match="fe_renal"):
            adjust_dose_for_ckd("Drug", 60.0, 1.1, 5.0, 50.0, 500.0)

    def test_fe_renal_negative_raises(self):
        with pytest.raises(ValueError, match="fe_renal"):
            adjust_dose_for_ckd("Drug", 60.0, -0.1, 5.0, 50.0, 500.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_normal"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 0.0, 50.0, 500.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 0.0, 500.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_normal"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_frozen_dataclass(self):
        r = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            r.dose_adjusted_mg = 999.0  # type: ignore[misc]


# ===========================================================================
# Phase 385 tests — RenalDosingResult / calculate_renal_dose / screen_ckd_stages
# ===========================================================================


class TestPhase385ReturnType:
    def test_returns_renal_dosing_result(self):
        result = calculate_renal_dose("DrugA", 100.0, 90.0, 0.5)
        assert isinstance(result, RenalDosingResult)

    def test_result_fields_populated(self):
        result = calculate_renal_dose("DrugA", 100.0, 60.0, 0.7)
        assert result.drug_name == "DrugA"
        assert result.egfr_mL_per_min == 60.0
        assert result.fe_unchanged == 0.7
        assert result.baseline_dose_mg == 100.0


class TestPhase385CKDStage:
    def test_ckd_stage_g1(self):
        assert calculate_renal_dose("X", 100.0, 95.0, 0.5).ckd_stage == "G1"

    def test_ckd_stage_g2(self):
        assert calculate_renal_dose("X", 100.0, 75.0, 0.5).ckd_stage == "G2"

    def test_ckd_stage_g3a(self):
        assert calculate_renal_dose("X", 100.0, 52.0, 0.5).ckd_stage == "G3a"

    def test_ckd_stage_g3b(self):
        assert calculate_renal_dose("X", 100.0, 37.0, 0.5).ckd_stage == "G3b"

    def test_ckd_stage_g4(self):
        assert calculate_renal_dose("X", 100.0, 22.0, 0.5).ckd_stage == "G4"

    def test_ckd_stage_g5(self):
        assert calculate_renal_dose("X", 100.0, 10.0, 0.5).ckd_stage == "G5"


class TestPhase385DoseAdjustment:
    def test_g1_low_fe_no_adjustment(self):
        """Normal eGFR + low fe => virtually no dose reduction."""
        result = calculate_renal_dose("DrugB", 200.0, 100.0, 0.05)
        assert abs(result.adjusted_dose_mg - 200.0) < 0.01
        assert abs(result.dose_reduction_pct) < 0.01
        assert result.dosing_interval_adjustment == "standard"
        assert result.clinical_guidance == "No dose adjustment required"

    def test_g5_high_fe_large_reduction(self):
        result = calculate_renal_dose("DrugC", 500.0, 5.0, 0.9)
        assert result.adjusted_dose_mg < 200.0
        assert result.dose_reduction_pct > 50.0
        assert result.ckd_stage == "G5"

    def test_dose_reduction_pct_calculation(self):
        egfr = 30.0
        fe = 0.6
        result = calculate_renal_dose("DrugD", 100.0, egfr, fe)
        renal_cl = egfr / 100.0
        expected_tcf = (1 - fe) + fe * renal_cl
        expected_pct = (1 - expected_tcf) * 100.0
        assert abs(result.dose_reduction_pct - expected_pct) < 1e-6

    def test_adjusted_dose_matches_model(self):
        baseline = 250.0
        egfr = 40.0
        fe = 0.5
        result = calculate_renal_dose("DrugE", baseline, egfr, fe)
        renal_cl = egfr / 100.0
        total_cl = (1 - fe) + fe * renal_cl
        assert abs(result.adjusted_dose_mg - baseline * total_cl) < 1e-9


class TestPhase385AUCRatio:
    def test_auc_ratio_increases_with_impairment(self):
        r_normal = calculate_renal_dose("DrugF", 100.0, 100.0, 0.7)
        r_moderate = calculate_renal_dose("DrugF", 100.0, 45.0, 0.7)
        r_severe = calculate_renal_dose("DrugF", 100.0, 10.0, 0.7)
        assert r_normal.auc_ratio_impaired_to_normal <= r_moderate.auc_ratio_impaired_to_normal
        assert r_moderate.auc_ratio_impaired_to_normal <= r_severe.auc_ratio_impaired_to_normal

    def test_auc_ratio_one_at_normal_egfr(self):
        result = calculate_renal_dose("DrugG", 100.0, 100.0, 0.8, normal_egfr=100.0)
        assert abs(result.auc_ratio_impaired_to_normal - 1.0) < 1e-9


class TestPhase385SafetyFlags:
    def test_monitoring_required_for_g4(self):
        assert calculate_renal_dose("DrugH", 100.0, 22.0, 0.5).monitoring_required is True

    def test_monitoring_required_for_g5(self):
        assert calculate_renal_dose("DrugH", 100.0, 5.0, 0.5).monitoring_required is True

    def test_monitoring_not_required_g1_low_fe(self):
        assert calculate_renal_dose("DrugH", 100.0, 95.0, 0.1).monitoring_required is False

    def test_dialysis_supplement_g5_high_fe(self):
        assert calculate_renal_dose("DrugI", 100.0, 5.0, 0.8).dialysis_supplement_needed is True

    def test_no_dialysis_supplement_g3(self):
        assert calculate_renal_dose("DrugI", 100.0, 45.0, 0.8).dialysis_supplement_needed is False


class TestPhase385ScreenCKDStages:
    def test_returns_six_results(self):
        assert len(screen_ckd_stages("DrugJ", 100.0, 0.6)) == 6

    def test_sorted_descending(self):
        results = screen_ckd_stages("DrugJ", 100.0, 0.6)
        egfrs = [r.egfr_mL_per_min for r in results]
        assert egfrs == sorted(egfrs, reverse=True)

    def test_covers_all_ckd_stages(self):
        results = screen_ckd_stages("DrugJ", 100.0, 0.6)
        stages = {r.ckd_stage for r in results}
        assert stages == {"G1", "G2", "G3a", "G3b", "G4", "G5"}


class TestPhase385Validation:
    def test_negative_egfr(self):
        with pytest.raises(ValueError, match="egfr_mL_per_min"):
            calculate_renal_dose("DrugX", 100.0, -1.0, 0.5)

    def test_fe_above_one(self):
        with pytest.raises(ValueError, match="fe_unchanged"):
            calculate_renal_dose("DrugX", 100.0, 60.0, 1.5)

    def test_fe_negative(self):
        with pytest.raises(ValueError, match="fe_unchanged"):
            calculate_renal_dose("DrugX", 100.0, 60.0, -0.1)

    def test_negative_baseline_dose(self):
        with pytest.raises(ValueError, match="baseline_dose_mg"):
            calculate_renal_dose("DrugX", -50.0, 60.0, 0.5)

    def test_zero_normal_egfr(self):
        with pytest.raises(ValueError, match="normal_egfr"):
            calculate_renal_dose("DrugX", 100.0, 60.0, 0.5, normal_egfr=0.0)
