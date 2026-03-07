"""Tests for Phase 784 — renal_dose_adjustment.py"""

import pytest

from omega_pbpk.clinical.renal_dose_adjustment import (
    RenalDoseAdjustmentResult,
    adjust_dose_for_renal_impairment,
    ckd_stage_from_egfr,
    recommend_dose_adjustment,
)

# ---------------------------------------------------------------------------
# ckd_stage_from_egfr — boundary tests
# ---------------------------------------------------------------------------


def test_ckd_stage_g1_at_90():
    assert ckd_stage_from_egfr(90.0) == "G1"


def test_ckd_stage_g1_above_90():
    assert ckd_stage_from_egfr(120.0) == "G1"


def test_ckd_stage_g2_at_60():
    assert ckd_stage_from_egfr(60.0) == "G2"


def test_ckd_stage_g2_at_89():
    assert ckd_stage_from_egfr(89.0) == "G2"


def test_ckd_stage_g3a_at_45():
    assert ckd_stage_from_egfr(45.0) == "G3a"


def test_ckd_stage_g3b_at_30():
    assert ckd_stage_from_egfr(30.0) == "G3b"


def test_ckd_stage_g4_at_15():
    assert ckd_stage_from_egfr(15.0) == "G4"


def test_ckd_stage_g5_below_15():
    assert ckd_stage_from_egfr(10.0) == "G5"


def test_ckd_stage_g5_at_zero():
    assert ckd_stage_from_egfr(0.0) == "G5"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = adjust_dose_for_renal_impairment(
        drug_name="DrugA",
        normal_dose_mg=500.0,
        normal_interval_h=12.0,
        fe_urine=0.6,
        egfr_ml_per_min=30.0,
    )
    assert isinstance(result, RenalDoseAdjustmentResult)


# ---------------------------------------------------------------------------
# Dose fraction < 1 when renally impaired + high fe_urine
# ---------------------------------------------------------------------------


def test_dose_fraction_reduced_with_high_fe_urine_and_low_egfr():
    result = adjust_dose_for_renal_impairment(
        drug_name="RenalDrug",
        normal_dose_mg=500.0,
        normal_interval_h=12.0,
        fe_urine=0.8,
        egfr_ml_per_min=20.0,
        normal_egfr_ml_per_min=100.0,
        method="dose_reduction",
    )
    assert result.dose_fraction < 1.0


def test_adjusted_dose_lower_than_normal_dose_reduction():
    result = adjust_dose_for_renal_impairment(
        drug_name="RenalDrug",
        normal_dose_mg=500.0,
        normal_interval_h=12.0,
        fe_urine=0.8,
        egfr_ml_per_min=20.0,
        method="dose_reduction",
    )
    assert result.adjusted_dose_mg < result.normal_dose_mg


# ---------------------------------------------------------------------------
# fe_urine = 0 → no adjustment
# ---------------------------------------------------------------------------


def test_no_adjustment_when_fe_urine_zero():
    result = adjust_dose_for_renal_impairment(
        drug_name="HepaticDrug",
        normal_dose_mg=100.0,
        normal_interval_h=24.0,
        fe_urine=0.0,
        egfr_ml_per_min=10.0,  # severe renal impairment
        method="dose_reduction",
    )
    # df = 1 - 0*(1 - ratio) = 1.0, so dose unchanged
    assert abs(result.adjusted_dose_mg - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def test_interval_extension_keeps_dose_same():
    result = adjust_dose_for_renal_impairment(
        drug_name="DrugA",
        normal_dose_mg=250.0,
        normal_interval_h=8.0,
        fe_urine=0.7,
        egfr_ml_per_min=30.0,
        method="interval_extension",
    )
    assert abs(result.adjusted_dose_mg - 250.0) < 1e-6
    assert result.adjusted_interval_h >= result.normal_interval_h


def test_combined_method_adjusts_both():
    result = adjust_dose_for_renal_impairment(
        drug_name="DrugA",
        normal_dose_mg=500.0,
        normal_interval_h=12.0,
        fe_urine=0.8,
        egfr_ml_per_min=20.0,
        method="combined",
    )
    assert result.adjusted_dose_mg < result.normal_dose_mg
    assert result.adjusted_interval_h > result.normal_interval_h


# ---------------------------------------------------------------------------
# Dialysis supplement flag
# ---------------------------------------------------------------------------


def test_dialysis_supplement_true_when_g5_high_fe_urine():
    result = adjust_dose_for_renal_impairment(
        drug_name="DialysisDrug",
        normal_dose_mg=100.0,
        normal_interval_h=12.0,
        fe_urine=0.7,
        egfr_ml_per_min=5.0,
    )
    assert result.requires_dialysis_supplement is True


def test_dialysis_supplement_false_when_low_fe_urine():
    result = adjust_dose_for_renal_impairment(
        drug_name="HepaticDrug",
        normal_dose_mg=100.0,
        normal_interval_h=12.0,
        fe_urine=0.2,
        egfr_ml_per_min=5.0,
    )
    assert result.requires_dialysis_supplement is False


def test_dialysis_supplement_false_when_normal_egfr():
    result = adjust_dose_for_renal_impairment(
        drug_name="DrugA",
        normal_dose_mg=100.0,
        normal_interval_h=12.0,
        fe_urine=0.9,
        egfr_ml_per_min=90.0,
    )
    assert result.requires_dialysis_supplement is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_fe_urine_negative_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_renal_impairment("X", 100.0, 12.0, fe_urine=-0.1, egfr_ml_per_min=60.0)


def test_invalid_fe_urine_above_one_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_renal_impairment("X", 100.0, 12.0, fe_urine=1.5, egfr_ml_per_min=60.0)


def test_invalid_egfr_negative_raises():
    with pytest.raises(ValueError):
        adjust_dose_for_renal_impairment("X", 100.0, 12.0, fe_urine=0.5, egfr_ml_per_min=-1.0)


# ---------------------------------------------------------------------------
# recommend_dose_adjustment
# ---------------------------------------------------------------------------


def test_recommend_returns_string():
    s = recommend_dose_adjustment("DrugA", 500.0, 12.0, 0.6, 30.0)
    assert isinstance(s, str)
    assert len(s) > 0


def test_recommend_contains_drug_name():
    s = recommend_dose_adjustment("Metformin", 500.0, 12.0, 0.6, 30.0)
    assert "Metformin" in s
