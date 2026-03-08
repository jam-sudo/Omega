"""Tests for warfarin dose optimization (Phase 1019)."""

import pytest

from omega_pbpk.clinical.warfarin_dose_optimization import (
    WarfarinDoseOptResult,
    compare_genotype_doses,
    optimize_warfarin_dose,
)


def test_returns_dataclass():
    result = optimize_warfarin_dose("P001")
    assert isinstance(result, WarfarinDoseOptResult)


def test_weekly_dose_in_range():
    result = optimize_warfarin_dose("P001")
    assert 5.0 <= result.weekly_dose_mg <= 70.0


def test_daily_dose_equals_weekly_div_7():
    result = optimize_warfarin_dose("P001")
    assert abs(result.daily_dose_mg - result.weekly_dose_mg / 7) < 1e-3


def test_predicted_inr_positive():
    result = optimize_warfarin_dose("P001", target_inr=2.5)
    assert result.predicted_inr > 0


def test_predicted_inr_matches_target():
    result = optimize_warfarin_dose("P001", target_inr=2.5)
    assert result.predicted_inr == 2.5


def test_therapeutic_range_probability_in_range():
    result = optimize_warfarin_dose("P001")
    assert 0.0 <= result.therapeutic_range_probability <= 1.0


def test_bleeding_risk_valid():
    result = optimize_warfarin_dose("P001")
    assert result.bleeding_risk in {"low", "moderate", "high"}


def test_initiation_dose_valid():
    result = optimize_warfarin_dose("P001")
    assert result.initiation_dose_mg in {"low", "standard", "high"}


def test_monitoring_frequency_valid():
    result = optimize_warfarin_dose("P001")
    assert result.monitoring_frequency in {"daily", "every_2_days", "weekly"}


def test_star3_star3_lower_dose_than_star1_star1():
    r1 = optimize_warfarin_dose("P001", cyp2c9_genotype="*1/*1")
    r3 = optimize_warfarin_dose("P001", cyp2c9_genotype="*3/*3")
    assert r3.weekly_dose_mg < r1.weekly_dose_mg


def test_aa_vkorc1_lower_dose_than_gg():
    r_gg = optimize_warfarin_dose("P001", vkorc1_genotype="GG")
    r_aa = optimize_warfarin_dose("P001", vkorc1_genotype="AA")
    assert r_aa.weekly_dose_mg < r_gg.weekly_dose_mg


def test_older_patient_lower_dose():
    r_young = optimize_warfarin_dose("P001", age=45)
    r_old = optimize_warfarin_dose("P001", age=85)
    assert r_old.weekly_dose_mg <= r_young.weekly_dose_mg


def test_star3_star3_monitoring_daily():
    result = optimize_warfarin_dose("P001", cyp2c9_genotype="*3/*3")
    assert result.monitoring_frequency == "daily"


def test_aa_vkorc1_monitoring_daily():
    result = optimize_warfarin_dose("P001", vkorc1_genotype="AA")
    assert result.monitoring_frequency == "daily"


def test_star1_star1_gg_monitoring_weekly():
    result = optimize_warfarin_dose("P001", cyp2c9_genotype="*1/*1", vkorc1_genotype="GG")
    assert result.monitoring_frequency == "weekly"


def test_compare_genotype_doses_returns_six():
    results = compare_genotype_doses("P001")
    assert len(results) == 6


def test_compare_genotype_doses_sorted_ascending():
    results = compare_genotype_doses("P001")
    doses = [r.weekly_dose_mg for r in results]
    assert doses == sorted(doses)


def test_invalid_cyp2c9_raises():
    with pytest.raises(ValueError, match="CYP2C9"):
        optimize_warfarin_dose("P001", cyp2c9_genotype="*4/*4")


def test_invalid_vkorc1_raises():
    with pytest.raises(ValueError, match="VKORC1"):
        optimize_warfarin_dose("P001", vkorc1_genotype="TT")


def test_invalid_target_inr_raises():
    with pytest.raises(ValueError, match="target_inr"):
        optimize_warfarin_dose("P001", target_inr=-1.0)


def test_high_inr_target_increases_dose():
    r_lo = optimize_warfarin_dose("P001", target_inr=2.0)
    r_hi = optimize_warfarin_dose("P001", target_inr=3.0)
    assert r_hi.weekly_dose_mg >= r_lo.weekly_dose_mg


def test_patient_id_preserved():
    result = optimize_warfarin_dose("WARFARIN_TEST_123")
    assert result.patient_id == "WARFARIN_TEST_123"
