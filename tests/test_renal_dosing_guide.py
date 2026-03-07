"""Tests for clinical/renal_dosing_guide.py — Phase 596."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.renal_dosing_guide import (
    RenalDoseResult,
    calculate_renal_dose,
    ckd_stage_from_egfr,
    renal_function_category,
    screen_renal_impairment,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 500.0  # mg
INTERVAL = 12.0  # h
FE = 0.6  # 60% renally eliminated


def calc(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        baseline_dose_mg=DOSE,
        baseline_interval_h=INTERVAL,
        fe_renal=FE,
        egfr_mL_per_min=100.0,
    )
    defaults.update(kwargs)
    return calculate_renal_dose(**defaults)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_result():
    result = calc()
    assert isinstance(result, RenalDoseResult)


# ---------------------------------------------------------------------------
# Dose factor logic
# ---------------------------------------------------------------------------


def test_normal_egfr_full_dose():
    result = calc(egfr_mL_per_min=100.0)
    assert abs(result.dose_adjustment_factor - 1.0) < 1e-9


def test_severe_impairment_reduced_dose():
    result = calc(egfr_mL_per_min=20.0)
    assert result.recommended_dose_mg < DOSE


def test_esrd_lowest_dose():
    r_severe = calc(egfr_mL_per_min=20.0)
    r_esrd = calc(egfr_mL_per_min=5.0)
    assert r_esrd.recommended_dose_mg < r_severe.recommended_dose_mg


def test_dose_factor_in_range():
    for egfr in [5, 20, 45, 70, 100]:
        result = calc(egfr_mL_per_min=float(egfr))
        assert 0.0 < result.dose_adjustment_factor <= 1.0


# ---------------------------------------------------------------------------
# CKD staging
# ---------------------------------------------------------------------------


def test_ckd_stage_g1():
    assert ckd_stage_from_egfr(95.0) == "G1"


def test_ckd_stage_g2():
    assert ckd_stage_from_egfr(70.0) == "G2"


def test_ckd_stage_g3a():
    assert ckd_stage_from_egfr(52.0) == "G3a"


def test_ckd_stage_g3b():
    assert ckd_stage_from_egfr(35.0) == "G3b"


def test_ckd_stage_g4():
    assert ckd_stage_from_egfr(20.0) == "G4"


def test_ckd_stage_g5():
    assert ckd_stage_from_egfr(10.0) == "G5"


# ---------------------------------------------------------------------------
# Renal function categories
# ---------------------------------------------------------------------------


def test_renal_category_normal():
    assert renal_function_category(100.0) == "normal"


def test_renal_category_mild():
    assert renal_function_category(75.0) == "mild"


def test_renal_category_moderate():
    assert renal_function_category(45.0) == "moderate"


def test_renal_category_severe():
    assert renal_function_category(20.0) == "severe"


def test_renal_category_esrd():
    assert renal_function_category(5.0) == "esrd"


# ---------------------------------------------------------------------------
# Contraindication
# ---------------------------------------------------------------------------


def test_contraindicated_flag():
    result = calc(egfr_mL_per_min=20.0, contraindicated_below_egfr=30.0)
    assert result.contraindicated is True


def test_not_contraindicated_above_threshold():
    result = calc(egfr_mL_per_min=50.0, contraindicated_below_egfr=30.0)
    assert result.contraindicated is False


# ---------------------------------------------------------------------------
# Monitoring requirement
# ---------------------------------------------------------------------------


def test_monitoring_required_for_g3():
    result = calc(egfr_mL_per_min=50.0)  # < 60, so monitoring required
    assert result.monitoring_required is True


def test_monitoring_not_required_normal():
    result = calc(egfr_mL_per_min=100.0, fe_renal=0.1)
    assert result.monitoring_required is False


# ---------------------------------------------------------------------------
# Dose rounding
# ---------------------------------------------------------------------------


def test_dose_rounding():
    result = calc(egfr_mL_per_min=40.0, dose_rounding_mg=25.0)
    # recommended dose should be divisible by 25
    assert abs(result.recommended_dose_mg % 25.0) < 1e-9


# ---------------------------------------------------------------------------
# screen_renal_impairment
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_renal_impairment(DRUG, DOSE, INTERVAL, FE, [100, 60, 30, 15, 5])
    assert isinstance(results, list)


def test_screen_length_matches_egfr_values():
    egfr_values = [90, 70, 50, 30, 10]
    results = screen_renal_impairment(DRUG, DOSE, INTERVAL, FE, egfr_values)
    assert len(results) == len(egfr_values)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fe_renal_raises():
    with pytest.raises(ValueError):
        calc(fe_renal=1.5)


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        calc(baseline_dose_mg=0.0)
