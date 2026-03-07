"""Tests for clinical/plasma_level_monitoring.py — Phase 655."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.plasma_level_monitoring import (
    PlasmaMonitoringResult,
    design_monitoring_schedule,
    monitor_plasma_level,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_result(
    measured_conc_mg_L: float = 2.5,
    sample_time_h: float = 4.0,
    current_dose_mg: float = 100.0,
    target_range_low_mg_L: float = 1.0,
    target_range_high_mg_L: float = 5.0,
    **kwargs,
) -> PlasmaMonitoringResult:
    return monitor_plasma_level(
        drug_name="TestDrug",
        measured_conc_mg_L=measured_conc_mg_L,
        sample_time_h=sample_time_h,
        current_dose_mg=current_dose_mg,
        target_range_low_mg_L=target_range_low_mg_L,
        target_range_high_mg_L=target_range_high_mg_L,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: return type
# ---------------------------------------------------------------------------


def test_returns_plasma_monitoring_result():
    result = make_result()
    assert isinstance(result, PlasmaMonitoringResult)


# ---------------------------------------------------------------------------
# Test: status values
# ---------------------------------------------------------------------------


def test_status_in_valid_set():
    result = make_result()
    assert result.status in {"sub-therapeutic", "therapeutic", "supra-therapeutic"}


def test_status_therapeutic_when_conc_in_range():
    result = make_result(
        measured_conc_mg_L=2.5, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.status == "therapeutic"


def test_status_sub_therapeutic_when_conc_below_low():
    result = make_result(
        measured_conc_mg_L=0.5, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.status == "sub-therapeutic"


def test_status_supra_therapeutic_when_conc_above_high():
    result = make_result(
        measured_conc_mg_L=7.0, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.status == "supra-therapeutic"


# ---------------------------------------------------------------------------
# Test: dose adjustment factor
# ---------------------------------------------------------------------------


def test_dose_adjustment_factor_gt_1_when_sub_therapeutic():
    result = make_result(
        measured_conc_mg_L=0.4, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.dose_adjustment_factor > 1.0


def test_dose_adjustment_factor_lt_1_when_supra_therapeutic():
    result = make_result(
        measured_conc_mg_L=10.0, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.dose_adjustment_factor < 1.0


def test_dose_adjustment_factor_approx_1_when_therapeutic():
    # Measure exactly at mid-range: factor should be close to 1
    result = make_result(
        measured_conc_mg_L=3.0, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.dose_adjustment_factor == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# Test: recommended dose
# ---------------------------------------------------------------------------


def test_recommended_dose_mg_positive():
    result = make_result()
    assert result.recommended_dose_mg > 0


def test_recommended_dose_equals_current_times_factor():
    current_dose = 200.0
    result = make_result(measured_conc_mg_L=2.5, current_dose_mg=current_dose)
    expected = current_dose * result.dose_adjustment_factor
    assert result.recommended_dose_mg == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: probability therapeutic
# ---------------------------------------------------------------------------


def test_probability_therapeutic_in_0_1():
    result = make_result()
    assert 0.0 <= result.probability_therapeutic <= 1.0


def test_high_cv_gives_lower_probability_therapeutic():
    # With narrow target range, higher CV → lower probability
    low_cv_result = make_result(
        measured_conc_mg_L=3.0,
        target_range_low_mg_L=2.9,
        target_range_high_mg_L=3.1,
        cv_pct=5.0,
    )
    high_cv_result = make_result(
        measured_conc_mg_L=3.0,
        target_range_low_mg_L=2.9,
        target_range_high_mg_L=3.1,
        cv_pct=80.0,
    )
    assert low_cv_result.probability_therapeutic >= high_cv_result.probability_therapeutic


# ---------------------------------------------------------------------------
# Test: estimated PK parameters
# ---------------------------------------------------------------------------


def test_estimated_cmax_positive():
    result = make_result()
    assert result.estimated_cmax_mg_L > 0


def test_estimated_ke_positive():
    result = make_result()
    assert result.estimated_ke_per_h > 0


def test_next_sample_time_positive():
    result = make_result()
    assert result.next_sample_time_h > 0


# ---------------------------------------------------------------------------
# Test: percent deviation
# ---------------------------------------------------------------------------


def test_percent_deviation_positive_when_supra_therapeutic():
    result = make_result(
        measured_conc_mg_L=8.0, target_range_low_mg_L=1.0, target_range_high_mg_L=5.0
    )
    assert result.percent_deviation > 0


# ---------------------------------------------------------------------------
# Test: notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = make_result()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Test: design_monitoring_schedule
# ---------------------------------------------------------------------------


def test_design_monitoring_schedule_returns_correct_length():
    schedule = design_monitoring_schedule(
        "TestDrug", t_half_h=8.0, dosing_interval_h=12.0, n_samples=3
    )
    assert len(schedule) == 3


def test_design_monitoring_schedule_each_entry_has_time_h():
    schedule = design_monitoring_schedule(
        "TestDrug", t_half_h=6.0, dosing_interval_h=12.0, n_samples=4
    )
    for entry in schedule:
        assert "time_h" in entry


def test_design_monitoring_schedule_n_samples_1():
    schedule = design_monitoring_schedule(
        "TestDrug", t_half_h=8.0, dosing_interval_h=24.0, n_samples=1
    )
    assert len(schedule) == 1
    assert "time_h" in schedule[0]


# ---------------------------------------------------------------------------
# Test: validation errors
# ---------------------------------------------------------------------------


def test_raises_error_measured_conc_negative():
    with pytest.raises(ValueError):
        monitor_plasma_level(
            drug_name="X",
            measured_conc_mg_L=-1.0,
            sample_time_h=4.0,
            current_dose_mg=100.0,
        )


def test_raises_error_sample_time_zero():
    with pytest.raises(ValueError):
        monitor_plasma_level(
            drug_name="X",
            measured_conc_mg_L=2.0,
            sample_time_h=0.0,
            current_dose_mg=100.0,
        )


def test_raises_error_target_low_ge_target_high():
    with pytest.raises(ValueError):
        monitor_plasma_level(
            drug_name="X",
            measured_conc_mg_L=2.0,
            sample_time_h=4.0,
            current_dose_mg=100.0,
            target_range_low_mg_L=5.0,
            target_range_high_mg_L=5.0,
        )
