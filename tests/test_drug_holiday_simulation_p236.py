"""Tests for clinical/drug_holiday_simulation_p236.py — Phase 236."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_holiday_simulation_p236 import (
    DrugHolidayResult,
    optimize_holiday_duration,
    simulate_drug_holiday,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> DrugHolidayResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        dosing_interval_h=12.0,
        holiday_duration_h=48.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_oral=0.8,
        mic_threshold_mg_L=0.5,
    )
    defaults.update(kwargs)
    return simulate_drug_holiday(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_drug_holiday_result():
    r = _default()
    assert isinstance(r, DrugHolidayResult)


def test_result_is_frozen():
    r = _default()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field types and basic ranges
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    r = _default(drug_name="Aspirin")
    assert r.drug_name == "Aspirin"


def test_washout_fraction_in_range():
    r = _default()
    assert 0.0 < r.washout_fraction <= 1.0


def test_washout_fraction_less_than_one():
    """Any non-zero holiday should reduce concentration."""
    r = _default(holiday_duration_h=1.0)
    assert r.washout_fraction < 1.0


def test_c_at_restart_positive():
    r = _default()
    assert r.c_at_restart_mg_L > 0.0


def test_c_at_restart_less_than_css():
    r = _default()
    assert r.c_at_restart_mg_L < r.css_before_holiday_mg_L


def test_css_before_positive():
    r = _default()
    assert r.css_before_holiday_mg_L > 0.0


def test_time_below_mic_nonnegative():
    r = _default()
    assert r.time_below_mic_h >= 0.0


def test_rebound_auc_positive():
    r = _default()
    assert r.rebound_auc_mg_h_per_L > 0.0


def test_notes_is_str():
    r = _default()
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Holiday duration effects
# ---------------------------------------------------------------------------


def test_longer_holiday_lower_washout_fraction():
    r_short = _default(holiday_duration_h=6.0)
    r_long = _default(holiday_duration_h=96.0)
    assert r_long.washout_fraction < r_short.washout_fraction


def test_longer_holiday_lower_c_at_restart():
    r_short = _default(holiday_duration_h=6.0)
    r_long = _default(holiday_duration_h=96.0)
    assert r_long.c_at_restart_mg_L < r_short.c_at_restart_mg_L


def test_higher_clearance_faster_washout():
    r_fast = _default(cl_L_per_h=20.0)
    r_slow = _default(cl_L_per_h=1.0)
    assert r_fast.washout_fraction < r_slow.washout_fraction


# ---------------------------------------------------------------------------
# restart_recommended logic
# ---------------------------------------------------------------------------


def test_restart_recommended_when_washout_below_20pct():
    """Very long holiday with fast clearance should achieve full washout."""
    r = _default(holiday_duration_h=200.0, cl_L_per_h=10.0)
    # washout_fraction should be tiny → restart_recommended = True
    assert r.washout_fraction < 0.2
    assert r.restart_recommended is True


def test_restart_not_recommended_when_washout_above_20pct():
    """Short holiday with slow clearance → washout_fraction >= 0.2 → False."""
    r = _default(holiday_duration_h=1.0, cl_L_per_h=0.5)
    assert r.washout_fraction >= 0.2
    assert r.restart_recommended is False


def test_restart_recommended_is_bool():
    r = _default()
    assert isinstance(r.restart_recommended, bool)


# ---------------------------------------------------------------------------
# MIC threshold effects
# ---------------------------------------------------------------------------


def test_high_mic_more_time_below():
    r_low = _default(mic_threshold_mg_L=0.001)
    r_high = _default(mic_threshold_mg_L=10.0)
    assert r_high.time_below_mic_h > r_low.time_below_mic_h


def test_zero_mic_no_time_below():
    r = _default(mic_threshold_mg_L=0.0)
    assert r.time_below_mic_h == 0.0


# ---------------------------------------------------------------------------
# optimize_holiday_duration
# ---------------------------------------------------------------------------


def test_optimize_returns_dict():
    result = optimize_holiday_duration(
        "Drug", dose_mg=100.0, dosing_interval_h=12.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert isinstance(result, dict)


def test_optimize_dict_has_required_keys():
    result = optimize_holiday_duration(
        "Drug", dose_mg=100.0, dosing_interval_h=12.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert "optimal_holiday_h" in result
    assert "expected_washout_fraction" in result
    assert "time_below_mic_h" in result
    assert "achievable" in result


def test_optimize_achievable_true():
    result = optimize_holiday_duration(
        "Drug",
        dose_mg=100.0,
        dosing_interval_h=12.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        target_washout_fraction=0.1,
    )
    assert result["achievable"] is True


def test_optimize_expected_washout_near_target():
    target = 0.1
    result = optimize_holiday_duration(
        "Drug",
        dose_mg=100.0,
        dosing_interval_h=12.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        target_washout_fraction=target,
    )
    assert abs(result["expected_washout_fraction"] - target) < 0.01


def test_optimize_time_below_mic_nonnegative():
    result = optimize_holiday_duration(
        "Drug", dose_mg=100.0, dosing_interval_h=12.0, cl_L_per_h=5.0, vd_L=50.0
    )
    assert result["time_below_mic_h"] >= 0.0


def test_optimize_optimal_holiday_positive():
    result = optimize_holiday_duration(
        "Drug",
        dose_mg=100.0,
        dosing_interval_h=12.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        target_washout_fraction=0.05,
    )
    assert result["optimal_holiday_h"] > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday(
            "D",
            dose_mg=0.0,
            dosing_interval_h=12.0,
            holiday_duration_h=48.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_dosing_interval_zero():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=0.0,
            holiday_duration_h=48.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_holiday_duration_zero():
    with pytest.raises(ValueError, match="holiday_duration_h"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=12.0,
            holiday_duration_h=0.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=12.0,
            holiday_duration_h=48.0,
            cl_L_per_h=0.0,
            vd_L=50.0,
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=12.0,
            holiday_duration_h=48.0,
            cl_L_per_h=5.0,
            vd_L=0.0,
        )


def test_validation_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=12.0,
            holiday_duration_h=48.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            f_oral=1.5,
        )


def test_validation_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_drug_holiday(
            "D",
            dose_mg=100.0,
            dosing_interval_h=12.0,
            holiday_duration_h=48.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            f_oral=0.0,
        )
