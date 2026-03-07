"""Tests for clinical/drug_holiday.py — Phase 266."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_holiday import (
    DrugHolidayResult,
    simulate_drug_holiday,
)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday("Drug", dose_mg=0, cl_L_per_h=5, vd_L=50)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday("Drug", dose_mg=-10, cl_L_per_h=5, vd_L=50)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=0, vd_L=50)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=-1, vd_L=50)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=0)


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=-10)


def test_invalid_dosing_interval_zero():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, dosing_interval_h=0)


def test_invalid_n_doses_before_zero():
    with pytest.raises(ValueError, match="n_doses_before"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, n_doses_before=0)


def test_invalid_holiday_duration_zero():
    with pytest.raises(ValueError, match="holiday_duration_h"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, holiday_duration_h=0)


def test_invalid_n_doses_after_zero():
    with pytest.raises(ValueError, match="n_doses_after"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, n_doses_after=0)


def test_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, ka_per_h=0)


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, f_oral=0)


def test_invalid_f_oral_above_1():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_drug_holiday("Drug", dose_mg=100, cl_L_per_h=5, vd_L=50, f_oral=1.5)


# ---------------------------------------------------------------------------
# Basic result properties
# ---------------------------------------------------------------------------


def _default_result() -> DrugHolidayResult:
    return simulate_drug_holiday(
        "TestDrug",
        dose_mg=100,
        cl_L_per_h=5,
        vd_L=50,
        dosing_interval_h=12,
        n_doses_before=10,
        holiday_duration_h=48,
        n_doses_after=5,
        ka_per_h=1.0,
        f_oral=1.0,
    )


def test_returns_drug_holiday_result():
    r = _default_result()
    assert isinstance(r, DrugHolidayResult)


def test_drug_name_stored():
    r = _default_result()
    assert r.drug_name == "TestDrug"


def test_times_and_concentrations_same_length():
    r = _default_result()
    assert len(r.times_h) == len(r.c_plasma_mg_L)


def test_c_at_holiday_start_positive():
    """After 10 doses, concentration should be near steady state (> 0)."""
    r = _default_result()
    assert r.c_at_holiday_start > 0


def test_c_at_holiday_end_less_than_start():
    """Drug washes out during holiday."""
    r = _default_result()
    assert r.c_at_holiday_end < r.c_at_holiday_start


def test_c_min_during_holiday_nonnegative():
    r = _default_result()
    assert r.c_min_during_holiday >= 0


def test_time_below_mic_nonnegative():
    r = _default_result()
    assert r.time_below_mic_h >= 0


def test_washout_90pct_positive():
    r = _default_result()
    assert r.washout_90pct_h > 0


def test_n_missed_doses_nonnegative():
    r = _default_result()
    assert r.n_missed_doses >= 0


def test_n_missed_doses_value():
    """48h holiday with 12h interval -> 4 missed doses."""
    r = _default_result()
    assert r.n_missed_doses == 4


# ---------------------------------------------------------------------------
# Washout kinetics
# ---------------------------------------------------------------------------


def test_long_halflife_slow_washout():
    """Low CL (long t1/2) should result in slower washout."""
    r_slow = simulate_drug_holiday(
        "SlowDrug",
        dose_mg=100,
        cl_L_per_h=0.5,  # long t1/2
        vd_L=50,
        dosing_interval_h=12,
        n_doses_before=20,
        holiday_duration_h=96,
        n_doses_after=3,
    )
    r_fast = simulate_drug_holiday(
        "FastDrug",
        dose_mg=100,
        cl_L_per_h=20,  # short t1/2
        vd_L=50,
        dosing_interval_h=12,
        n_doses_before=20,
        holiday_duration_h=96,
        n_doses_after=3,
    )
    # c_at_holiday_end should be higher for slow-clearance drug
    assert r_slow.c_at_holiday_end > r_fast.c_at_holiday_end


def test_short_halflife_faster_washout():
    """High CL (short t1/2): washout_90pct should be shorter than holiday."""
    r = simulate_drug_holiday(
        "FastDrug",
        dose_mg=100,
        cl_L_per_h=50,  # very high CL, ke very large
        vd_L=50,
        dosing_interval_h=12,
        n_doses_before=10,
        holiday_duration_h=48,
        n_doses_after=3,
    )
    # Should reach 90% washout well before end of holiday
    assert r.washout_90pct_h < 48


def test_concentrations_all_nonnegative():
    r = _default_result()
    assert all(c >= 0 for c in r.c_plasma_mg_L)


def test_holiday_start_matches_param():
    r = simulate_drug_holiday(
        "Drug",
        dose_mg=100,
        cl_L_per_h=5,
        vd_L=50,
        dosing_interval_h=12,
        n_doses_before=8,
        holiday_duration_h=24,
        n_doses_after=3,
    )
    assert r.holiday_start_h == pytest.approx(8 * 12)
