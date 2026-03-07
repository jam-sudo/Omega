"""Tests for clinical/drug_holiday_simulation.py — Phase 635."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_holiday_simulation import (
    DrugHolidayResult,
    simulate_drug_holiday,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> DrugHolidayResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        dosing_interval_h=12.0,
        n_doses_before=10,
        holiday_duration_h=48.0,
        n_doses_after=5,
        mec_mg_L=0.5,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_drug_holiday(**defaults)


# ---------------------------------------------------------------------------
# Return type and field types
# ---------------------------------------------------------------------------


def test_returns_drug_holiday_result():
    r = _default_result()
    assert isinstance(r, DrugHolidayResult)


def test_field_drug_name_str():
    r = _default_result()
    assert isinstance(r.drug_name, str)
    assert r.drug_name == "TestDrug"


def test_field_dose_mg_float():
    r = _default_result()
    assert isinstance(r.dose_mg, float)


def test_field_times_h_list():
    r = _default_result()
    assert isinstance(r.times_h, list)


def test_field_c_plasma_list():
    r = _default_result()
    assert isinstance(r.c_plasma_mg_L, list)


def test_field_css_before_float():
    r = _default_result()
    assert isinstance(r.css_before_mg_L, float)


def test_field_notes_str():
    r = _default_result()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_len_times_equals_len_c_plasma():
    r = _default_result()
    assert len(r.times_h) == len(r.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Core holiday behaviour
# ---------------------------------------------------------------------------


def test_c_nadir_less_than_css_before():
    """Concentration drops during holiday."""
    r = _default_result()
    assert r.c_nadir_mg_L < r.css_before_mg_L


def test_time_to_nadir_positive():
    r = _default_result()
    assert r.time_to_nadir_h > 0


def test_c_at_holiday_end_nonnegative():
    r = _default_result()
    assert r.c_at_holiday_end_mg_L >= 0.0


def test_time_in_subtherapeutic_nonnegative():
    r = _default_result()
    assert r.time_in_subtherapeutic_h >= 0.0


def test_therapeutic_window_restored_nonnegative():
    r = _default_result()
    assert r.therapeutic_window_restored_h >= 0.0


def test_css_before_positive():
    r = _default_result()
    assert r.css_before_mg_L > 0.0


def test_rebound_cmax_positive():
    r = _default_result()
    assert r.rebound_cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# Comparative tests
# ---------------------------------------------------------------------------


def test_short_holiday_smaller_drop_than_long_holiday():
    """Short holiday: c_nadir higher (less drop) vs long holiday."""
    r_short = _default_result(holiday_duration_h=6.0)
    r_long = _default_result(holiday_duration_h=72.0)
    assert r_short.c_nadir_mg_L > r_long.c_nadir_mg_L


def test_longer_halflife_slower_drop():
    """Longer half-life drug retains more drug at holiday end."""
    r_slow = _default_result(cl_L_per_h=0.5)  # t_half ~ 69h
    r_fast = _default_result(cl_L_per_h=20.0)  # t_half ~ 1.7h
    assert r_slow.c_at_holiday_end_mg_L > r_fast.c_at_holiday_end_mg_L


def test_shorter_halflife_faster_drop():
    """Faster clearance → lower nadir during holiday."""
    r_fast = _default_result(cl_L_per_h=20.0, n_doses_before=15)
    r_slow = _default_result(cl_L_per_h=1.0, n_doses_before=15)
    assert r_fast.c_nadir_mg_L < r_slow.c_nadir_mg_L


def test_double_dose_higher_css_before():
    r1 = _default_result(dose_mg=100.0)
    r2 = _default_result(dose_mg=200.0)
    assert r2.css_before_mg_L > r1.css_before_mg_L


def test_double_dose_higher_rebound_cmax():
    r1 = _default_result(dose_mg=100.0)
    r2 = _default_result(dose_mg=200.0)
    assert r2.rebound_cmax_mg_L > r1.rebound_cmax_mg_L


def test_longer_holiday_lower_c_at_end():
    r_short = _default_result(holiday_duration_h=6.0)
    r_long = _default_result(holiday_duration_h=72.0)
    assert r_long.c_at_holiday_end_mg_L < r_short.c_at_holiday_end_mg_L


def test_zero_mec_no_subtherapeutic_time():
    """MEC=0 means concentration is always >= MEC, so subtherapeutic time = 0."""
    r = _default_result(mec_mg_L=0.0)
    assert r.time_in_subtherapeutic_h == 0.0


def test_high_mec_more_subtherapeutic_time():
    """Higher MEC → more time below threshold."""
    r_low = _default_result(mec_mg_L=0.01)
    r_high = _default_result(mec_mg_L=5.0)
    assert r_high.time_in_subtherapeutic_h > r_low.time_in_subtherapeutic_h


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday("Drug", dose_mg=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_drug_holiday("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_holiday_duration_zero():
    with pytest.raises(ValueError, match="holiday_duration_h"):
        simulate_drug_holiday("Drug", dose_mg=100.0, holiday_duration_h=0.0)


def test_validation_n_doses_before_zero():
    with pytest.raises(ValueError, match="n_doses_before"):
        simulate_drug_holiday("Drug", dose_mg=100.0, n_doses_before=0)


def test_validation_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_drug_holiday("Drug", dose_mg=100.0, f_oral=1.5)
