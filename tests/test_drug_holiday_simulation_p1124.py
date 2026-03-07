"""Tests for clinical/drug_holiday_simulation_p1124.py — Phase 1124."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.drug_holiday_simulation_p1124 import (
    DrugHolidayResult,
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
        cl_L_per_h=5.0,
        vd_L=50.0,
        holiday_duration_h=48.0,
        n_pre_holiday_doses=10,
        n_post_holiday_doses=10,
        ka_per_h=1.0,
        f_bioavail=1.0,
    )
    defaults.update(kwargs)
    return simulate_drug_holiday(**defaults)


# ---------------------------------------------------------------------------
# Return type and field types
# ---------------------------------------------------------------------------


def test_returns_drug_holiday_result():
    r = _default()
    assert isinstance(r, DrugHolidayResult)


def test_drug_name_preserved():
    r = _default(drug_name="Bisphosphonate")
    assert r.drug_name == "Bisphosphonate"


def test_times_h_is_list():
    r = _default()
    assert isinstance(r.times_h, list)


def test_c_plasma_is_list():
    r = _default()
    assert isinstance(r.c_plasma_mg_L, list)


def test_times_and_plasma_same_length():
    r = _default()
    assert len(r.times_h) == len(r.c_plasma_mg_L)


def test_notes_nonempty_str():
    r = _default()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_t_half_positive():
    r = _default()
    assert r.t_half_h > 0.0


def test_t_half_from_cl_vd():
    """t_half = 0.693 * Vd / CL."""
    r = _default(cl_L_per_h=5.0, vd_L=50.0)
    expected = math.log(2) * 50.0 / 5.0
    assert r.t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_css_avg_positive():
    r = _default()
    assert r.css_avg_mg_L > 0.0


def test_c_at_holiday_end_nonnegative():
    r = _default()
    assert r.c_at_holiday_end_mg_L >= 0.0


def test_c_percent_remaining_in_range():
    """c_percent_remaining should be in [0, 150] (some accumulation possible)."""
    r = _default()
    assert 0.0 <= r.c_percent_remaining_pct <= 150.0


def test_c_percent_remaining_formula():
    r = _default()
    if r.css_avg_mg_L > 0:
        expected = r.c_at_holiday_end_mg_L / r.css_avg_mg_L * 100.0
        assert r.c_percent_remaining_pct == pytest.approx(expected, rel=1e-6)


def test_long_holiday_lower_c_at_end():
    """Long holiday → concentration decays more."""
    r_short = _default(holiday_duration_h=6.0)
    r_long = _default(holiday_duration_h=200.0)
    assert r_long.c_at_holiday_end_mg_L < r_short.c_at_holiday_end_mg_L


def test_short_holiday_high_percent_remaining():
    """Short holiday relative to t½ → >50% remaining."""
    r = _default(cl_L_per_h=1.0, vd_L=50.0, holiday_duration_h=2.0)
    assert r.c_percent_remaining_pct > 50.0


def test_very_long_holiday_approaches_zero():
    """After many half-lives, almost nothing remains."""
    r = _default(cl_L_per_h=5.0, vd_L=10.0, holiday_duration_h=200.0)
    assert r.c_percent_remaining_pct < 5.0


def test_higher_dose_higher_css():
    r1 = _default(dose_mg=50.0)
    r2 = _default(dose_mg=200.0)
    assert r2.css_avg_mg_L > r1.css_avg_mg_L


def test_higher_clearance_lower_css():
    r_low_cl = _default(cl_L_per_h=1.0)
    r_high_cl = _default(cl_L_per_h=20.0)
    assert r_low_cl.css_avg_mg_L > r_high_cl.css_avg_mg_L


def test_doses_to_resteady_state_positive_int():
    r = _default()
    assert isinstance(r.doses_to_resteady_state, int)
    assert r.doses_to_resteady_state >= 1


def test_c_plasma_all_nonnegative():
    r = _default()
    assert all(c >= 0.0 for c in r.c_plasma_mg_L)


def test_times_h_starts_at_zero():
    r = _default()
    assert r.times_h[0] == pytest.approx(0.0, abs=1e-9)


def test_longer_holiday_lower_percent_remaining():
    r_short = _default(holiday_duration_h=12.0)
    r_long = _default(holiday_duration_h=96.0)
    assert r_long.c_percent_remaining_pct < r_short.c_percent_remaining_pct


def test_bioavailability_scales_css():
    r_full = _default(f_bioavail=1.0)
    r_half = _default(f_bioavail=0.5)
    assert r_full.css_avg_mg_L == pytest.approx(r_half.css_avg_mg_L * 2, rel=0.05)


def test_holiday_duration_stored():
    r = _default(holiday_duration_h=72.0)
    assert r.holiday_duration_h == pytest.approx(72.0)


def test_dose_mg_stored():
    r = _default(dose_mg=250.0)
    assert r.dose_mg == pytest.approx(250.0)


def test_dosing_interval_stored():
    r = _default(dosing_interval_h=24.0)
    assert r.dosing_interval_h == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday(
            "Drug", dose_mg=0.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=48.0,
        )


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_drug_holiday(
            "Drug", dose_mg=-10.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=48.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=12.0,
            cl_L_per_h=0.0, vd_L=50.0, holiday_duration_h=48.0,
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=0.0, holiday_duration_h=48.0,
        )


def test_validation_holiday_zero():
    with pytest.raises(ValueError, match="holiday_duration_h"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=0.0,
        )


def test_validation_interval_zero():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=0.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=48.0,
        )


def test_validation_f_bioavail_above_one():
    with pytest.raises(ValueError, match="f_bioavail"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=48.0,
            f_bioavail=1.5,
        )


def test_validation_f_bioavail_zero():
    with pytest.raises(ValueError, match="f_bioavail"):
        simulate_drug_holiday(
            "Drug", dose_mg=100.0, dosing_interval_h=12.0,
            cl_L_per_h=5.0, vd_L=50.0, holiday_duration_h=48.0,
            f_bioavail=0.0,
        )
