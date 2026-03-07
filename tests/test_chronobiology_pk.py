"""
Tests for Phase 746 — Chronobiology Effect on PK.
"""

import math

import pytest

from omega_pbpk.core.chronobiology_pk import (
    ChronobiologyPKResult,
    compare_dosing_times,
    simulate_chronobiology_pk,
)

# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_chronobiology_pk_result():
    result = simulate_chronobiology_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, ChronobiologyPKResult)


def test_cmax_positive():
    result = simulate_chronobiology_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_chronobiology_pk("TestDrug", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_drug_name_preserved():
    result = simulate_chronobiology_pk("Midazolam", dose_mg=5.0)
    assert result.drug_name == "Midazolam"


def test_dose_preserved():
    result = simulate_chronobiology_pk("Drug", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_dosing_clock_h_preserved():
    result = simulate_chronobiology_pk("Drug", dose_mg=50.0, dosing_clock_h=20.0)
    assert result.dosing_clock_h == 20.0


def test_tmax_within_simulation_time():
    result = simulate_chronobiology_pk("Drug", dose_mg=50.0, t_end_h=24.0)
    assert 0.0 <= result.tmax_h <= 24.0


# ---------------------------------------------------------------------------
# Circadian effect on AUC
# ---------------------------------------------------------------------------


def test_evening_dosing_higher_auc_than_morning_high_cyp3a4():
    """Evening dose (20:00) should give higher AUC than morning (08:00) for
    high CYP3A4 substrate (fm_cyp3a4=0.8) because CYP3A4 activity is lower at night.
    """
    morning = simulate_chronobiology_pk(
        "HighCYP3A4Drug",
        dose_mg=100.0,
        dosing_clock_h=8.0,
        fm_cyp3a4=0.8,
        fm_renal=0.1,
        t_end_h=24.0,
    )
    evening = simulate_chronobiology_pk(
        "HighCYP3A4Drug",
        dose_mg=100.0,
        dosing_clock_h=20.0,
        fm_cyp3a4=0.8,
        fm_renal=0.1,
        t_end_h=24.0,
    )
    assert evening.auc_mg_h_per_L > morning.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# compare_dosing_times
# ---------------------------------------------------------------------------


def test_compare_dosing_times_returns_correct_length():
    clock_hours = [0.0, 6.0, 12.0, 18.0]
    results = compare_dosing_times("Drug", dose_mg=100.0, clock_hours=clock_hours)
    assert len(results) == len(clock_hours)


def test_compare_dosing_times_all_results_are_correct_type():
    clock_hours = [8.0, 20.0]
    results = compare_dosing_times("Drug", dose_mg=100.0, clock_hours=clock_hours)
    for r in results:
        assert isinstance(r, ChronobiologyPKResult)


def test_compare_dosing_times_empty_list():
    results = compare_dosing_times("Drug", dose_mg=100.0, clock_hours=[])
    assert results == []


# ---------------------------------------------------------------------------
# optimal_dosing_window
# ---------------------------------------------------------------------------


def test_optimal_dosing_window_is_str():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0, fm_cyp3a4=0.7)
    assert isinstance(result.optimal_dosing_window, str)


def test_optimal_dosing_window_high_cyp3a4():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0, fm_cyp3a4=0.7, fm_renal=0.1)
    assert "evening" in result.optimal_dosing_window.lower()


def test_optimal_dosing_window_low_cyp3a4():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0, fm_cyp3a4=0.2, fm_renal=0.3)
    assert "minimal" in result.optimal_dosing_window.lower()


# ---------------------------------------------------------------------------
# t_half_avg_h and cl_at_tmax
# ---------------------------------------------------------------------------


def test_t_half_avg_positive():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0)
    assert result.t_half_avg_h > 0.0


def test_t_half_avg_matches_formula():
    """t½ ≈ ln(2) × Vd / CL_base."""
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0, cl_base_L_per_h=10.0, vd_L=50.0)
    expected = math.log(2.0) * 50.0 / 10.0
    assert abs(result.t_half_avg_h - expected) < 1e-9


def test_cl_at_tmax_positive():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0)
    assert result.cl_at_tmax > 0.0


# ---------------------------------------------------------------------------
# No circadian variation when fm_cyp3a4=0 and fm_renal=0
# ---------------------------------------------------------------------------


def test_no_circadian_variation_when_fm_zero():
    """With fm_cyp3a4=0 and fm_renal=0, CL is constant; AUC must be identical
    for all dosing times (within numerical precision).
    """
    r_morning = simulate_chronobiology_pk(
        "Drug",
        dose_mg=100.0,
        dosing_clock_h=8.0,
        fm_cyp3a4=0.0,
        fm_renal=0.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    r_evening = simulate_chronobiology_pk(
        "Drug",
        dose_mg=100.0,
        dosing_clock_h=20.0,
        fm_cyp3a4=0.0,
        fm_renal=0.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    # AUC should be virtually identical (same CL throughout)
    rel_diff = abs(r_morning.auc_mg_h_per_L - r_evening.auc_mg_h_per_L) / max(
        r_morning.auc_mg_h_per_L, 1e-12
    )
    assert rel_diff < 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validate_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=0.0)


def test_validate_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=-1.0)


def test_validate_dosing_clock_h_24_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=100.0, dosing_clock_h=24.0)


def test_validate_dosing_clock_h_negative_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=100.0, dosing_clock_h=-1.0)


def test_validate_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=100.0, cl_base_L_per_h=0.0)


def test_validate_vd_zero_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=100.0, vd_L=0.0)


def test_validate_fm_sum_exceeds_one_raises():
    with pytest.raises(ValueError):
        simulate_chronobiology_pk("Drug", dose_mg=100.0, fm_cyp3a4=0.7, fm_renal=0.5)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_chronobiology_pk("Drug", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
