"""Tests for circadian dosing optimizer (Phase 334)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.circadian_dosing import (
    CircadianDosingResult,
    compare_dosing_schedules,
    optimize_dosing_time,
)


# ── Input validation ──────────────────────────────────────────────────────────


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=0.0, vd_L=50.0, dose_mg=100.0)


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=-5.0, vd_L=50.0, dose_mg=100.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=0.0, dose_mg=100.0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=-10.0, dose_mg=100.0)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=-100.0)


def test_invalid_objective_raises():
    with pytest.raises(ValueError, match="objective must be one of"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, objective="wrong")


def test_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_per_h must be > 0"):
        optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, ka_per_h=0.0)


# ── Return type and structure ─────────────────────────────────────────────────


def test_returns_circadian_dosing_result():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert isinstance(result, CircadianDosingResult)


def test_optimal_dosing_hour_in_valid_range():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert 0 <= result.optimal_dosing_hour <= 23


def test_all_hours_auc_fraction_has_24_elements():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert len(result.all_hours_auc_fraction) == 24


def test_drug_name_matches_input():
    result = optimize_dosing_time("Warfarin", cl_L_per_h=2.0, vd_L=10.0, dose_mg=5.0)
    assert result.drug_name == "Warfarin"


def test_objective_stored_in_result():
    for obj in ["max_efficacy", "min_toxicity", "balanced"]:
        result = optimize_dosing_time(
            "Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, objective=obj
        )
        assert result.objective == obj


# ── Peak and trough ───────────────────────────────────────────────────────────


def test_peak_positive():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert result.peak_at_optimal_h > 0.0


def test_trough_positive():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert result.trough_at_optimal_h > 0.0


def test_peak_greater_than_trough():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert result.peak_at_optimal_h >= result.trough_at_optimal_h


# ── Objective-specific behavior ───────────────────────────────────────────────


def test_max_efficacy_near_efficacy_peak():
    """Optimal hour for max_efficacy should be within ±8h of efficacy_peak_h."""
    efficacy_peak = 10.0
    result = optimize_dosing_time(
        "Drug",
        cl_L_per_h=5.0,
        vd_L=50.0,
        dose_mg=100.0,
        efficacy_peak_h=efficacy_peak,
        objective="max_efficacy",
    )
    # The optimal hour should shift drug peak toward efficacy window
    # Check that optimal hour is within ±6h (circular) of efficacy_peak
    diff = abs(result.optimal_dosing_hour - efficacy_peak)
    circular_diff = min(diff, 24 - diff)
    assert circular_diff <= 6.0, (
        f"Optimal hour {result.optimal_dosing_hour} not aligned with efficacy peak {efficacy_peak}"
    )


def test_min_toxicity_not_at_toxicity_peak():
    """Optimal hour for min_toxicity should not be at the toxicity peak itself."""
    toxicity_peak = 2.0
    result = optimize_dosing_time(
        "Drug",
        cl_L_per_h=5.0,
        vd_L=50.0,
        dose_mg=100.0,
        toxicity_peak_h=toxicity_peak,
        objective="min_toxicity",
    )
    # Optimal hour should differ from the exact toxicity peak hour
    assert result.optimal_dosing_hour != toxicity_peak, (
        f"Optimal hour {result.optimal_dosing_hour} should not equal toxicity peak {toxicity_peak}"
    )


def test_balanced_objective_works():
    result = optimize_dosing_time(
        "Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, objective="balanced"
    )
    assert result.optimal_dosing_hour in range(24)


# ── Notes ─────────────────────────────────────────────────────────────────────


def test_notes_contains_drug_name():
    result = optimize_dosing_time("SpecialDrug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert "SpecialDrug" in result.notes


def test_rationale_nonempty():
    result = optimize_dosing_time("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert len(result.rationale) > 0


# ── compare_dosing_schedules ──────────────────────────────────────────────────


def test_compare_dosing_schedules_returns_correct_length():
    results = compare_dosing_schedules(
        "Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, hours=[0, 6, 12, 18]
    )
    assert len(results) == 4


def test_compare_dosing_schedules_default_hours_length():
    results = compare_dosing_schedules("Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0)
    assert len(results) == 4


def test_compare_dosing_schedules_sorted_descending():
    results = compare_dosing_schedules(
        "Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, hours=[0, 6, 12, 18]
    )
    fracs = [r.optimal_auc_fraction for r in results]
    assert fracs == sorted(fracs, reverse=True)


def test_compare_dosing_schedules_drug_name_correct():
    results = compare_dosing_schedules(
        "MetforminX", cl_L_per_h=5.0, vd_L=50.0, dose_mg=500.0, hours=[0, 12]
    )
    for r in results:
        assert r.drug_name == "MetforminX"


def test_compare_dosing_schedules_all_hours_auc_has_24_elements():
    results = compare_dosing_schedules(
        "Drug", cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, hours=[6, 18]
    )
    for r in results:
        assert len(r.all_hours_auc_fraction) == 24
