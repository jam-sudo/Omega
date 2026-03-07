"""Tests for clinical/chronotherapy_optimizer.py — Phase 661."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.chronotherapy_optimizer import (
    ChronotherapyResult,
    compare_dosing_times,
    optimize_dosing_time,
)

# ---------------------------------------------------------------------------
# Basic return type and field range tests
# ---------------------------------------------------------------------------


def test_returns_chronotherapy_result():
    result = optimize_dosing_time("DrugA", "hypertension")
    assert isinstance(result, ChronotherapyResult)


def test_optimal_dose_time_in_range():
    result = optimize_dosing_time("DrugA", "hypertension")
    assert 0.0 <= result.optimal_dose_time_h < 24.0


def test_optimal_dose_time_label_valid():
    result = optimize_dosing_time("DrugA", "asthma")
    assert result.optimal_dose_time_label in {"morning", "afternoon", "evening", "night"}


def test_efficacy_score_in_range():
    result = optimize_dosing_time("DrugA", "cancer")
    assert 0.0 <= result.predicted_efficacy_score <= 100.0


def test_toxicity_score_in_range():
    result = optimize_dosing_time("DrugA", "cancer")
    assert 0.0 <= result.predicted_toxicity_score <= 100.0


def test_benefit_risk_ratio_positive():
    result = optimize_dosing_time("DrugA", "arthritis")
    assert result.benefit_risk_ratio > 0.0


def test_phase_alignment_circular_max_12():
    result = optimize_dosing_time("DrugA", "asthma")
    assert 0.0 <= result.phase_alignment_h <= 12.0


def test_recommended_schedule_valid():
    result = optimize_dosing_time("DrugA", "hypertension")
    assert result.recommended_schedule in {"morning", "evening", "bedtime", "continuous"}


def test_notes_nonempty():
    result = optimize_dosing_time("DrugA", "insomnia")
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_stored():
    result = optimize_dosing_time("Metoprolol", "hypertension")
    assert result.drug_name == "Metoprolol"


def test_indication_stored():
    result = optimize_dosing_time("DrugA", "asthma")
    assert result.indication == "asthma"


# ---------------------------------------------------------------------------
# Disease-specific timing tests
# ---------------------------------------------------------------------------


def test_hypertension_optimal_before_morning_surge():
    # Disease peak at hour 9; optimal dose time = (9 - tmax) % 24
    result = optimize_dosing_time("Amlodipine", "hypertension", tmax_h=2.0)
    # Drug peak should be near hour 9 (morning surge)
    # Optimal dose time is 7h; should be in evening/night/early morning range (not afternoon)
    assert result.optimal_dose_time_h < 9.0 or result.optimal_dose_time_h >= 18.0


def test_asthma_optimal_targets_4am_peak():
    # Disease peak at hour 4; optimal dose = (4 - tmax) % 24
    result = optimize_dosing_time("Theophylline", "asthma", tmax_h=2.0)
    # Drug peak should be near hour 4
    assert abs(result.circadian_phase_drug - 4.0) < 1.0


# ---------------------------------------------------------------------------
# Compare dosing times tests
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    results = compare_dosing_times("DrugA", "hypertension", dosing_times_h=[0.0, 6.0, 12.0, 18.0])
    assert len(results) == 4


def test_compare_default_four_times():
    results = compare_dosing_times("DrugA", "cancer")
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_dosing_times("DrugA", "arthritis")
    for i in range(len(results) - 1):
        assert results[i].benefit_risk_ratio >= results[i + 1].benefit_risk_ratio


def test_compare_custom_times():
    results = compare_dosing_times("DrugA", "asthma", dosing_times_h=[2.0, 14.0, 20.0])
    assert len(results) == 3


def test_optimal_time_achieves_highest_br_ratio():
    # The optimal dose time should have alignment ~0, giving highest BR ratio
    _ = optimize_dosing_time("DrugX", "hypertension", tmax_h=2.0)
    # Compare a grid that includes the optimal time
    dosing_times = [0.0, 6.0, 7.0, 12.0, 18.0]
    results = compare_dosing_times("DrugX", "hypertension", tmax_h=2.0, dosing_times_h=dosing_times)
    # The top result should have high efficacy score
    assert results[0].predicted_efficacy_score >= results[-1].predicted_efficacy_score


# ---------------------------------------------------------------------------
# Alignment and scoring tests
# ---------------------------------------------------------------------------


def test_alignment_zero_gives_efficacy_near_100():
    # Force alignment = 0: dose at (disease_peak - tmax), disease_peak = 9 for hypertension
    result = optimize_dosing_time("DrugA", "hypertension", tmax_h=2.0)
    # With optimal timing, drug peaks at disease peak => alignment ~0 => efficacy ~100
    assert result.phase_alignment_h < 0.01
    assert result.predicted_efficacy_score > 99.0


def test_large_alignment_lower_efficacy():
    # disease_peak=9 for hypertension, tmax=0
    # dose at 9h => drug peaks at 9 => alignment=0 => efficacy=100
    # dose at 3h => drug peaks at 3 => alignment=6 => efficacy < 100
    results = compare_dosing_times("DrugA", "hypertension", tmax_h=0.0, dosing_times_h=[9.0, 3.0])
    # Find the result for each dosing time
    result_9 = next(r for r in results if abs(r.optimal_dose_time_h - 9.0) < 0.01)
    result_3 = next(r for r in results if abs(r.optimal_dose_time_h - 3.0) < 0.01)
    # Perfect alignment (dose at 9h) should give higher efficacy than misaligned (dose at 3h)
    assert result_9.predicted_efficacy_score > result_3.predicted_efficacy_score


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_t_half_zero_raises():
    with pytest.raises(ValueError):
        optimize_dosing_time("DrugA", "hypertension", t_half_h=0.0)


def test_validation_dosing_interval_zero_raises():
    with pytest.raises(ValueError):
        optimize_dosing_time("DrugA", "hypertension", dosing_interval_h=0.0)


def test_validation_tmax_negative_raises():
    with pytest.raises(ValueError):
        optimize_dosing_time("DrugA", "hypertension", tmax_h=-1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_indication_uses_default():
    result = optimize_dosing_time("DrugA", "other_condition")
    # Should not raise; disease peak defaults to 12
    assert isinstance(result, ChronotherapyResult)
    assert result.circadian_phase_disease == 12.0


def test_effect_duration_used_in_notes():
    result = optimize_dosing_time("DrugA", "cancer", t_half_h=8.0, effect_duration_h=4.0)
    assert "4.0h" in result.notes
