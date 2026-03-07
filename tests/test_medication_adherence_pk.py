"""Tests for Phase 666 — clinical/medication_adherence_pk.py"""

import pytest

from omega_pbpk.clinical.medication_adherence_pk import (
    AdherencePKResult,
    adherence_sensitivity_analysis,
    simulate_adherence_pk,
)

VALID_CONCERNS = {"none", "minor", "moderate", "severe"}


# --- Basic return type tests ---


def test_returns_adherence_pk_result():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, AdherencePKResult)


def test_adherence_pct_in_range():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert 0 <= result.adherence_pct <= 100


def test_cmax_mg_L_positive():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L > 0


def test_time_in_range_pct_in_range():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert 0 <= result.time_in_range_pct <= 100


def test_auc_vs_perfect_pct_positive():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert result.auc_vs_perfect_pct > 0


def test_clinical_concern_valid():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert result.clinical_concern in VALID_CONCERNS


def test_n_doses_taken_le_n_doses_planned():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0, n_doses_planned=14)
    assert result.n_doses_taken <= result.n_doses_taken + result.n_missed_doses
    assert result.n_doses_taken + result.n_missed_doses == 14


def test_notes_nonempty():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert result.notes


# --- Scientific behavior tests ---


def test_perfect_adherence_100_pct():
    result = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, missed_dose_indices=[]
    )
    assert result.adherence_pct == pytest.approx(100.0)


def test_perfect_adherence_auc_approx_100_pct():
    result = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, missed_dose_indices=[]
    )
    assert result.auc_vs_perfect_pct == pytest.approx(100.0, rel=1e-3)


def test_50_pct_missed_adherence_pct():
    # Miss 7 of 14 doses
    missed = list(range(0, 14, 2))  # 7 doses: 0,2,4,6,8,10,12
    result = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, missed_dose_indices=missed
    )
    assert result.adherence_pct == pytest.approx(50.0, rel=0.01)


def test_missed_doses_reduce_auc():
    perfect = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, missed_dose_indices=[]
    )
    missed_result = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, missed_dose_indices=[1, 3, 5]
    )
    assert missed_result.auc_vs_perfect_pct < perfect.auc_vs_perfect_pct


def test_late_doses_time_subtherapeutic_nonnegative():
    result = simulate_adherence_pk(
        "TestDrug", dose_mg=100.0, n_doses_planned=14, late_dose_delays_h={2: 3.0, 5: 4.0}
    )
    assert result.time_subtherapeutic_h >= 0


def test_partial_doses_lower_auc():
    perfect = simulate_adherence_pk("TestDrug", dose_mg=100.0, n_doses_planned=14)
    partial = simulate_adherence_pk(
        "TestDrug",
        dose_mg=100.0,
        n_doses_planned=14,
        partial_dose_fractions={i: 0.5 for i in range(14)},
    )
    assert partial.auc_vs_perfect_pct < perfect.auc_vs_perfect_pct


# --- Sensitivity analysis tests ---


def test_sensitivity_analysis_returns_4_results():
    results = adherence_sensitivity_analysis("TestDrug", dose_mg=100.0)
    assert len(results) == 4


def test_sensitivity_sorted_descending_adherence():
    results = adherence_sensitivity_analysis("TestDrug", dose_mg=100.0)
    adherences = [r.adherence_pct for r in results]
    assert adherences == sorted(adherences, reverse=True)


def test_sensitivity_higher_adherence_higher_auc():
    results = adherence_sensitivity_analysis("TestDrug", dose_mg=100.0)
    # results[0] = 100%, results[-1] = 40%
    assert results[0].auc_vs_perfect_pct >= results[-1].auc_vs_perfect_pct


def test_sensitivity_40_pct_clinical_concern_severe():
    results = adherence_sensitivity_analysis("TestDrug", dose_mg=100.0, n_doses_planned=14)
    # Find the 40% adherence result
    forty_pct = [r for r in results if r.adherence_pct < 50]
    assert len(forty_pct) >= 1
    assert forty_pct[0].clinical_concern == "severe"


# --- Array length consistency ---


def test_times_and_concentrations_same_length():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


# --- Validation tests ---


def test_validation_dose_mg_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adherence_pk("X", dose_mg=0.0)


def test_validation_mec_ge_mtc_raises():
    with pytest.raises(ValueError, match="mec_mg_L"):
        simulate_adherence_pk("X", dose_mg=100.0, mec_mg_L=5.0, mtc_mg_L=5.0)


def test_validation_n_doses_planned_zero_raises():
    with pytest.raises(ValueError, match="n_doses_planned"):
        simulate_adherence_pk("X", dose_mg=100.0, n_doses_planned=0)


# --- Time budget test ---


def test_time_subtherapeutic_plus_supratherapeutic_le_total():
    result = simulate_adherence_pk("TestDrug", dose_mg=100.0, n_doses_planned=14, dt_h=0.1)
    total_sim_time = result.times_h[-1] if result.times_h else 0.0
    assert (
        result.time_subtherapeutic_h + result.time_supratherapeutic_h
        <= total_sim_time + 0.2  # allow small rounding tolerance
    )
