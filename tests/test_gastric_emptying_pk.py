"""Tests for Phase 709 — Gastric emptying-controlled PK model."""

import pytest

from omega_pbpk.core.gastric_emptying_pk import (
    GastricEmptyingResult,
    compare_fed_fasted,
    simulate_gastric_emptying_pk,
)

# --- Basic return type and initial conditions ---


def test_returns_gastric_emptying_result():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, GastricEmptyingResult)


def test_stomach_initial_equals_dose():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.a_stomach_mg[0] == pytest.approx(100.0)


def test_intestine_initial_is_zero():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.a_intestine_mg[0] == pytest.approx(0.0)


def test_plasma_initial_is_zero():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


# --- Food state effects ---


def test_fed_tmax_greater_than_fasted_tmax():
    fasted = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fasted")
    fed = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fed")
    assert fed.tmax_h > fasted.tmax_h


def test_fed_kge_less_than_fasted_kge():
    fasted = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fasted")
    fed = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fed")
    assert fed.kge_per_h < fasted.kge_per_h


def test_fed_kge_matches_expected():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fed")
    assert result.kge_per_h == pytest.approx(0.5)


def test_fasted_kge_matches_expected():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="fasted")
    assert result.kge_per_h == pytest.approx(2.0)


# --- PK metrics ---


def test_cmax_positive():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_lag_time_positive():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.lag_time_h > 0.0


def test_t_half_positive():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert result.t_half_h > 0.0


def test_notes_nonempty():
    result = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    assert len(result.notes) > 0


# --- Dose proportionality ---


def test_double_dose_doubles_cmax():
    result1 = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    result2 = simulate_gastric_emptying_pk("TestDrug", dose_mg=200.0)
    assert result2.cmax_mg_L == pytest.approx(result1.cmax_mg_L * 2.0, rel=0.01)


def test_double_dose_doubles_auc():
    result1 = simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0)
    result2 = simulate_gastric_emptying_pk("TestDrug", dose_mg=200.0)
    assert result2.auc_mg_h_per_L == pytest.approx(result1.auc_mg_h_per_L * 2.0, rel=0.01)


# --- compare_fed_fasted ---


def test_compare_fed_fasted_returns_dict():
    result = compare_fed_fasted("TestDrug", dose_mg=100.0)
    assert isinstance(result, dict)


def test_compare_fed_fasted_has_required_keys():
    result = compare_fed_fasted("TestDrug", dose_mg=100.0)
    assert "fasted" in result
    assert "fed" in result
    assert "tmax_delay_h" in result
    assert "auc_ratio" in result
    assert "notes" in result


def test_compare_fed_fasted_results_are_correct_type():
    result = compare_fed_fasted("TestDrug", dose_mg=100.0)
    assert isinstance(result["fasted"], GastricEmptyingResult)
    assert isinstance(result["fed"], GastricEmptyingResult)


def test_tmax_delay_nonnegative():
    result = compare_fed_fasted("TestDrug", dose_mg=100.0)
    assert result["tmax_delay_h"] >= 0.0


def test_compare_notes_nonempty():
    result = compare_fed_fasted("TestDrug", dose_mg=100.0)
    assert len(result["notes"]) > 0


# --- Validation ---


def test_invalid_food_state_raises():
    with pytest.raises(ValueError):
        simulate_gastric_emptying_pk("TestDrug", dose_mg=100.0, food_state="random_state")


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_gastric_emptying_pk("TestDrug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_gastric_emptying_pk("TestDrug", dose_mg=-50.0)
