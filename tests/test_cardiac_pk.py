"""
Tests for Phase 894 — Cardiac Drug Distribution (core/cardiac_pk.py)
"""

import pytest

from omega_pbpk.core.cardiac_pk import (
    CardiacPKResult,
    compare_cardiac_scenarios,
    simulate_cardiac_pk,
)


# ---------------------------------------------------------------------------
# Basic instantiation / return type
# ---------------------------------------------------------------------------


def test_returns_cardiac_pk_result():
    result = simulate_cardiac_pk("Amiodarone", 200.0)
    assert isinstance(result, CardiacPKResult)


def test_drug_name_and_dose_stored():
    result = simulate_cardiac_pk("Sotalol", 80.0)
    assert result.drug_name == "Sotalol"
    assert result.dose_mg == 80.0


# ---------------------------------------------------------------------------
# Time series shape
# ---------------------------------------------------------------------------


def test_time_series_length_matches():
    result = simulate_cardiac_pk("DrugA", 10.0, t_end_h=10.0, dt_h=0.5)
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_myocardium_mg_L)


def test_time_series_starts_at_zero():
    result = simulate_cardiac_pk("DrugA", 10.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_series_ends_near_t_end():
    result = simulate_cardiac_pk("DrugA", 10.0, t_end_h=12.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(12.0, abs=0.2)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_initial_plasma_concentration():
    dose = 200.0
    vd = 70.0
    result = simulate_cardiac_pk("DrugB", dose, vd_sys_L=vd)
    assert result.c_plasma_mg_L[0] == pytest.approx(dose / vd)


def test_initial_myocardium_zero():
    result = simulate_cardiac_pk("DrugB", 200.0)
    assert result.c_myocardium_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def test_plasma_declines_over_time():
    # Use low k_uptake so plasma declines monotonically (numerically stable)
    result = simulate_cardiac_pk(
        "DrugC", 200.0, k_uptake_per_h=0.5, k_efflux_per_h=0.1, kp_heart=1.0
    )
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_myocardium_rises_from_zero():
    result = simulate_cardiac_pk("DrugC", 200.0, t_end_h=5.0)
    assert result.c_myocardium_mg_L[-1] > 0.0


def test_cmax_plasma_equals_initial():
    # Use low k_uptake so plasma starts at peak and declines (IV bolus)
    dose = 100.0
    vd = 70.0
    result = simulate_cardiac_pk(
        "DrugD", dose, vd_sys_L=vd, k_uptake_per_h=0.5, k_efflux_per_h=0.1, kp_heart=1.0
    )
    expected_cmax = dose / vd
    assert result.cmax_plasma == pytest.approx(expected_cmax)


def test_cmax_myocardium_positive():
    result = simulate_cardiac_pk("DrugD", 100.0)
    assert result.cmax_myocardium > 0.0


# ---------------------------------------------------------------------------
# AUC metrics
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_cardiac_pk("DrugE", 50.0)
    assert result.auc_plasma > 0.0


def test_auc_myocardium_positive():
    result = simulate_cardiac_pk("DrugE", 50.0)
    assert result.auc_myocardium > 0.0


def test_heart_plasma_kp_is_ratio():
    result = simulate_cardiac_pk("DrugF", 50.0)
    if result.auc_plasma > 0:
        expected = result.auc_myocardium / result.auc_plasma
        assert result.heart_plasma_kp == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Cardiac exposure score
# ---------------------------------------------------------------------------


def test_cardiac_exposure_score_in_range():
    result = simulate_cardiac_pk("DrugG", 50.0)
    assert 0.0 <= result.cardiac_exposure_score <= 100.0


def test_higher_kp_gives_higher_exposure_score():
    low = simulate_cardiac_pk("Drug", 50.0, kp_heart=1.0)
    high = simulate_cardiac_pk("Drug", 50.0, kp_heart=10.0)
    assert high.cardiac_exposure_score >= low.cardiac_exposure_score


# ---------------------------------------------------------------------------
# QTc risk
# ---------------------------------------------------------------------------


def test_qtc_risk_low_by_default_small_dose():
    # Very small dose => cmax_uM very small => occupancy low
    result = simulate_cardiac_pk(
        "Safe", 0.01, kp_heart=1.0, herg_ic50_uM=100.0, mw_Da=400.0
    )
    assert result.qtc_risk_level == "low"


def test_qtc_risk_high_with_large_dose_low_ic50():
    # Large dose, low IC50 => high occupancy
    result = simulate_cardiac_pk(
        "Dangerous", 5000.0, kp_heart=5.0, herg_ic50_uM=0.1, mw_Da=300.0,
        v_heart_L=0.3
    )
    assert result.qtc_risk_level == "high"


def test_notes_low_risk():
    result = simulate_cardiac_pk(
        "Safe", 0.01, kp_heart=1.0, herg_ic50_uM=100.0, mw_Da=400.0
    )
    assert result.notes == "Low cardiac exposure risk."


def test_notes_high_risk():
    result = simulate_cardiac_pk(
        "Dangerous", 5000.0, kp_heart=5.0, herg_ic50_uM=0.1, mw_Da=300.0
    )
    assert "QTc monitoring required" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_pk("Drug", -10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_pk("Drug", 0.0)


def test_invalid_kp_raises():
    with pytest.raises(ValueError, match="kp_heart"):
        simulate_cardiac_pk("Drug", 10.0, kp_heart=0.0)


def test_invalid_v_heart_raises():
    with pytest.raises(ValueError, match="v_heart_L"):
        simulate_cardiac_pk("Drug", 10.0, v_heart_L=-0.1)


def test_invalid_herg_ic50_raises():
    with pytest.raises(ValueError, match="herg_ic50_uM"):
        simulate_cardiac_pk("Drug", 10.0, herg_ic50_uM=0.0)


# ---------------------------------------------------------------------------
# compare_cardiac_scenarios
# ---------------------------------------------------------------------------


def test_compare_scenarios_returns_sorted():
    scenarios = [
        {"kp_heart": 1.0},
        {"kp_heart": 5.0},
        {"kp_heart": 3.0},
    ]
    results = compare_cardiac_scenarios("Antiarrhythmic", 100.0, scenarios)
    scores = [r.cardiac_exposure_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_compare_scenarios_count():
    scenarios = [{"kp_heart": 1.0}, {"kp_heart": 2.0}]
    results = compare_cardiac_scenarios("Drug", 50.0, scenarios)
    assert len(results) == 2


def test_compare_scenarios_dose_override():
    scenarios = [{"dose_mg": 200.0}, {"dose_mg": 50.0}]
    results = compare_cardiac_scenarios("Drug", 100.0, scenarios)
    # Both should produce valid results (no exceptions)
    assert len(results) == 2
