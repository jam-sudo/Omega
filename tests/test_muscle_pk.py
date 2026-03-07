"""
Tests for Phase 893 — Muscle Drug Distribution (core/muscle_pk.py)
"""

import pytest

from omega_pbpk.core.muscle_pk import (
    MusclePKResult,
    compare_statin_kp,
    simulate_muscle_pk,
)

# ---------------------------------------------------------------------------
# Basic instantiation / return type
# ---------------------------------------------------------------------------


def test_returns_muscle_pk_result():
    result = simulate_muscle_pk("Simvastatin", 40.0)
    assert isinstance(result, MusclePKResult)


def test_drug_name_and_dose_stored():
    result = simulate_muscle_pk("Atorvastatin", 80.0)
    assert result.drug_name == "Atorvastatin"
    assert result.dose_mg == 80.0


# ---------------------------------------------------------------------------
# Time series shape
# ---------------------------------------------------------------------------


def test_time_series_length_matches():
    result = simulate_muscle_pk("DrugA", 10.0, t_end_h=10.0, dt_h=0.5)
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_muscle_interstitial_mg_L)
    assert n == len(result.c_muscle_intracell_mg_L)


def test_time_series_starts_at_zero():
    result = simulate_muscle_pk("DrugA", 10.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_series_ends_near_t_end():
    result = simulate_muscle_pk("DrugA", 10.0, t_end_h=12.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(12.0, abs=0.2)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_initial_plasma_concentration():
    dose = 50.0
    vd = 50.0
    result = simulate_muscle_pk("DrugB", dose, vd_sys_L=vd)
    assert result.c_plasma_mg_L[0] == pytest.approx(dose / vd)


def test_initial_interstitial_zero():
    result = simulate_muscle_pk("DrugB", 50.0)
    assert result.c_muscle_interstitial_mg_L[0] == pytest.approx(0.0)


def test_initial_intracell_zero():
    result = simulate_muscle_pk("DrugB", 50.0)
    assert result.c_muscle_intracell_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Decay and accumulation dynamics
# ---------------------------------------------------------------------------


def test_plasma_declines_over_time():
    result = simulate_muscle_pk("DrugC", 100.0)
    # Plasma should decline monotonically after t=0
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_intracell_rises_initially():
    result = simulate_muscle_pk("DrugC", 100.0, t_end_h=5.0)
    # Intracellular should rise from 0
    assert result.c_muscle_intracell_mg_L[-1] > 0.0


def test_cmax_plasma_equals_initial():
    dose = 100.0
    vd = 50.0
    result = simulate_muscle_pk("DrugD", dose, vd_sys_L=vd)
    expected_cmax = dose / vd
    assert result.cmax_plasma == pytest.approx(expected_cmax)


# ---------------------------------------------------------------------------
# AUC metrics
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_muscle_pk("DrugE", 50.0)
    assert result.auc_plasma > 0.0


def test_auc_muscle_intracell_positive():
    result = simulate_muscle_pk("DrugE", 50.0)
    assert result.auc_muscle_intracell > 0.0


def test_muscle_plasma_kp_is_ratio():
    result = simulate_muscle_pk("DrugF", 50.0)
    if result.auc_plasma > 0:
        expected = result.auc_muscle_intracell / result.auc_plasma
        assert result.muscle_plasma_kp == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------


def test_risk_score_in_range():
    result = simulate_muscle_pk("DrugG", 50.0)
    assert 0.0 <= result.myopathy_risk_score <= 100.0


def test_high_kp_gives_higher_risk():
    low_kp = simulate_muscle_pk("Drug", 50.0, kp_muscle=1.0)
    high_kp = simulate_muscle_pk("Drug", 50.0, kp_muscle=10.0)
    assert high_kp.myopathy_risk_score >= low_kp.myopathy_risk_score


def test_risk_level_low():
    result = simulate_muscle_pk("LowRisk", 5.0, kp_muscle=0.5, k_uptake_per_h=0.01)
    assert result.myopathy_risk_level == "low"


def test_risk_level_high():
    result = simulate_muscle_pk(
        "HighRisk", 100.0, kp_muscle=10.0, k_uptake_per_h=1.0, cl_sys_L_per_h=2.0
    )
    assert result.myopathy_risk_level == "high"


def test_notes_low_risk():
    result = simulate_muscle_pk("LowRisk", 5.0, kp_muscle=0.5, k_uptake_per_h=0.01)
    assert result.notes == "Low myopathy risk."


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("Drug", -10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("Drug", 0.0)


def test_invalid_kp_raises():
    with pytest.raises(ValueError, match="kp_muscle"):
        simulate_muscle_pk("Drug", 10.0, kp_muscle=-1.0)


def test_invalid_v_muscle_raises():
    with pytest.raises(ValueError, match="v_muscle_L"):
        simulate_muscle_pk("Drug", 10.0, v_muscle_L=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_muscle_pk("Drug", 10.0, cl_sys_L_per_h=-5.0)


# ---------------------------------------------------------------------------
# compare_statin_kp
# ---------------------------------------------------------------------------


def test_compare_statin_kp_returns_sorted():
    kp_values = [1.0, 3.0, 5.0, 2.0]
    results = compare_statin_kp("Statin", 40.0, kp_values)
    scores = [r.myopathy_risk_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_compare_statin_kp_count():
    kp_values = [1.0, 2.0, 4.0]
    results = compare_statin_kp("Statin", 40.0, kp_values)
    assert len(results) == 3
