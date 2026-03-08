"""
Tests for Phase 981 — Muscle Drug Distribution (core/muscle_pk_p981.py)
"""

import pytest

from omega_pbpk.core.muscle_pk_p981 import MusclePKResult, simulate_muscle_pk

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_muscle_pk_result():
    result = simulate_muscle_pk("Gentamicin", 80.0)
    assert isinstance(result, MusclePKResult)


# ---------------------------------------------------------------------------
# Positive metrics
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.cmax_plasma_mg_L > 0


def test_cmax_muscle_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.cmax_muscle_mg_L > 0


def test_tmax_muscle_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.tmax_muscle_h > 0


def test_auc_plasma_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_auc_muscle_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.auc_muscle_mg_h_per_L > 0


def test_muscle_to_plasma_ratio_positive():
    result = simulate_muscle_pk("DrugA", 100.0)
    assert result.muscle_to_plasma_ratio > 0


# ---------------------------------------------------------------------------
# Route-specific initial conditions
# ---------------------------------------------------------------------------


def test_iv_initial_plasma_nonzero():
    result = simulate_muscle_pk("DrugB", 50.0, route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_iv_initial_muscle_zero():
    result = simulate_muscle_pk("DrugB", 50.0, route="iv")
    assert result.c_muscle_mg_L[0] == pytest.approx(0.0)


def test_oral_initial_plasma_zero():
    result = simulate_muscle_pk("DrugC", 50.0, route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_oral_initial_muscle_zero():
    result = simulate_muscle_pk("DrugC", 50.0, route="oral")
    assert result.c_muscle_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


def test_higher_logp_higher_kp_muscle():
    low = simulate_muscle_pk("DrugD", 100.0, logp=0.5)
    high = simulate_muscle_pk("DrugD", 100.0, logp=3.0)
    assert high.kp_muscle > low.kp_muscle


def test_higher_logp_higher_muscle_to_plasma_ratio():
    low = simulate_muscle_pk("DrugE", 100.0, logp=0.5)
    high = simulate_muscle_pk("DrugE", 100.0, logp=4.0)
    assert high.muscle_to_plasma_ratio > low.muscle_to_plasma_ratio


# ---------------------------------------------------------------------------
# kp_muscle stored correctly
# ---------------------------------------------------------------------------


def test_kp_muscle_stored():
    result = simulate_muscle_pk("DrugF", 100.0, logp=1.0, fup=0.5)
    assert result.kp_muscle > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("Drug", -10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("Drug", 0.0)


def test_invalid_v_plasma_raises():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_muscle_pk("Drug", 100.0, v_plasma_L=0.0)


def test_invalid_v_muscle_raises():
    with pytest.raises(ValueError, match="v_muscle_L"):
        simulate_muscle_pk("Drug", 100.0, v_muscle_L=-5.0)


def test_invalid_cl_plasma_raises():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_muscle_pk("Drug", 100.0, cl_plasma_L_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_muscle_pk("Drug", 100.0, route="intramuscular")


# ---------------------------------------------------------------------------
# Notes field populated
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = simulate_muscle_pk("DrugG", 100.0)
    assert len(result.notes) > 0
