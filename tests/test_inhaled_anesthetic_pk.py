"""Tests for Phase 944: inhaled_anesthetic_pk."""

import pytest

from omega_pbpk.core.inhaled_anesthetic_pk import (
    InhaledAnestheticPKResult,
    simulate_inhaled_anesthetic,
    compare_agents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_default(**kwargs):
    """Simulate with default sevoflurane params, allow overrides."""
    defaults = dict(
        agent_name="sevoflurane",
        inspired_fraction=0.02,
        lambda_bg=0.65,
        t_induction_h=0.25,
        t_maintenance_h=1.0,
        t_recovery_h=1.0,
        dt_h=1.0 / 600.0,
    )
    defaults.update(kwargs)
    return simulate_inhaled_anesthetic(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_return_type_is_result():
    result = _simulate_default()
    assert isinstance(result, InhaledAnestheticPKResult)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def test_p_blood_starts_at_zero():
    result = _simulate_default()
    assert result.p_blood[0] == 0.0


def test_p_brain_starts_at_zero():
    result = _simulate_default()
    assert result.p_brain[0] == 0.0


# ---------------------------------------------------------------------------
# Range checks
# ---------------------------------------------------------------------------

def test_p_blood_all_in_0_1():
    result = _simulate_default()
    for val in result.p_blood:
        assert 0.0 <= val <= 1.0, f"p_blood out of range: {val}"


def test_p_brain_all_in_0_1():
    result = _simulate_default()
    for val in result.p_brain:
        assert 0.0 <= val <= 1.0, f"p_brain out of range: {val}"


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def test_cmax_brain_positive():
    result = _simulate_default()
    assert result.cmax_brain > 0.0


def test_time_to_equilibrium_brain_positive():
    result = _simulate_default()
    assert result.time_to_equilibrium_brain_h > 0.0


def test_recovery_half_time_positive():
    result = _simulate_default()
    assert result.recovery_half_time_h > 0.0


# ---------------------------------------------------------------------------
# Lambda_bg effect on equilibration speed
# ---------------------------------------------------------------------------

def test_lower_lambda_bg_faster_equilibrium():
    """Desflurane (low lambda_bg) should equilibrate faster than isoflurane (high)."""
    desflurane = simulate_inhaled_anesthetic(
        agent_name="desflurane",
        inspired_fraction=0.02,
        lambda_bg=0.42,
        t_induction_h=0.25,
        t_maintenance_h=1.0,
        t_recovery_h=1.0,
        dt_h=1.0 / 600.0,
    )
    isoflurane = simulate_inhaled_anesthetic(
        agent_name="isoflurane",
        inspired_fraction=0.02,
        lambda_bg=1.4,
        t_induction_h=0.25,
        t_maintenance_h=1.0,
        t_recovery_h=1.0,
        dt_h=1.0 / 600.0,
    )
    assert desflurane.time_to_equilibrium_brain_h <= isoflurane.time_to_equilibrium_brain_h


# ---------------------------------------------------------------------------
# compare_agents
# ---------------------------------------------------------------------------

def test_compare_agents_returns_3_by_default():
    results = compare_agents()
    assert len(results) == 3


def test_compare_agents_sorted_by_recovery_ascending():
    results = compare_agents()
    times = [r.recovery_half_time_h for r in results]
    assert times == sorted(times)


def test_compare_agents_desflurane_fastest_recovery():
    results = compare_agents()
    # desflurane has lowest lambda_bg = 0.42 → fastest washout
    agent_names = [r.agent_name for r in results]
    assert agent_names[0] == "desflurane"


# ---------------------------------------------------------------------------
# Physiological behavior
# ---------------------------------------------------------------------------

def test_p_brain_increases_during_induction():
    result = _simulate_default(dt_h=1.0 / 60.0)  # coarser for speed
    # Brain should increase in the first half of induction
    n_induction = int(0.25 / (1.0 / 60.0))
    mid = n_induction // 2
    if mid > 0 and mid < len(result.p_brain):
        assert result.p_brain[mid] >= result.p_brain[0]


def test_p_brain_decreases_during_recovery():
    result = _simulate_default(dt_h=1.0 / 60.0)
    # Recovery starts after induction + maintenance
    t_off = 0.25 + 1.0  # 1.25 h
    dt = 1.0 / 60.0
    idx_off = int(round(t_off / dt))
    idx_off = min(idx_off, len(result.p_brain) - 1)
    # Last index
    idx_end = len(result.p_brain) - 1
    if idx_end > idx_off + 1:
        assert result.p_brain[idx_end] < result.p_brain[idx_off]


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_non_empty():
    result = _simulate_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Lambda preserved
# ---------------------------------------------------------------------------

def test_lambda_bg_preserved():
    result = simulate_inhaled_anesthetic(
        agent_name="desflurane",
        lambda_bg=0.42,
    )
    assert result.lambda_bg == 0.42


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_validation_inspired_fraction_zero():
    with pytest.raises(ValueError, match="inspired_fraction"):
        simulate_inhaled_anesthetic(inspired_fraction=0.0)


def test_validation_inspired_fraction_negative():
    with pytest.raises(ValueError, match="inspired_fraction"):
        simulate_inhaled_anesthetic(inspired_fraction=-0.01)


def test_validation_inspired_fraction_above_1():
    with pytest.raises(ValueError, match="inspired_fraction"):
        simulate_inhaled_anesthetic(inspired_fraction=1.01)


def test_validation_lambda_bg_zero():
    with pytest.raises(ValueError, match="lambda_bg"):
        simulate_inhaled_anesthetic(lambda_bg=0.0)


def test_validation_lambda_bg_negative():
    with pytest.raises(ValueError, match="lambda_bg"):
        simulate_inhaled_anesthetic(lambda_bg=-1.0)


def test_validation_t_induction_zero():
    with pytest.raises(ValueError, match="t_induction_h"):
        simulate_inhaled_anesthetic(t_induction_h=0.0)


def test_validation_t_maintenance_zero():
    with pytest.raises(ValueError, match="t_maintenance_h"):
        simulate_inhaled_anesthetic(t_maintenance_h=0.0)


# ---------------------------------------------------------------------------
# List length consistency
# ---------------------------------------------------------------------------

def test_times_len_equals_p_brain_len():
    result = _simulate_default()
    assert len(result.times_h) == len(result.p_brain)


def test_times_len_equals_p_blood_len():
    result = _simulate_default()
    assert len(result.times_h) == len(result.p_blood)
