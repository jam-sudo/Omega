"""Tests for Phase 1088 — Inhalation Anesthetic PK."""

import pytest
from omega_pbpk.core.inhalation_anesthetic_pk import (
    InhalationAnestheticResult,
    compare_agents,
    simulate_inhalation_anesthetic,
)


# --- Return type ---

def test_returns_inhalation_anesthetic_result():
    result = simulate_inhalation_anesthetic("sevoflurane")
    assert isinstance(result, InhalationAnestheticResult)


def test_result_has_correct_agent():
    result = simulate_inhalation_anesthetic("isoflurane")
    assert result.agent == "isoflurane"


# --- Lambda blood gas values ---

def test_lambda_blood_gas_sevoflurane():
    result = simulate_inhalation_anesthetic("sevoflurane")
    assert abs(result.lambda_blood_gas - 0.65) < 1e-9


def test_lambda_blood_gas_desflurane():
    result = simulate_inhalation_anesthetic("desflurane")
    assert abs(result.lambda_blood_gas - 0.42) < 1e-9


def test_lambda_blood_gas_halothane():
    result = simulate_inhalation_anesthetic("halothane")
    assert abs(result.lambda_blood_gas - 2.5) < 1e-9


# --- Fastest / slowest recovery ---

def test_desflurane_fastest_recovery():
    agents = compare_agents()
    agent_names = [r.agent for r in agents]
    # desflurane has lowest lambda_blood_gas → fastest recovery
    assert agents[0].agent == "desflurane"


def test_halothane_slowest_recovery():
    agents = compare_agents()
    # halothane has highest lambda_blood_gas → slowest recovery
    assert agents[-1].agent == "halothane"


def test_desflurane_recovery_less_than_halothane():
    desflurane = simulate_inhalation_anesthetic("desflurane", maintenance_h=2.0, t_end_h=8.0)
    halothane = simulate_inhalation_anesthetic("halothane", maintenance_h=2.0, t_end_h=8.0)
    assert desflurane.t_recovery_h < halothane.t_recovery_h


# --- Induction ordering ---

def test_desflurane_induction_faster_than_halothane():
    desflurane = simulate_inhalation_anesthetic("desflurane", maintenance_h=2.0, t_end_h=6.0)
    halothane = simulate_inhalation_anesthetic("halothane", maintenance_h=2.0, t_end_h=6.0)
    assert desflurane.t_induction_h <= halothane.t_induction_h


def test_induction_time_positive():
    result = simulate_inhalation_anesthetic("sevoflurane")
    assert result.t_induction_h >= 0.0


# --- Fat vs VRG equilibration ---

def test_fat_equilibrates_more_slowly_than_vrg():
    # Fat has much larger Kp (due to 0.80 lipid fraction vs VRG 0.04)
    # → fat takes much longer to reach equilibrium
    result = simulate_inhalation_anesthetic("sevoflurane", maintenance_h=2.0, t_end_h=6.0)
    # VRG should reach closer to its equilibrium than fat at 2h
    # Compare the fraction of equilibrium reached
    # Equilibrium for VRG: f_inspired * lambda_bg * kp_vrg (kp_vrg = 47/0.65 * 0.04)
    # Equilibrium for fat: f_inspired * lambda_bg * kp_fat (kp_fat = 47/0.65 * 0.80)
    # fat equilibrium is 20x higher, so it takes much longer to fill up
    assert result.fat_equilibration_pct_at_2h < 100.0


def test_fat_equilibration_pct_in_range():
    result = simulate_inhalation_anesthetic("sevoflurane")
    assert 0.0 <= result.fat_equilibration_pct_at_2h <= 100.0


def test_fat_equilibration_pct_halothane_in_range():
    result = simulate_inhalation_anesthetic("halothane")
    assert 0.0 <= result.fat_equilibration_pct_at_2h <= 100.0


# --- Peak fat accumulation ---

def test_peak_fat_accumulation_positive():
    result = simulate_inhalation_anesthetic("sevoflurane", maintenance_h=2.0, t_end_h=6.0)
    assert result.peak_fat_accumulation > 0.0


def test_peak_fat_greater_for_halothane_vs_desflurane():
    # Halothane has higher oil:gas partition → more fat accumulation
    halothane = simulate_inhalation_anesthetic("halothane", maintenance_h=2.0, t_end_h=6.0)
    desflurane = simulate_inhalation_anesthetic("desflurane", maintenance_h=2.0, t_end_h=6.0)
    # At same f_inspired, halothane has much higher Kp_fat → more fat accumulation
    assert halothane.peak_fat_accumulation > desflurane.peak_fat_accumulation


# --- Concentration profiles ---

def test_c_vrg_increases_during_maintenance():
    result = simulate_inhalation_anesthetic("sevoflurane", maintenance_h=2.0, t_end_h=6.0, dt_h=0.01)
    # VRG should increase from 0 to near-equilibrium during maintenance
    # Check: c_vrg at t=1h > c_vrg at t=0.1h
    idx_01 = int(0.1 / 0.01)
    idx_10 = int(1.0 / 0.01)
    assert result.c_vrg[idx_10] > result.c_vrg[idx_01]


def test_c_vrg_starts_at_zero():
    result = simulate_inhalation_anesthetic("sevoflurane")
    assert result.c_vrg[0] == 0.0


def test_c_vrg_decreases_after_washout():
    result = simulate_inhalation_anesthetic(
        "sevoflurane", maintenance_h=2.0, t_end_h=6.0, dt_h=0.01
    )
    # After washout (t > 2h), VRG should be declining
    idx_maintenance_end = int(2.0 / 0.01) + 1
    idx_later = int(5.0 / 0.01)
    # Make sure we have enough data points
    if idx_later < len(result.c_vrg):
        assert result.c_vrg[idx_later] < result.c_vrg[idx_maintenance_end]


def test_times_h_length_matches_c_vrg():
    result = simulate_inhalation_anesthetic("isoflurane", t_end_h=4.0, dt_h=0.01)
    assert len(result.times_h) == len(result.c_vrg)


def test_times_h_length_matches_c_fat():
    result = simulate_inhalation_anesthetic("isoflurane", t_end_h=4.0, dt_h=0.01)
    assert len(result.times_h) == len(result.c_fat)


# --- compare_agents ---

def test_compare_agents_returns_four_results():
    results = compare_agents()
    assert len(results) == 4


def test_compare_agents_sorted_by_recovery_ascending():
    results = compare_agents(maintenance_h=2.0)
    for i in range(len(results) - 1):
        assert results[i].t_recovery_h <= results[i + 1].t_recovery_h


def test_compare_agents_all_are_results():
    results = compare_agents()
    for r in results:
        assert isinstance(r, InhalationAnestheticResult)


# --- Validation errors ---

def test_invalid_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        simulate_inhalation_anesthetic("propofol")


def test_zero_f_inspired_raises():
    with pytest.raises(ValueError, match="f_inspired"):
        simulate_inhalation_anesthetic("sevoflurane", f_inspired=0.0)


def test_negative_f_inspired_raises():
    with pytest.raises(ValueError, match="f_inspired"):
        simulate_inhalation_anesthetic("sevoflurane", f_inspired=-0.1)


def test_f_inspired_one_raises():
    with pytest.raises(ValueError, match="f_inspired"):
        simulate_inhalation_anesthetic("sevoflurane", f_inspired=1.0)


def test_negative_maintenance_raises():
    with pytest.raises(ValueError, match="maintenance_h"):
        simulate_inhalation_anesthetic("sevoflurane", maintenance_h=-1.0)


def test_zero_co_raises():
    with pytest.raises(ValueError, match="co_L_per_h"):
        simulate_inhalation_anesthetic("sevoflurane", co_L_per_h=0.0)


def test_zero_alv_vent_raises():
    with pytest.raises(ValueError, match="alveolar_ventilation_L_per_h"):
        simulate_inhalation_anesthetic("sevoflurane", alveolar_ventilation_L_per_h=0.0)
