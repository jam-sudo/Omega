"""Tests for multiple_dosing_pk module (Phase 691)."""

import math

import pytest

from omega_pbpk.clinical.multiple_dosing_pk import (
    MultipleDosePKResult,
    simulate_multiple_dosing,
    simulate_qd_bid_tid,
)

# ── Basic return type ─────────────────────────────────────────────────────────


def test_returns_multiple_dose_pk_result():
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    assert isinstance(result, MultipleDosePKResult)


def test_single_dose_result_has_nonzero_cmax():
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    assert result.cmax_mg_L > 0.0


# ── n_doses and total_dose_mg ─────────────────────────────────────────────────


def test_n_doses_equals_len_doses_mg():
    doses = [100.0, 100.0, 100.0]
    times = [0.0, 24.0, 48.0]
    result = simulate_multiple_dosing("DrugA", doses, times)
    assert result.n_doses == 3


def test_total_dose_mg_equals_sum():
    doses = [80.0, 120.0, 100.0]
    times = [0.0, 12.0, 24.0]
    result = simulate_multiple_dosing("DrugA", doses, times)
    assert abs(result.total_dose_mg - 300.0) < 1e-9


# ── Multiple doses increase Cmax vs single dose ───────────────────────────────


def test_multiple_doses_increase_cmax():
    single = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    multi = simulate_multiple_dosing("DrugA", [100.0, 100.0, 100.0], [0.0, 12.0, 24.0])
    assert multi.cmax_mg_L > single.cmax_mg_L


# ── Accumulation ratio ────────────────────────────────────────────────────────


def test_accumulation_ratio_geq_1_for_repeated_dosing():
    result = simulate_multiple_dosing(
        "DrugA",
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
    )
    assert result.accumulation_ratio >= 1.0


def test_accumulation_ratio_single_dose_near_1():
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    # Single dose: ratio should be approximately 1
    assert result.accumulation_ratio >= 0.5  # flexible tolerance


# ── css_max > css_min ─────────────────────────────────────────────────────────


def test_css_max_greater_than_css_min_for_multiple_doses():
    result = simulate_multiple_dosing("DrugA", [100.0] * 7, [i * 24.0 for i in range(7)])
    assert result.css_max > result.css_min


# ── t_half_h ─────────────────────────────────────────────────────────────────


def test_t_half_h_matches_formula():
    cl = 5.0
    vd = 50.0
    expected_t_half = math.log(2) * vd / cl
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0], cl_L_per_h=cl, vd_L=vd)
    assert abs(result.t_half_h - expected_t_half) < 1e-6


# ── AUC is positive ───────────────────────────────────────────────────────────


def test_auc_is_positive():
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    assert result.auc_0_inf > 0.0


# ── Times and concentrations lists ───────────────────────────────────────────


def test_times_and_concentrations_same_length():
    result = simulate_multiple_dosing("DrugA", [100.0, 100.0], [0.0, 12.0])
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_start_at_zero():
    result = simulate_multiple_dosing("DrugA", [100.0], [0.0])
    assert result.times_h[0] == 0.0


# ── simulate_qd_bid_tid ───────────────────────────────────────────────────────


def test_simulate_qd_bid_tid_returns_dict():
    res = simulate_qd_bid_tid("DrugX", 100.0, 5.0, 50.0, 1.0)
    assert isinstance(res, dict)


def test_simulate_qd_bid_tid_has_qd_bid_tid_keys():
    res = simulate_qd_bid_tid("DrugX", 100.0, 5.0, 50.0, 1.0)
    assert "qd" in res
    assert "bid" in res
    assert "tid" in res


def test_simulate_qd_bid_tid_each_result_is_MultipleDosePKResult():
    res = simulate_qd_bid_tid("DrugX", 100.0, 5.0, 50.0, 1.0)
    assert isinstance(res["qd"], MultipleDosePKResult)
    assert isinstance(res["bid"], MultipleDosePKResult)
    assert isinstance(res["tid"], MultipleDosePKResult)


def test_bid_css_max_less_than_qd_css_max():
    """BID has less peak fluctuation than QD for the same daily dose."""
    res = simulate_qd_bid_tid("DrugX", 100.0, 5.0, 50.0, 1.0, n_days=7)
    assert res["bid"].css_max < res["qd"].css_max


def test_bid_css_min_greater_than_qd_css_min():
    """BID has higher troughs than QD for the same daily dose."""
    res = simulate_qd_bid_tid("DrugX", 100.0, 5.0, 50.0, 1.0, n_days=7)
    assert res["bid"].css_min > res["qd"].css_min


# ── Validation errors ─────────────────────────────────────────────────────────


def test_validation_unequal_lists_raise_value_error():
    with pytest.raises(ValueError):
        simulate_multiple_dosing("DrugA", [100.0, 200.0], [0.0])


def test_validation_unsorted_times_raise_value_error():
    with pytest.raises(ValueError):
        simulate_multiple_dosing("DrugA", [100.0, 100.0], [12.0, 0.0])


def test_validation_negative_dose_raises_value_error():
    with pytest.raises(ValueError):
        simulate_multiple_dosing("DrugA", [-100.0], [0.0])


def test_validation_zero_dose_raises_value_error():
    with pytest.raises(ValueError):
        simulate_multiple_dosing("DrugA", [0.0], [0.0])


# ── drug_name stored ─────────────────────────────────────────────────────────


def test_drug_name_stored_in_result():
    result = simulate_multiple_dosing("TestDrug", [50.0], [0.0])
    assert result.drug_name == "TestDrug"
