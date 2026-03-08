"""Tests for Phase 262 — adipose drug accumulation PK model."""

import pytest

from omega_pbpk.core.adipose_drug_accumulation_pk import (
    AdiposeAccumulationResult,
    compare_logp_accumulation,
    simulate_adipose_accumulation,
)

# --- Return type ---


def test_returns_adipose_accumulation_result():
    result = simulate_adipose_accumulation(
        drug_name="DrugA",
        dose_mg=100.0,
        logp=3.0,
        dosing_interval_h=24.0,
        n_doses=5,
        cl_L_per_h=2.0,
    )
    assert isinstance(result, AdiposeAccumulationResult)


# --- Plasma starts high after first dose ---


def test_plasma_positive_after_first_dose():
    result = simulate_adipose_accumulation(
        drug_name="DrugA",
        dose_mg=100.0,
        logp=2.0,
        dosing_interval_h=24.0,
        n_doses=3,
        cl_L_per_h=1.0,
    )
    # After first dose application (index 0), c_plasma should be > 0
    assert result.c_plasma_mg_L[0] > 0.0


# --- Adipose starts at zero ---


def test_adipose_zero_at_start():
    result = simulate_adipose_accumulation(
        drug_name="DrugA",
        dose_mg=100.0,
        logp=3.0,
        dosing_interval_h=24.0,
        n_doses=5,
        cl_L_per_h=2.0,
    )
    assert result.c_adipose_mg_L[0] == 0.0


# --- Adipose accumulates over multiple doses ---


def test_adipose_accumulates_over_doses():
    result = simulate_adipose_accumulation(
        drug_name="DrugB",
        dose_mg=100.0,
        logp=4.0,
        dosing_interval_h=24.0,
        n_doses=10,
        cl_L_per_h=1.0,
    )
    # Adipose at end should be greater than at start
    assert result.c_adipose_mg_L[-1] > result.c_adipose_mg_L[0]


def test_adipose_max_increases_with_doses():
    result3 = simulate_adipose_accumulation(
        "D", dose_mg=100.0, logp=3.0, dosing_interval_h=24.0, n_doses=3, cl_L_per_h=1.0
    )
    result10 = simulate_adipose_accumulation(
        "D", dose_mg=100.0, logp=3.0, dosing_interval_h=24.0, n_doses=10, cl_L_per_h=1.0
    )
    assert max(result10.c_adipose_mg_L) >= max(result3.c_adipose_mg_L)


# --- Higher logP → higher adipose accumulation ratio ---


def test_higher_logp_more_adipose_accumulation():
    # Higher logP → larger k_pa → more drug transferred to adipose → higher accumulation ratio
    low = simulate_adipose_accumulation(
        "D", dose_mg=100.0, logp=1.0, dosing_interval_h=24.0, n_doses=10, cl_L_per_h=2.0
    )
    high = simulate_adipose_accumulation(
        "D", dose_mg=100.0, logp=5.0, dosing_interval_h=24.0, n_doses=10, cl_L_per_h=2.0
    )
    assert high.accumulation_ratio_adipose >= low.accumulation_ratio_adipose


# --- accumulation_ratio_plasma >= 1.0 ---


def test_accumulation_ratio_plasma_gte_1():
    result = simulate_adipose_accumulation(
        drug_name="DrugC",
        dose_mg=50.0,
        logp=2.0,
        dosing_interval_h=12.0,
        n_doses=7,
        cl_L_per_h=1.5,
    )
    assert result.accumulation_ratio_plasma >= 1.0


# --- css_plasma_avg > 0 ---


def test_css_plasma_avg_positive():
    result = simulate_adipose_accumulation(
        drug_name="DrugD",
        dose_mg=100.0,
        logp=2.5,
        dosing_interval_h=24.0,
        n_doses=5,
        cl_L_per_h=2.0,
    )
    assert result.css_plasma_avg > 0.0


# --- css_adipose_avg > 0 ---


def test_css_adipose_avg_positive():
    result = simulate_adipose_accumulation(
        drug_name="DrugD",
        dose_mg=100.0,
        logp=3.0,
        dosing_interval_h=24.0,
        n_doses=8,
        cl_L_per_h=2.0,
    )
    assert result.css_adipose_avg > 0.0


# --- washout_t50_h > 0 ---


def test_washout_t50_positive():
    result = simulate_adipose_accumulation(
        drug_name="DrugE",
        dose_mg=100.0,
        logp=3.0,
        dosing_interval_h=24.0,
        n_doses=5,
        cl_L_per_h=2.0,
    )
    assert result.washout_t50_h > 0.0


def test_washout_t50_reflects_k_ap():
    # k_ap = 0.02/h → t50 = 0.693/0.02 ≈ 34.65 h
    result = simulate_adipose_accumulation(
        drug_name="Drug", dose_mg=100.0, logp=2.0, dosing_interval_h=24.0, n_doses=3, cl_L_per_h=1.0
    )
    assert abs(result.washout_t50_h - 0.693 / 0.02) < 0.01


# --- compare_logp_accumulation ---


def test_compare_returns_5_results():
    results = compare_logp_accumulation("Drug", dose_mg=100.0, cl_L_per_h=2.0)
    assert len(results) == 5


def test_compare_sorted_descending_by_adipose_ratio():
    results = compare_logp_accumulation("Drug", dose_mg=100.0, cl_L_per_h=2.0)
    ratios = [r.accumulation_ratio_adipose for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_custom_logp_values():
    results = compare_logp_accumulation(
        "Drug", dose_mg=100.0, cl_L_per_h=2.0, logp_values=[1.0, 3.0, 5.0]
    )
    assert len(results) == 3


# --- Validation errors ---


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation(
            "Bad", dose_mg=0.0, logp=2.0, dosing_interval_h=24.0, n_doses=5, cl_L_per_h=1.0
        )


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation(
            "Bad", dose_mg=-100.0, logp=2.0, dosing_interval_h=24.0, n_doses=5, cl_L_per_h=1.0
        )


def test_n_doses_zero_raises():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_adipose_accumulation(
            "Bad", dose_mg=100.0, logp=2.0, dosing_interval_h=24.0, n_doses=0, cl_L_per_h=1.0
        )


def test_dosing_interval_zero_raises():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        simulate_adipose_accumulation(
            "Bad", dose_mg=100.0, logp=2.0, dosing_interval_h=0.0, n_doses=5, cl_L_per_h=1.0
        )


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_adipose_accumulation(
            "Bad", dose_mg=100.0, logp=2.0, dosing_interval_h=24.0, n_doses=5, cl_L_per_h=0.0
        )
