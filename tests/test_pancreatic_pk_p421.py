"""Tests for pancreatic PK model — Phase 421."""

import pytest

from omega_pbpk.core.pancreatic_pk_p421 import (
    PancreaticPKResult,
    compare_pancreatic_penetration,
    simulate_pancreatic_pk,
)

# --- Helper ---


def make_default_result(**kwargs) -> PancreaticPKResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
    )
    params.update(kwargs)
    return simulate_pancreatic_pk(**params)


# --- Return type ---


def test_return_type():
    result = make_default_result()
    assert isinstance(result, PancreaticPKResult)


def test_drug_name_preserved():
    result = make_default_result(drug_name="Metformin")
    assert result.drug_name == "Metformin"


def test_dose_preserved():
    result = make_default_result(dose_mg=50.0)
    assert result.dose_mg == 50.0


# --- Array consistency ---


def test_times_start_at_zero():
    result = make_default_result()
    assert result.times_h[0] == 0.0


def test_arrays_consistent_length():
    result = make_default_result()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_pancreas_mg_L) == n
    assert len(result.c_pancreatic_juice_mg_L) == n


def test_times_monotonically_increasing():
    result = make_default_result()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# --- Initial conditions ---


def test_plasma_initial_concentration():
    dose_mg = 100.0
    vd_plasma = 5.0
    result = simulate_pancreatic_pk(
        drug_name="Drug",
        dose_mg=dose_mg,
        cl_sys_L_per_h=10.0,
        vd_plasma_L=vd_plasma,
    )
    assert abs(result.c_plasma_mg_L[0] - dose_mg / vd_plasma) < 1e-9


def test_pancreas_starts_at_zero():
    result = make_default_result()
    assert result.c_pancreas_mg_L[0] == 0.0


# --- Pharmacokinetic metrics ---


def test_cmax_pancreas_positive():
    result = make_default_result()
    assert result.cmax_pancreas_mg_L > 0.0


def test_cmax_plasma_positive():
    result = make_default_result()
    assert result.cmax_plasma_mg_L > 0.0


def test_auc_pancreas_positive():
    result = make_default_result()
    assert result.auc_pancreas_mg_h_per_L > 0.0


def test_auc_plasma_positive():
    result = make_default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_pancreas_to_plasma_ratio_positive():
    result = make_default_result()
    assert result.pancreas_to_plasma_ratio > 0.0


def test_juice_secretion_rate_nonnegative():
    result = make_default_result()
    assert result.juice_secretion_rate_mg_per_h >= 0.0


def test_t_half_pancreas_positive():
    result = make_default_result()
    assert result.t_half_pancreas_h > 0.0


def test_t_half_pancreas_formula():
    k_tp = 0.3
    k_sec = 0.2
    result = simulate_pancreatic_pk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_sys_L_per_h=10.0,
        k_tp_per_h=k_tp,
        k_secretion_per_h=k_sec,
    )
    expected = 0.693 / (k_tp + k_sec)
    assert abs(result.t_half_pancreas_h - expected) < 1e-6


# --- Sensitivity ---


def test_higher_k_pt_higher_pancreatic_penetration():
    low_k_pt = simulate_pancreatic_pk("Drug", 100.0, 10.0, k_pt_per_h=0.1)
    high_k_pt = simulate_pancreatic_pk("Drug", 100.0, 10.0, k_pt_per_h=2.0)
    assert high_k_pt.pancreas_to_plasma_ratio > low_k_pt.pancreas_to_plasma_ratio


def test_dose_linearity_cmax():
    r1 = simulate_pancreatic_pk("Drug", 100.0, 10.0)
    r2 = simulate_pancreatic_pk("Drug", 200.0, 10.0)
    assert abs(r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L - 2.0) < 1e-6


def test_dose_linearity_cmax_pancreas():
    r1 = simulate_pancreatic_pk("Drug", 100.0, 10.0)
    r2 = simulate_pancreatic_pk("Drug", 200.0, 10.0)
    assert abs(r2.cmax_pancreas_mg_L / r1.cmax_pancreas_mg_L - 2.0) < 1e-6


# --- compare_pancreatic_penetration ---


def test_compare_returns_correct_length():
    k_pt_values = [0.1, 0.5, 1.0, 2.0]
    results = compare_pancreatic_penetration("Drug", 100.0, 10.0, k_pt_values)
    assert len(results) == len(k_pt_values)


def test_compare_sorted_descending():
    k_pt_values = [0.1, 0.5, 1.0, 2.0]
    results = compare_pancreatic_penetration("Drug", 100.0, 10.0, k_pt_values)
    ratios = [r.pancreas_to_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_all_are_results():
    k_pt_values = [0.2, 0.8]
    results = compare_pancreatic_penetration("Drug", 100.0, 10.0, k_pt_values)
    for r in results:
        assert isinstance(r, PancreaticPKResult)


# --- Validation errors ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pancreatic_pk("Drug", 0.0, 10.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pancreatic_pk("Drug", -1.0, 10.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_pancreatic_pk("Drug", 100.0, 0.0)


def test_validation_cl_negative():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_pancreatic_pk("Drug", 100.0, -5.0)
