"""Tests for Phase 415 — intranasal_pk.py"""

import pytest

from omega_pbpk.core.intranasal_pk import (
    IntranasalPKResult,
    compare_intranasal_clearance,
    simulate_intranasal_pk,
)

# --- Fixtures ---


def base_result():
    return simulate_intranasal_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        ka_nasal_per_h=2.0,
        k_clearance_per_h=0.5,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
        t_end_h=12.0,
        dt_h=0.05,
    )


# --- Return type ---


def test_return_type():
    result = base_result()
    assert isinstance(result, IntranasalPKResult)


def test_route_is_intranasal():
    result = base_result()
    assert result.route == "intranasal"


def test_drug_name_preserved():
    result = base_result()
    assert result.drug_name == "TestDrug"


def test_dose_preserved():
    result = base_result()
    assert result.dose_mg == 10.0


# --- Array consistency ---


def test_times_start_at_zero():
    result = base_result()
    assert result.times_h[0] == 0.0


def test_arrays_consistent_length():
    result = base_result()
    n = len(result.times_h)
    assert len(result.c_nasal_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


def test_times_monotonic():
    result = base_result()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# --- Nasal compartment behavior ---


def test_nasal_starts_high():
    result = base_result()
    # Initial nasal concentration = dose_mg / v_nasal_mL = 10 / 0.4 = 25 mg/mL
    assert result.c_nasal_mg_mL[0] == pytest.approx(25.0, rel=1e-3)


def test_nasal_decreases_overall():
    result = base_result()
    assert result.c_nasal_mg_mL[-1] < result.c_nasal_mg_mL[0]


# --- Systemic compartment behavior ---


def test_systemic_starts_at_zero():
    result = base_result()
    assert result.c_systemic_mg_L[0] == 0.0


def test_systemic_rises_then_falls():
    result = base_result()
    c = result.c_systemic_mg_L
    tmax_idx = c.index(max(c))
    # Check it rises from 0 to max
    assert c[tmax_idx] > c[0]
    # Check it falls after tmax
    assert c[-1] < c[tmax_idx]


# --- Derived PK parameters ---


def test_cmax_positive():
    result = base_result()
    assert result.cmax_systemic_mg_L > 0.0


def test_auc_positive():
    result = base_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_t_half_positive():
    result = base_result()
    assert result.t_half_systemic_h > 0.0


def test_t_half_formula():
    # t_half = 0.693 * Vd / CL
    result = simulate_intranasal_pk(
        drug_name="X",
        dose_mg=5.0,
        ka_nasal_per_h=1.0,
        k_clearance_per_h=0.0,
        cl_sys_L_per_h=4.0,
        vd_sys_L=40.0,
    )
    expected_t_half = 0.693 * 40.0 / 4.0
    assert result.t_half_systemic_h == pytest.approx(expected_t_half, rel=1e-3)


# --- Bioavailability ---


def test_bioavailability_between_0_and_100():
    result = base_result()
    assert 0.0 <= result.bioavailability_pct <= 100.0


def test_zero_clearance_full_bioavailability():
    result = simulate_intranasal_pk(
        drug_name="X",
        dose_mg=10.0,
        ka_nasal_per_h=2.0,
        k_clearance_per_h=0.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
    )
    assert result.bioavailability_pct == pytest.approx(100.0, rel=1e-3)


def test_higher_clearance_lower_bioavailability():
    r1 = simulate_intranasal_pk("X", 10.0, 2.0, 0.1, 5.0, 20.0)
    r2 = simulate_intranasal_pk("X", 10.0, 2.0, 2.0, 5.0, 20.0)
    assert r1.bioavailability_pct > r2.bioavailability_pct


# --- Dose linearity ---


def test_dose_linearity_cmax():
    r1 = simulate_intranasal_pk("X", 10.0, 2.0, 0.5, 5.0, 20.0)
    r2 = simulate_intranasal_pk("X", 20.0, 2.0, 0.5, 5.0, 20.0)
    assert r2.cmax_systemic_mg_L == pytest.approx(2.0 * r1.cmax_systemic_mg_L, rel=1e-3)


def test_dose_linearity_auc():
    r1 = simulate_intranasal_pk("X", 10.0, 2.0, 0.5, 5.0, 20.0)
    r2 = simulate_intranasal_pk("X", 20.0, 2.0, 0.5, 5.0, 20.0)
    assert r2.auc_systemic_mg_h_per_L == pytest.approx(2.0 * r1.auc_systemic_mg_h_per_L, rel=1e-3)


# --- Higher ka -> higher cmax ---


def test_higher_ka_higher_cmax():
    r_slow = simulate_intranasal_pk("X", 10.0, 0.5, 0.5, 5.0, 20.0)
    r_fast = simulate_intranasal_pk("X", 10.0, 4.0, 0.5, 5.0, 20.0)
    assert r_fast.cmax_systemic_mg_L > r_slow.cmax_systemic_mg_L


# --- compare_intranasal_clearance ---


def test_compare_returns_list():
    results = compare_intranasal_clearance("TestDrug", 10.0, 2.0, [0.0, 0.5, 1.0, 2.0], 5.0, 20.0)
    assert isinstance(results, list)


def test_compare_correct_length():
    rates = [0.0, 0.5, 1.0, 2.0]
    results = compare_intranasal_clearance("X", 10.0, 2.0, rates, 5.0, 20.0)
    assert len(results) == len(rates)


def test_compare_sorted_by_cmax_descending():
    rates = [0.0, 0.5, 1.0, 2.0]
    results = compare_intranasal_clearance("X", 10.0, 2.0, rates, 5.0, 20.0)
    for i in range(1, len(results)):
        assert results[i - 1].cmax_systemic_mg_L >= results[i].cmax_systemic_mg_L


def test_compare_all_intranasal_routes():
    results = compare_intranasal_clearance("X", 10.0, 2.0, [0.0, 1.0], 5.0, 20.0)
    for r in results:
        assert r.route == "intranasal"


# --- Validation errors ---


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", 0.0, 2.0, 0.5, 5.0, 20.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", -5.0, 2.0, 0.5, 5.0, 20.0)


def test_validation_ka_zero():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", 10.0, 0.0, 0.5, 5.0, 20.0)


def test_validation_ka_negative():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", 10.0, -1.0, 0.5, 5.0, 20.0)


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", 10.0, 2.0, 0.5, 0.0, 20.0)


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError):
        simulate_intranasal_pk("X", 10.0, 2.0, 0.5, 5.0, 0.0)
