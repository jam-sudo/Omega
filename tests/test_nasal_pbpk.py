"""Tests for nasal drug absorption PBPK model (Phase 675)."""

import math

import pytest

from omega_pbpk.core.nasal_pbpk import NasalPKResult, compare_nasal_routes, simulate_nasal_pk

# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_nasal_pk_result():
    r = simulate_nasal_pk("sumatriptan", dose_mg=20.0)
    assert isinstance(r, NasalPKResult)


def test_result_has_correct_drug_name():
    r = simulate_nasal_pk("desmopressin", dose_mg=0.1)
    assert r.drug_name == "desmopressin"


def test_result_has_correct_dose():
    r = simulate_nasal_pk("TestDrug", dose_mg=5.0)
    assert r.dose_mg == 5.0


# ---------------------------------------------------------------------------
# Time-course correctness
# ---------------------------------------------------------------------------


def test_a_nasal_starts_at_dose():
    dose = 10.0
    r = simulate_nasal_pk("TestDrug", dose_mg=dose)
    assert r.a_nasal_mg[0] == pytest.approx(dose, rel=1e-9)


def test_a_nasal_decreases_monotonically():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    for i in range(1, len(r.a_nasal_mg)):
        assert r.a_nasal_mg[i] <= r.a_nasal_mg[i - 1] + 1e-9


def test_a_nasal_approaches_zero():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0, t_end_h=24.0)
    assert r.a_nasal_mg[-1] < 0.01


def test_c_plasma_starts_at_zero():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_times_first_value_is_zero():
    r = simulate_nasal_pk("TestDrug", dose_mg=5.0)
    assert r.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert r.cmax_mg_L > 0


def test_tmax_positive():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert r.tmax_h >= 0.0


def test_auc_positive():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert r.auc_mg_h_per_L > 0


def test_t_half_matches_vd_cl():
    cl = 5.0
    vd = 50.0
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0, cl_L_per_h=cl, vd_L=vd)
    expected = math.log(2.0) * vd / cl
    assert r.t_half_h == pytest.approx(expected, rel=1e-6)


def test_linearity_double_dose_doubles_cmax():
    r1 = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    r2 = simulate_nasal_pk("TestDrug", dose_mg=20.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-6)


def test_linearity_double_dose_doubles_auc():
    r1 = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    r2 = simulate_nasal_pk("TestDrug", dose_mg=20.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-6)


# ---------------------------------------------------------------------------
# Fraction calculations
# ---------------------------------------------------------------------------


def test_f_nasal_plus_f_swallowed_equals_one():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0, ka_nasal_per_h=1.0, k_mc_per_h=1.0)
    assert r.f_nasal + r.f_swallowed == pytest.approx(1.0, rel=1e-9)


def test_f_systemic_total_leq_one():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert r.f_systemic_total <= 1.0 + 1e-9


def test_f_systemic_total_formula():
    r = simulate_nasal_pk(
        "TestDrug",
        dose_mg=10.0,
        ka_nasal_per_h=2.0,
        k_mc_per_h=1.0,
        f_oral_swallowed=0.3,
    )
    expected = r.f_nasal + r.f_swallowed * 0.3
    assert r.f_systemic_total == pytest.approx(expected, rel=1e-9)


def test_high_k_mc_reduces_f_nasal():
    r_low = simulate_nasal_pk("TestDrug", dose_mg=10.0, k_mc_per_h=0.5)
    r_high = simulate_nasal_pk("TestDrug", dose_mg=10.0, k_mc_per_h=2.0)
    assert r_high.f_nasal < r_low.f_nasal


def test_high_ka_nasal_reduces_tmax():
    r_slow = simulate_nasal_pk("TestDrug", dose_mg=10.0, ka_nasal_per_h=0.5)
    r_fast = simulate_nasal_pk("TestDrug", dose_mg=10.0, ka_nasal_per_h=3.0)
    assert r_fast.tmax_h <= r_slow.tmax_h


# ---------------------------------------------------------------------------
# compare_nasal_routes
# ---------------------------------------------------------------------------


def test_compare_nasal_routes_returns_list():
    scenarios = [
        {"label": "Scenario A", "ka_nasal_per_h": 1.0},
        {"label": "Scenario B", "ka_nasal_per_h": 2.0},
        {"label": "Scenario C", "k_mc_per_h": 0.5},
    ]
    results = compare_nasal_routes("TestDrug", 10.0, scenarios)
    assert isinstance(results, list)


def test_compare_nasal_routes_correct_length():
    scenarios = [{"ka_nasal_per_h": 1.0}, {"ka_nasal_per_h": 2.0}]
    results = compare_nasal_routes("TestDrug", 10.0, scenarios)
    assert len(results) == 2


def test_compare_nasal_routes_returns_nasal_pk_results():
    scenarios = [{"ka_nasal_per_h": 1.0}]
    results = compare_nasal_routes("TestDrug", 10.0, scenarios)
    assert isinstance(results[0], NasalPKResult)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = simulate_nasal_pk("TestDrug", dose_mg=10.0)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_pk("TestDrug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_pk("TestDrug", dose_mg=-5.0)


def test_zero_ka_nasal_raises():
    with pytest.raises(ValueError, match="ka_nasal_per_h"):
        simulate_nasal_pk("TestDrug", dose_mg=10.0, ka_nasal_per_h=0.0)


def test_zero_k_mc_raises():
    with pytest.raises(ValueError, match="k_mc_per_h"):
        simulate_nasal_pk("TestDrug", dose_mg=10.0, k_mc_per_h=0.0)


def test_zero_ka_sw_raises():
    with pytest.raises(ValueError, match="ka_sw_per_h"):
        simulate_nasal_pk("TestDrug", dose_mg=10.0, ka_sw_per_h=0.0)


def test_invalid_f_oral_swallowed_zero_raises():
    with pytest.raises(ValueError, match="f_oral_swallowed"):
        simulate_nasal_pk("TestDrug", dose_mg=10.0, f_oral_swallowed=0.0)


def test_invalid_f_oral_swallowed_gt_one_raises():
    with pytest.raises(ValueError, match="f_oral_swallowed"):
        simulate_nasal_pk("TestDrug", dose_mg=10.0, f_oral_swallowed=1.5)
