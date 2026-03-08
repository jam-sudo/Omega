"""Tests for Phase 427: Spinal cord PK simulation."""

import pytest

from omega_pbpk.core.spinal_cord_pk import (
    SpinalCordPKResult,
    compare_spinal_routes,
    simulate_spinal_cord_pk,
)


# --- Helper ---
def default_iv():
    return simulate_spinal_cord_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="iv",
        cl_sys_L_per_h=1.0,
        vd_plasma_L=5.0,
    )


def default_it():
    return simulate_spinal_cord_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="intrathecal",
        cl_sys_L_per_h=1.0,
        vd_plasma_L=5.0,
    )


def default_oral():
    return simulate_spinal_cord_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="oral",
        cl_sys_L_per_h=1.0,
        vd_plasma_L=5.0,
    )


# --- Return type ---
def test_return_type_iv():
    result = default_iv()
    assert isinstance(result, SpinalCordPKResult)


def test_return_type_intrathecal():
    result = default_it()
    assert isinstance(result, SpinalCordPKResult)


def test_return_type_oral():
    result = default_oral()
    assert isinstance(result, SpinalCordPKResult)


# --- Array consistency ---
def test_times_start_at_zero_iv():
    result = default_iv()
    assert result.times_h[0] == 0.0


def test_arrays_consistent_length():
    result = default_iv()
    n = len(result.times_h)
    assert len(result.c_csf_mg_mL) == n
    assert len(result.c_spinal_cord_mg_L) == n
    assert len(result.c_systemic_mg_L) == n


def test_times_monotonically_increasing():
    result = default_iv()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# --- Initial conditions ---
def test_intrathecal_csf_starts_high():
    result = default_it()
    # CSF concentration at t=0 should be dose / V_csf / 1000
    # 10 mg / 0.15 L / 1000 = 0.0667 mg/mL
    assert result.c_csf_mg_mL[0] > 0.05


def test_intrathecal_plasma_starts_near_zero():
    result = default_it()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_iv_plasma_starts_at_dose_over_vd():
    result = default_iv()
    # dose=10, vd=5 -> C(0) = 2.0 mg/L
    assert result.c_systemic_mg_L[0] == pytest.approx(2.0, rel=1e-6)


def test_iv_csf_starts_near_zero():
    result = default_iv()
    assert result.c_csf_mg_mL[0] == pytest.approx(0.0, abs=1e-9)


# --- cmax / auc positivity ---
def test_cmax_csf_positive_iv():
    result = default_iv()
    assert result.cmax_csf_mg_mL > 0.0


def test_cmax_csf_positive_intrathecal():
    result = default_it()
    assert result.cmax_csf_mg_mL > 0.0


def test_cmax_spinal_cord_positive_iv():
    result = default_iv()
    assert result.cmax_spinal_cord_mg_L > 0.0


def test_cmax_spinal_cord_positive_intrathecal():
    result = default_it()
    assert result.cmax_spinal_cord_mg_L > 0.0


def test_auc_csf_positive():
    result = default_iv()
    assert result.auc_csf_mg_h_per_mL > 0.0


def test_auc_spinal_cord_positive():
    result = default_iv()
    assert result.auc_spinal_cord_mg_h_per_L > 0.0


# --- t_half_csf ---
def test_t_half_csf_positive():
    result = default_iv()
    assert result.t_half_csf_h > 0.0


def test_t_half_csf_formula():
    # k_csf_to_plasma=0.3, k_csf_to_sc=0.1 -> total=0.4 -> t_half = 0.693/0.4
    result = default_iv()
    expected = 0.693 / (0.3 + 0.1)
    assert result.t_half_csf_h == pytest.approx(expected, rel=1e-4)


# --- Route comparison ---
def test_intrathecal_higher_auc_csf_than_iv():
    it_result = default_it()
    iv_result = default_iv()
    assert it_result.auc_csf_mg_h_per_mL > iv_result.auc_csf_mg_h_per_mL


# --- compare_spinal_routes ---
def test_compare_routes_returns_three():
    results = compare_spinal_routes("Drug", 10.0, 1.0, 5.0)
    assert len(results) == 3


def test_compare_routes_sorted_descending():
    results = compare_spinal_routes("Drug", 10.0, 1.0, 5.0)
    aucs = [r.auc_spinal_cord_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_routes_all_types():
    results = compare_spinal_routes("Drug", 10.0, 1.0, 5.0)
    routes = {r.route for r in results}
    assert routes == {"intrathecal", "iv", "oral"}


# --- Validation errors ---
def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spinal_cord_pk("Drug", dose_mg=0.0, route="iv", cl_sys_L_per_h=1.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spinal_cord_pk("Drug", dose_mg=-5.0, route="iv", cl_sys_L_per_h=1.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_spinal_cord_pk("Drug", dose_mg=10.0, route="subcutaneous", cl_sys_L_per_h=1.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_spinal_cord_pk("Drug", dose_mg=10.0, route="iv", cl_sys_L_per_h=0.0)
