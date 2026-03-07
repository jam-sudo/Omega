"""Tests for Phase 989 — portal_vein_pk module."""

import pytest
from omega_pbpk.core.portal_vein_pk import (
    PortalVeinPKResult,
    simulate_portal_vein_pk,
)


def default_result():
    return simulate_portal_vein_pk("TestDrug", 100.0)


# --- Return type ---

def test_returns_result_type():
    r = default_result()
    assert isinstance(r, PortalVeinPKResult)


# --- Basic positive-valued outputs ---

def test_cmax_portal_positive():
    r = default_result()
    assert r.cmax_portal_mg_L > 0.0


def test_cmax_systemic_positive():
    r = default_result()
    assert r.cmax_systemic_mg_L > 0.0


def test_auc_portal_positive():
    r = default_result()
    assert r.auc_portal_mg_h_per_L > 0.0


def test_auc_systemic_positive():
    r = default_result()
    assert r.auc_systemic_mg_h_per_L > 0.0


# --- Fraction bounds ---

def test_f_gut_in_unit_interval():
    r = default_result()
    assert 0.0 < r.f_gut <= 1.0


def test_f_hepatic_in_unit_interval():
    r = default_result()
    assert 0.0 < r.f_hepatic <= 1.0


def test_bioavailability_in_unit_interval():
    r = default_result()
    assert 0.0 < r.bioavailability_f <= 1.0


def test_bioavailability_product():
    r = default_result()
    assert abs(r.bioavailability_f - r.f_gut * r.f_hepatic) < 1e-9


# --- Ratios ---

def test_portal_to_systemic_ratio_positive():
    r = default_result()
    assert r.portal_to_systemic_ratio > 0.0


# --- Initial conditions ---

def test_c_portal_zero_at_t0():
    r = default_result()
    assert r.c_portal_mg_L[0] == pytest.approx(0.0)


def test_c_systemic_zero_at_t0():
    r = default_result()
    assert r.c_systemic_mg_L[0] == pytest.approx(0.0)


# --- High extraction sensitivity ---

def test_high_clint_liver_lowers_f_hepatic():
    r_low = simulate_portal_vein_pk("Drug", 100.0, clint_liver_mL_min_per_g=1.0)
    r_high = simulate_portal_vein_pk("Drug", 100.0, clint_liver_mL_min_per_g=200.0)
    assert r_high.f_hepatic < r_low.f_hepatic


def test_high_clint_gut_lowers_f_gut():
    r_low = simulate_portal_vein_pk("Drug", 100.0, clint_gut_mL_min=1.0)
    r_high = simulate_portal_vein_pk("Drug", 100.0, clint_gut_mL_min=500.0)
    assert r_high.f_gut < r_low.f_gut


# --- Validation errors ---

def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", dose_mg=-50.0)


def test_invalid_v_portal_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", 100.0, v_portal_L=0.0)


def test_invalid_v_liver_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", 100.0, v_liver_L=-1.0)


def test_invalid_cl_systemic_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", 100.0, cl_systemic_L_per_h=0.0)


def test_invalid_vd_systemic_raises():
    with pytest.raises(ValueError):
        simulate_portal_vein_pk("Drug", 100.0, vd_systemic_L=0.0)
