"""Tests for Phase 368 — Topical Drug Dermatokinetics Model."""

import math

import pytest

from omega_pbpk.core.dermatokinetics_model import (
    DermatokineticsResult,
    simulate_dermatokinetics,
)

# --- Baseline parameters ---
BASE = dict(
    drug_name="Hydrocortisone",
    applied_dose_mg=10.0,
    logP=1.6,
    mw_Da=362.0,
    k_formulation_per_h=0.5,
    k_sc_absorption_per_h=0.3,
    k_ve_transfer_per_h=0.4,
    k_dermis_to_sys_per_h=0.2,
    cl_sys_L_per_h=2.0,
    vd_sys_L=20.0,
    t_end_h=24.0,
    dt_h=0.05,
    area_cm2=100.0,
    v_sc_mL=0.5,
    v_ve_mL=2.0,
    v_dermis_mL=10.0,
)


def make_result(**overrides):
    params = {**BASE, **overrides}
    return simulate_dermatokinetics(**params)


# --- Return type ---
def test_returns_dermatokinetics_result():
    r = make_result()
    assert isinstance(r, DermatokineticsResult)


def test_result_has_list_fields():
    r = make_result()
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_sc_mg_mL, list)
    assert isinstance(r.c_dermis_mg_mL, list)
    assert isinstance(r.c_systemic_mg_L, list)


def test_array_lengths_consistent():
    r = make_result()
    n = len(r.times_h)
    assert len(r.c_formulation_mg_mL) == n
    assert len(r.c_sc_mg_mL) == n
    assert len(r.c_viable_epidermis_mg_mL) == n
    assert len(r.c_dermis_mg_mL) == n
    assert len(r.c_systemic_mg_L) == n


def test_times_start_at_zero():
    r = make_result()
    assert r.times_h[0] == 0.0


def test_times_end_near_t_end():
    r = make_result()
    assert math.isclose(r.times_h[-1], 24.0, rel_tol=1e-6)


# --- Initial conditions ---
def test_formulation_starts_at_dose():
    r = make_result()
    assert math.isclose(r.c_formulation_mg_mL[0], BASE["applied_dose_mg"], rel_tol=1e-9)


def test_sc_starts_at_zero():
    r = make_result()
    assert r.c_sc_mg_mL[0] == 0.0


def test_systemic_starts_at_zero():
    r = make_result()
    assert r.c_systemic_mg_L[0] == 0.0


# --- Dynamic behaviour ---
def test_formulation_decreases():
    r = make_result()
    assert r.c_formulation_mg_mL[-1] < r.c_formulation_mg_mL[0]


def test_sc_peaks_then_falls():
    r = make_result()
    cmax_idx = r.c_sc_mg_mL.index(max(r.c_sc_mg_mL))
    # SC should peak before the end
    assert 0 < cmax_idx < len(r.c_sc_mg_mL) - 1


def test_dermis_peaks_later_than_sc():
    r = make_result()
    tmax_sc = r.times_h[r.c_sc_mg_mL.index(max(r.c_sc_mg_mL))]
    tmax_derm = r.tmax_dermis_h
    assert tmax_derm > tmax_sc


# --- AUC and Cmax ---
def test_auc_dermis_positive():
    r = make_result()
    assert r.auc_dermis_mg_h_per_mL > 0


def test_auc_systemic_positive():
    r = make_result()
    assert r.auc_systemic_mg_h_per_L > 0


def test_cmax_dermis_equals_max_list():
    r = make_result()
    assert math.isclose(r.cmax_dermis_mg_mL, max(r.c_dermis_mg_mL), rel_tol=1e-9)


def test_cmax_systemic_equals_max_list():
    r = make_result()
    assert math.isclose(r.cmax_systemic_mg_L, max(r.c_systemic_mg_L), rel_tol=1e-9)


# --- ksc_factor ---
def test_higher_logp_higher_ksc_factor():
    r_low = make_result(logP=0.0, mw_Da=200.0)
    r_high = make_result(logP=3.0, mw_Da=200.0)
    assert r_high.ksc_factor > r_low.ksc_factor


def test_ksc_factor_clamped():
    # Extreme logP should be clamped to 10
    r = make_result(logP=10.0, mw_Da=50.0)
    assert r.ksc_factor <= 10.0
    # Extreme low logP + high MW should clamp to 0.01
    r2 = make_result(logP=-5.0, mw_Da=2000.0)
    assert r2.ksc_factor >= 0.01


# --- Dose linearity ---
def test_dose_linearity_auc_dermis():
    r1 = make_result(applied_dose_mg=5.0)
    r2 = make_result(applied_dose_mg=10.0)
    assert math.isclose(r2.auc_dermis_mg_h_per_mL / r1.auc_dermis_mg_h_per_mL, 2.0, rel_tol=0.01)


def test_dose_linearity_auc_systemic():
    r1 = make_result(applied_dose_mg=5.0)
    r2 = make_result(applied_dose_mg=10.0)
    assert math.isclose(r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L, 2.0, rel_tol=0.01)


# --- Validation errors ---
def test_error_dose_zero():
    with pytest.raises(ValueError, match="applied_dose_mg"):
        make_result(applied_dose_mg=0.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        make_result(cl_sys_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        make_result(vd_sys_L=0.0)


def test_error_k_formulation_zero():
    with pytest.raises(ValueError, match="k_formulation_per_h"):
        make_result(k_formulation_per_h=0.0)
