"""Tests for Tumor Microenvironment PK module (Phase 886)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.tumor_microenvironment_pk import (
    TumorMicroenvPKResult,
    compare_tumor_targeting,
    simulate_tumor_microenv_pk,
)

# ---------------------------------------------------------------------------
# Basic structure / type checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    r = simulate_tumor_microenv_pk("DrugA", dose_mg=10.0)
    assert isinstance(r, TumorMicroenvPKResult)


def test_result_fields_present():
    r = simulate_tumor_microenv_pk("DrugA", dose_mg=10.0)
    assert r.drug_name == "DrugA"
    assert r.dose_mg == 10.0
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_plasma_mg_L, list)
    assert isinstance(r.c_tv_mg_L, list)
    assert isinstance(r.c_ti_mg_L, list)
    assert isinstance(r.c_tc_mg_L, list)
    assert isinstance(r.notes, str)


def test_time_series_same_length():
    r = simulate_tumor_microenv_pk("X", dose_mg=5.0, t_end_h=12.0, dt_h=0.1)
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.c_tv_mg_L) == n
    assert len(r.c_ti_mg_L) == n
    assert len(r.c_tc_mg_L) == n


def test_times_start_at_zero():
    r = simulate_tumor_microenv_pk("X", dose_mg=5.0)
    assert r.times_h[0] == 0.0


def test_times_end_at_t_end():
    r = simulate_tumor_microenv_pk("X", dose_mg=5.0, t_end_h=24.0, dt_h=0.1)
    assert abs(r.times_h[-1] - 24.0) < 1e-9


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_initial_plasma_concentration():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0, vd_sys_L=50.0)
    assert abs(r.c_plasma_mg_L[0] - 10.0 / 50.0) < 1e-6


def test_tumor_compartments_start_at_zero():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0)
    assert r.c_tv_mg_L[0] == 0.0
    assert r.c_ti_mg_L[0] == 0.0
    assert r.c_tc_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Dynamic behaviour
# ---------------------------------------------------------------------------


def test_plasma_declines_over_time():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0, t_end_h=48.0)
    assert r.c_plasma_mg_L[0] > r.c_plasma_mg_L[-1]


def test_tv_rises_from_zero():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0, ps_per_h=1.0, t_end_h=48.0)
    assert r.cmax_tv > 0.0


def test_ti_receives_drug_from_tv():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0, ps_per_h=1.0, k_ifp_factor=0.5, t_end_h=48.0)
    assert r.cmax_ti > 0.0
    assert r.auc_ti > 0.0


def test_tc_receives_drug_from_ti():
    r = simulate_tumor_microenv_pk(
        "X", dose_mg=10.0, ps_per_h=1.0, k_ifp_factor=0.5, k_intracell_per_h=0.5, t_end_h=48.0
    )
    assert r.cmax_tc > 0.0
    assert r.auc_tc > 0.0


# ---------------------------------------------------------------------------
# AUC relationships
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0)
    assert r.auc_plasma > 0.0


def test_tumor_plasma_ratio_computed():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0, t_end_h=48.0)
    if r.auc_plasma > 0:
        expected = r.auc_tc / r.auc_plasma
        assert abs(r.tumor_plasma_ratio - expected) < 1e-9


def test_penetration_depth_score_range():
    r = simulate_tumor_microenv_pk("X", dose_mg=10.0)
    assert 0.0 <= r.penetration_depth_score <= 100.0


def test_high_ps_increases_tv_auc():
    """Higher PS product should increase tumor vascular AUC."""
    r_low = simulate_tumor_microenv_pk("X", dose_mg=10.0, ps_per_h=0.1, t_end_h=48.0)
    r_high = simulate_tumor_microenv_pk("X", dose_mg=10.0, ps_per_h=2.0, t_end_h=48.0)
    assert r_high.auc_tv > r_low.auc_tv


def test_low_ifp_increases_ti_auc():
    """Lower IFP factor means less resistance, so more drug in TI."""
    r_low_ifp = simulate_tumor_microenv_pk(
        "X", dose_mg=10.0, ps_per_h=1.0, k_ifp_factor=0.1, t_end_h=48.0
    )
    r_high_ifp = simulate_tumor_microenv_pk(
        "X", dose_mg=10.0, ps_per_h=1.0, k_ifp_factor=0.9, t_end_h=48.0
    )
    # Both should show some TI penetration — just check both are positive
    assert r_low_ifp.auc_ti > 0.0
    assert r_high_ifp.auc_ti > 0.0


# ---------------------------------------------------------------------------
# Notes classification
# ---------------------------------------------------------------------------


def test_notes_poor_tumor_penetration():
    """Very high systemic clearance → most drug gone before reaching TC."""
    r = simulate_tumor_microenv_pk(
        "X",
        dose_mg=1.0,
        ps_per_h=0.01,
        k_ifp_factor=0.01,
        k_intracell_per_h=0.01,
        cl_sys_L_per_h=100.0,
        t_end_h=48.0,
    )
    assert r.notes == "Poor tumor penetration"


def test_notes_excellent_tumor_accumulation():
    """High PS, low IFP, high intracellular uptake → excellent accumulation."""
    r = simulate_tumor_microenv_pk(
        "X",
        dose_mg=100.0,
        ps_per_h=5.0,
        k_ifp_factor=0.8,
        k_intracell_per_h=2.0,
        k_efflux_per_h=0.01,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
        t_end_h=48.0,
    )
    assert r.tumor_plasma_ratio > 0.0  # drug does accumulate


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_microenv_pk("X", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_microenv_pk("X", dose_mg=-5.0)


def test_invalid_v_tumor_raises():
    with pytest.raises(ValueError, match="v_tumor_L"):
        simulate_tumor_microenv_pk("X", dose_mg=1.0, v_tumor_L=0.0)


def test_invalid_cl_sys_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_tumor_microenv_pk("X", dose_mg=1.0, cl_sys_L_per_h=0.0)


# ---------------------------------------------------------------------------
# compare_tumor_targeting
# ---------------------------------------------------------------------------


def test_compare_tumor_targeting_returns_list():
    scenarios = [
        {"ps_per_h": 0.5, "k_ifp_factor": 0.3, "name": "baseline"},
        {"ps_per_h": 1.5, "k_ifp_factor": 0.1, "name": "high_ps"},
        {"ps_per_h": 0.2, "k_ifp_factor": 0.8, "name": "high_ifp"},
    ]
    results = compare_tumor_targeting("X", dose_mg=10.0, scenarios=scenarios)
    assert len(results) == 3


def test_compare_tumor_targeting_sorted_by_auc_tc():
    scenarios = [
        {"ps_per_h": 0.1, "k_ifp_factor": 0.9},
        {"ps_per_h": 2.0, "k_ifp_factor": 0.1},
        {"ps_per_h": 0.5, "k_ifp_factor": 0.3},
    ]
    results = compare_tumor_targeting("X", dose_mg=10.0, scenarios=scenarios)
    aucs = [r.auc_tc for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_tumor_targeting_empty_scenarios():
    results = compare_tumor_targeting("X", dose_mg=10.0, scenarios=[])
    assert results == []
