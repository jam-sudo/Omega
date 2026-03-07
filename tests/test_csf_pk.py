"""Tests for CSF PK module (Phase 885)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.csf_pk import CSFPKResult, compare_csf_routes, simulate_csf_pk


# ---------------------------------------------------------------------------
# Basic structure / type checks
# ---------------------------------------------------------------------------


def test_returns_csf_pk_result():
    result = simulate_csf_pk("TestDrug", dose_mg=1.0)
    assert isinstance(result, CSFPKResult)


def test_result_fields_present():
    r = simulate_csf_pk("TestDrug", dose_mg=1.0)
    assert r.drug_name == "TestDrug"
    assert r.dose_mg == 1.0
    assert r.route == "intrathecal"
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_csf_mg_L, list)
    assert isinstance(r.c_brain_mg_L, list)
    assert isinstance(r.c_plasma_mg_L, list)
    assert isinstance(r.notes, str)


def test_time_series_same_length():
    r = simulate_csf_pk("X", dose_mg=1.0, t_end_h=10.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_csf_mg_L) == len(r.c_brain_mg_L) == len(r.c_plasma_mg_L)


def test_times_start_at_zero():
    r = simulate_csf_pk("X", dose_mg=1.0)
    assert r.times_h[0] == 0.0


def test_times_end_at_t_end():
    r = simulate_csf_pk("X", dose_mg=5.0, t_end_h=12.0, dt_h=0.1)
    assert abs(r.times_h[-1] - 12.0) < 1e-9


# ---------------------------------------------------------------------------
# Initial conditions per route
# ---------------------------------------------------------------------------


def test_intrathecal_initial_csf_concentration():
    """At t=0, all drug is in CSF for intrathecal route."""
    r = simulate_csf_pk("X", dose_mg=2.0, route="intrathecal", v_csf_L=0.14)
    expected = 2.0 / 0.14
    assert abs(r.c_csf_mg_L[0] - expected) < 1e-6


def test_icv_initial_csf_concentration():
    """ICV route also starts with all drug in CSF."""
    r = simulate_csf_pk("X", dose_mg=2.0, route="ICV", v_csf_L=0.14)
    expected = 2.0 / 0.14
    assert abs(r.c_csf_mg_L[0] - expected) < 1e-6


def test_systemic_initial_plasma_concentration():
    """Systemic route starts with all drug in plasma."""
    r = simulate_csf_pk("X", dose_mg=10.0, route="systemic", vd_sys_L=50.0)
    expected = 10.0 / 50.0
    assert abs(r.c_plasma_mg_L[0] - expected) < 1e-6
    assert r.c_csf_mg_L[0] == 0.0
    assert r.c_brain_mg_L[0] == 0.0


def test_systemic_csf_rises_from_zero():
    """For systemic route, CSF concentration should rise from 0 and become positive."""
    r = simulate_csf_pk("X", dose_mg=10.0, route="systemic", bbb_perm_per_h=0.05,
                         t_end_h=24.0, dt_h=0.05)
    assert r.cmax_csf_mg_L > 0.0


# ---------------------------------------------------------------------------
# Pharmacokinetic properties
# ---------------------------------------------------------------------------


def test_csf_concentration_declines_after_intrathecal():
    """CSF concentration should decrease over time after intrathecal dose."""
    r = simulate_csf_pk("X", dose_mg=1.0, route="intrathecal", t_end_h=24.0, dt_h=0.05)
    assert r.c_csf_mg_L[0] > r.c_csf_mg_L[-1]


def test_brain_concentration_rises_then_falls():
    """Brain concentration should be 0 at t=0 then increase after intrathecal dose."""
    r = simulate_csf_pk("X", dose_mg=1.0, route="intrathecal", t_end_h=24.0, dt_h=0.05)
    assert r.c_brain_mg_L[0] == 0.0
    assert r.cmax_brain_mg_L > 0.0


def test_auc_csf_positive():
    r = simulate_csf_pk("X", dose_mg=1.0, t_end_h=24.0)
    assert r.auc_csf > 0.0


def test_auc_brain_positive_intrathecal():
    r = simulate_csf_pk("X", dose_mg=1.0, route="intrathecal", t_end_h=24.0)
    assert r.auc_brain > 0.0


def test_cmax_csf_equals_initial_for_intrathecal():
    """Cmax for CSF should be at t=0 for intrathecal (first-order decline)."""
    r = simulate_csf_pk("X", dose_mg=1.0, route="intrathecal")
    assert r.tmax_csf_h == 0.0
    assert abs(r.cmax_csf_mg_L - 1.0 / 0.14) < 1e-5


def test_csf_t_half_formula():
    """csf_t_half_h = ln(2) / k_csf_drain_per_h."""
    r = simulate_csf_pk("X", dose_mg=1.0, k_csf_drain_per_h=0.2)
    expected = math.log(2.0) / 0.2
    assert abs(r.csf_t_half_h - expected) < 1e-9


def test_brain_to_csf_ratio_computed():
    r = simulate_csf_pk("X", dose_mg=1.0, t_end_h=24.0)
    assert r.brain_to_csf_ratio == pytest.approx(r.auc_brain / r.auc_csf, rel=1e-6)


# ---------------------------------------------------------------------------
# Notes classification
# ---------------------------------------------------------------------------


def test_notes_high_brain_penetration():
    """High k_cp relative to k_pc should yield high brain penetration note."""
    r = simulate_csf_pk("X", dose_mg=1.0, k_cp_per_h=2.0, k_pc_per_h=0.01, t_end_h=48.0)
    assert r.notes == "High brain penetration from CSF"


def test_notes_limited_brain_penetration():
    """Very low k_cp should yield limited penetration note."""
    r = simulate_csf_pk("X", dose_mg=1.0, k_cp_per_h=0.001, k_pc_per_h=0.5, t_end_h=24.0)
    assert r.notes == "Limited brain penetration from CSF"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_pk("X", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_pk("X", dose_mg=-1.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_csf_pk("X", dose_mg=1.0, route="oral")


def test_invalid_v_csf_raises():
    with pytest.raises(ValueError, match="v_csf_L"):
        simulate_csf_pk("X", dose_mg=1.0, v_csf_L=0.0)


# ---------------------------------------------------------------------------
# compare_csf_routes
# ---------------------------------------------------------------------------


def test_compare_csf_routes_returns_three():
    results = compare_csf_routes("X", dose_mg=1.0, t_end_h=24.0)
    assert len(results) == 3


def test_compare_csf_routes_sorted_descending():
    results = compare_csf_routes("X", dose_mg=1.0, t_end_h=24.0)
    aucs = [r.auc_csf for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_csf_routes_covers_all_routes():
    results = compare_csf_routes("X", dose_mg=1.0, t_end_h=24.0)
    routes = {r.route for r in results}
    assert routes == {"intrathecal", "ICV", "systemic"}
