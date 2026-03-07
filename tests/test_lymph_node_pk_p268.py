"""Tests for lymph node PK model — Phase 268."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lymph_node_pk_p268 import (
    LymphNodePKResult,
    compare_admin_routes,
    simulate_lymph_node_pk,
)


def _run(route="sc_inguinal", **kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        administration_route=route,
        cl_L_per_h=1.0,
    )
    defaults.update(kwargs)
    return simulate_lymph_node_pk(**defaults)


# --- Return type ---


def test_returns_result_instance():
    r = _run()
    assert isinstance(r, LymphNodePKResult)


def test_all_routes_return_result():
    for route in ("sc_inguinal", "sc_axillary", "im", "iv"):
        r = _run(route=route)
        assert isinstance(r, LymphNodePKResult)


# --- Plasma initial conditions ---


def test_iv_plasma_starts_high():
    r = _run(route="iv")
    # IV: C_plasma(0) = dose / V_plasma
    assert r.c_plasma_mg_L[0] > 1.0  # dose=10, V=3 => ~3.33 mg/L


def test_sc_plasma_starts_near_zero():
    r = _run(route="sc_inguinal")
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# --- Lymph starts near zero ---


def test_lymph_starts_near_zero():
    for route in ("sc_inguinal", "sc_axillary", "im", "iv"):
        r = _run(route=route)
        assert r.c_lymph_mg_L[0] == pytest.approx(0.0, abs=1e-9), (
            f"c_lymph[0] should be 0 for {route}"
        )


# --- cmax_lymph > 0 ---


def test_cmax_lymph_positive_sc():
    r = _run(route="sc_inguinal")
    assert r.cmax_lymph_mg_L > 0


def test_cmax_lymph_positive_im():
    r = _run(route="im")
    assert r.cmax_lymph_mg_L > 0


def test_cmax_lymph_positive_iv():
    r = _run(route="iv")
    assert r.cmax_lymph_mg_L > 0


# --- SC inguinal gives more lymph targeting than IV ---


def test_sc_inguinal_more_lymph_than_iv():
    sc = _run(route="sc_inguinal")
    iv = _run(route="iv")
    assert sc.lymph_targeting_efficiency_pct > iv.lymph_targeting_efficiency_pct


# --- AUC values positive ---


def test_auc_plasma_positive():
    r = _run()
    assert r.auc_plasma_mg_h_per_L > 0


def test_auc_lymph_positive():
    r = _run()
    assert r.auc_lymph_mg_h_per_L > 0


# --- Times and lists same length ---


def test_times_and_lists_same_length():
    r = _run()
    assert len(r.times_h) == len(r.c_plasma_mg_L)
    assert len(r.times_h) == len(r.c_lymph_mg_L)


# --- compare_admin_routes ---


def test_compare_returns_four_results():
    results = compare_admin_routes("DrugA", 10.0, 1.0)
    assert len(results) == 4


def test_compare_all_different_routes():
    results = compare_admin_routes("DrugA", 10.0, 1.0)
    routes = {r.administration_route for r in results}
    assert len(routes) == 4


def test_compare_sorted_by_efficiency_descending():
    results = compare_admin_routes("DrugA", 10.0, 1.0)
    for i in range(1, len(results)):
        assert (
            results[i].lymph_targeting_efficiency_pct
            <= results[i - 1].lymph_targeting_efficiency_pct
        )


# --- lymph_plasma_ratio positive for SC ---


def test_lymph_plasma_ratio_positive():
    r = _run(route="sc_inguinal")
    assert r.lymph_plasma_ratio > 0


# --- Validation errors ---


def test_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-5.0)


def test_cl_le_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="administration_route"):
        _run(route="bogus_route")
