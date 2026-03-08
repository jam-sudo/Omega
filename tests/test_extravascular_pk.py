"""Tests for Phase 232: Extravascular PK Model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.extravascular_pk import (
    ExtravascularPKResult,
    compare_extravascular_routes,
    simulate_extravascular_pk,
)


def _sim(route="subcutaneous", **kw) -> ExtravascularPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        route=route,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_extravascular_pk(**defaults)


# --- Return type ---


def test_return_type():
    assert isinstance(_sim(), ExtravascularPKResult)


# --- Plasma starts at zero ---


def test_plasma_starts_at_zero():
    r = _sim()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


# --- Cmax positive ---


def test_cmax_positive():
    assert _sim().cmax_mg_L > 0.0


# --- Tmax positive ---


def test_tmax_positive():
    assert _sim().tmax_h > 0.0


# --- AUC positive ---


def test_auc_positive():
    assert _sim().auc_mg_h_per_L > 0.0


# --- Half-lives ---


def test_t_half_absorption_positive():
    r = _sim()
    assert r.t_half_absorption_h > 0.0
    assert not math.isnan(r.t_half_absorption_h)


def test_t_half_elimination_positive():
    r = _sim()
    assert r.t_half_elimination_h > 0.0


# --- Dose linearity ---


def test_dose_linearity_cmax():
    r1 = _sim(dose_mg=5.0)
    r2 = _sim(dose_mg=10.0)
    assert r2.cmax_mg_L == pytest.approx(r1.cmax_mg_L * 2.0, rel=0.02)


def test_dose_linearity_auc():
    r1 = _sim(dose_mg=5.0)
    r2 = _sim(dose_mg=10.0)
    assert r2.auc_mg_h_per_L == pytest.approx(r1.auc_mg_h_per_L * 2.0, rel=0.02)


# --- CL effect on AUC ---


def test_higher_cl_lower_auc():
    r_low = _sim(cl_L_per_h=2.0)
    r_high = _sim(cl_L_per_h=10.0)
    assert r_high.auc_mg_h_per_L < r_low.auc_mg_h_per_L


# --- Routes ---


def test_all_routes_valid():
    for route in ("subcutaneous", "intramuscular", "intraperitoneal", "sublingual"):
        r = _sim(route=route)
        assert r.cmax_mg_L > 0.0


def test_im_higher_f_than_sc():
    r_sc = _sim(route="subcutaneous")
    r_im = _sim(route="intramuscular")
    assert r_im.bioavailability > r_sc.bioavailability


def test_sublingual_fastest_absorption():
    r_sl = _sim(route="sublingual")
    r_sc = _sim(route="subcutaneous")
    assert r_sl.t_half_absorption_h < r_sc.t_half_absorption_h


# --- compare_extravascular_routes ---


def test_compare_returns_4():
    results = compare_extravascular_routes("D", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_descending():
    results = compare_extravascular_routes("D", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_routes_present():
    results = compare_extravascular_routes("D", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0)
    routes = {r.route for r in results}
    assert routes == {"subcutaneous", "intramuscular", "intraperitoneal", "sublingual"}


# --- Time array ---


def test_times_start_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0)


def test_time_arrays_consistent():
    r = _sim()
    assert len(r.times_h) == len(r.c_plasma_mg_L)


# --- Validation errors ---


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_extravascular_pk("D", dose_mg=0.0, route="subcutaneous", cl_L_per_h=5.0, vd_L=50.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_extravascular_pk(
            "D", dose_mg=10.0, route="subcutaneous", cl_L_per_h=0.0, vd_L=50.0
        )


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_extravascular_pk("D", dose_mg=10.0, route="subcutaneous", cl_L_per_h=5.0, vd_L=0.0)


def test_error_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_extravascular_pk("D", dose_mg=10.0, route="intravenous", cl_L_per_h=5.0, vd_L=50.0)
