"""Tests for clinical/bioavailability_comparison.py (Phase 742)."""

import pytest

from omega_pbpk.clinical.bioavailability_comparison import (
    BiAvailRouteResult,
    compare_bioavailability,
    rank_routes,
)

# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_list_of_results():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    assert isinstance(results, list)
    assert all(isinstance(r, BiAvailRouteResult) for r in results)


def test_default_six_routes():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    routes = {r.route for r in results}
    assert routes == {"iv", "oral", "sc", "im", "rectal", "intranasal"}


# ---------------------------------------------------------------------------
# IV route properties
# ---------------------------------------------------------------------------


def test_iv_f_absolute_is_one():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    iv = next(r for r in results if r.route == "iv")
    assert abs(iv.f_absolute - 1.0) < 1e-9


def test_iv_relative_bioavailability_is_100():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    iv = next(r for r in results if r.route == "iv")
    assert abs(iv.relative_bioavailability_pct - 100.0) < 0.5


# ---------------------------------------------------------------------------
# Route F-ordering
# ---------------------------------------------------------------------------


def test_oral_f_less_than_iv():
    results = compare_bioavailability("TestDrug", dose_mg=100.0, fh=0.7, fa=0.9)
    iv = next(r for r in results if r.route == "iv")
    oral = next(r for r in results if r.route == "oral")
    assert oral.f_absolute < iv.f_absolute


def test_sc_f_less_than_oral_f():
    """SC F = 0.7*fh; oral F = fa*fh. With fa=0.9, oral > sc."""
    results = compare_bioavailability("TestDrug", dose_mg=100.0, fh=0.7, fa=0.9)
    sc = next(r for r in results if r.route == "sc")
    oral = next(r for r in results if r.route == "oral")
    assert sc.f_absolute < oral.f_absolute


def test_all_routes_cmax_positive():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    for r in results:
        assert r.cmax_mg_L > 0.0, f"cmax <= 0 for route {r.route}"


def test_all_routes_auc_positive():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    for r in results:
        assert r.auc_mg_h_per_L > 0.0, f"auc <= 0 for route {r.route}"


# ---------------------------------------------------------------------------
# Sorted by f_absolute descending
# ---------------------------------------------------------------------------


def test_results_sorted_by_f_absolute_descending():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    f_vals = [r.f_absolute for r in results]
    assert f_vals == sorted(f_vals, reverse=True)


def test_iv_first_in_sorted_results():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    assert results[0].route == "iv"


# ---------------------------------------------------------------------------
# rank_routes
# ---------------------------------------------------------------------------


def test_rank_routes_returns_dict():
    result = rank_routes("TestDrug", dose_mg=100.0)
    assert isinstance(result, dict)
    assert "best_route" in result
    assert "worst_route" in result
    assert "results" in result


def test_rank_routes_best_is_iv():
    result = rank_routes("TestDrug", dose_mg=100.0)
    assert result["best_route"] == "iv"


def test_rank_routes_worst_not_iv():
    result = rank_routes("TestDrug", dose_mg=100.0)
    assert result["worst_route"] != "iv"


# ---------------------------------------------------------------------------
# Subset of routes
# ---------------------------------------------------------------------------


def test_subset_routes():
    results = compare_bioavailability("TestDrug", dose_mg=100.0, routes=["iv", "oral"])
    assert len(results) == 2
    routes = {r.route for r in results}
    assert routes == {"iv", "oral"}


def test_single_route_iv():
    results = compare_bioavailability("TestDrug", dose_mg=100.0, routes=["iv"])
    assert len(results) == 1
    assert results[0].route == "iv"
    assert abs(results[0].f_absolute - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Linear dose scaling
# ---------------------------------------------------------------------------


def test_dose_scaling_cmax():
    r1 = compare_bioavailability("Drug", dose_mg=100.0, routes=["oral"])
    r2 = compare_bioavailability("Drug", dose_mg=200.0, routes=["oral"])
    ratio = r2[0].cmax_mg_L / r1[0].cmax_mg_L
    assert abs(ratio - 2.0) < 0.01


def test_dose_scaling_auc():
    r1 = compare_bioavailability("Drug", dose_mg=100.0, routes=["oral"])
    r2 = compare_bioavailability("Drug", dose_mg=200.0, routes=["oral"])
    ratio = r2[0].auc_mg_h_per_L / r1[0].auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    results = compare_bioavailability("TestDrug", dose_mg=100.0)
    for r in results:
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0, f"Empty notes for route {r.route}"


def test_oral_relative_ba_less_than_100():
    results = compare_bioavailability("TestDrug", dose_mg=100.0, fh=0.7, fa=0.9)
    oral = next(r for r in results if r.route == "oral")
    assert oral.relative_bioavailability_pct < 100.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=-50.0)


def test_validation_fh_zero_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=100.0, fh=0.0)


def test_validation_fa_zero_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=100.0, fa=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError):
        compare_bioavailability("Drug", dose_mg=100.0, vd_L=0.0)
