"""Tests for Phase 264 -- Bone Marrow PK Model (bone_marrow_pk_p264)."""

import pytest

from omega_pbpk.core.bone_marrow_pk_p264 import (
    BoneMarrowPKResult,
    compare_routes,
    simulate_bone_marrow_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_iv(dose=100.0, cl=5.0, t_end=48.0, dt=0.05):
    return simulate_bone_marrow_pk(
        "DrugA",
        dose_mg=dose,
        route="iv_bolus",
        cl_L_per_h=cl,
        t_end_h=t_end,
        dt_h=dt,
    )


def _sim_oral(dose=100.0, cl=5.0, t_end=48.0, dt=0.05):
    return simulate_bone_marrow_pk(
        "DrugA",
        dose_mg=dose,
        route="oral",
        cl_L_per_h=cl,
        t_end_h=t_end,
        dt_h=dt,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_iv_returns_correct_type():
    assert isinstance(_sim_iv(), BoneMarrowPKResult)


def test_oral_returns_correct_type():
    assert isinstance(_sim_oral(), BoneMarrowPKResult)


# ---------------------------------------------------------------------------
# IV: plasma starts high
# ---------------------------------------------------------------------------


def test_iv_plasma_starts_high():
    r = _sim_iv(dose=100.0, cl=5.0)
    # IV bolus: first element = dose / V_plasma = 100 / 3 ~ 33.3 mg/L
    assert r.c_plasma_mg_L[0] > 10.0


# ---------------------------------------------------------------------------
# Oral: plasma starts at zero
# ---------------------------------------------------------------------------


def test_oral_plasma_starts_at_zero():
    r = _sim_oral()
    assert r.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Marrow starts at zero for both routes
# ---------------------------------------------------------------------------


def test_iv_marrow_starts_at_zero():
    r = _sim_iv()
    assert r.c_marrow_mg_L[0] == 0.0


def test_oral_marrow_starts_at_zero():
    r = _sim_oral()
    assert r.c_marrow_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# cmax_marrow > 0
# ---------------------------------------------------------------------------


def test_iv_cmax_marrow_positive():
    r = _sim_iv()
    assert r.cmax_marrow_mg_L > 0.0


def test_oral_cmax_marrow_positive():
    r = _sim_oral()
    assert r.cmax_marrow_mg_L > 0.0


# ---------------------------------------------------------------------------
# AUC positive
# ---------------------------------------------------------------------------


def test_iv_auc_plasma_positive():
    r = _sim_iv()
    assert r.auc_plasma_mg_h_per_L > 0.0


def test_iv_auc_marrow_positive():
    r = _sim_iv()
    assert r.auc_marrow_mg_h_per_L > 0.0


def test_oral_auc_plasma_positive():
    r = _sim_oral()
    assert r.auc_plasma_mg_h_per_L > 0.0


def test_oral_auc_marrow_positive():
    r = _sim_oral()
    assert r.auc_marrow_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# marrow_plasma_ratio > 0
# ---------------------------------------------------------------------------


def test_iv_marrow_plasma_ratio_positive():
    r = _sim_iv()
    assert r.marrow_plasma_ratio > 0.0


def test_oral_marrow_plasma_ratio_positive():
    r = _sim_oral()
    assert r.marrow_plasma_ratio > 0.0


def test_marrow_plasma_ratio_equals_auc_ratio():
    r = _sim_iv()
    expected = r.auc_marrow_mg_h_per_L / r.auc_plasma_mg_h_per_L
    assert abs(r.marrow_plasma_ratio - expected) < 1e-9


# ---------------------------------------------------------------------------
# myelosuppression_risk valid values
# ---------------------------------------------------------------------------


def test_myelosuppression_risk_valid_values():
    valid = {"low", "moderate", "high"}
    for route in ("iv_bolus", "oral"):
        r = simulate_bone_marrow_pk("DrugA", 100.0, route, 5.0)
        assert r.myelosuppression_risk in valid


# ---------------------------------------------------------------------------
# compare_routes
# ---------------------------------------------------------------------------


def test_compare_routes_returns_two():
    results = compare_routes("DrugA", 100.0, 5.0)
    assert len(results) == 2


def test_compare_routes_sorted_descending():
    results = compare_routes("DrugA", 100.0, 5.0)
    assert results[0].marrow_plasma_ratio >= results[1].marrow_plasma_ratio


def test_compare_routes_contains_both_routes():
    results = compare_routes("DrugA", 100.0, 5.0)
    routes = {r.route for r in results}
    assert routes == {"iv_bolus", "oral"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_bone_marrow_pk("DrugA", dose_mg=0.0, route="iv_bolus", cl_L_per_h=5.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        simulate_bone_marrow_pk("DrugA", dose_mg=100.0, route="iv_bolus", cl_L_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route must be"):
        simulate_bone_marrow_pk("DrugA", dose_mg=100.0, route="subcutaneous", cl_L_per_h=5.0)
