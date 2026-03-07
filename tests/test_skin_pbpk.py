"""Tests for src/omega_pbpk/core/skin_pbpk.py — Phase 685."""

from __future__ import annotations

import pytest

from omega_pbpk.core.skin_pbpk import (
    SkinPKResult,
    compare_transdermal_routes,
    simulate_skin_pk,
)

# ---------------------------------------------------------------------------
# Basic result type and initial conditions
# ---------------------------------------------------------------------------


def test_returns_skin_pk_result():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert isinstance(result, SkinPKResult)


def test_a_sc_zero_at_t0():
    """Nothing has entered SC at t=0 (dose starts in patch)."""
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.a_sc_mg[0] == 0.0


def test_c_plasma_zero_at_t0():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.c_plasma_mg_L[0] == 0.0


def test_cmax_positive():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_positive_after_simulation():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.tmax_h > 0.0


# ---------------------------------------------------------------------------
# Route differences
# ---------------------------------------------------------------------------


def test_gel_faster_than_patch():
    """Gel (higher k_dose_to_sc) should produce earlier tmax than patch."""
    patch = simulate_skin_pk("Drug_A", dose_mg=10.0, route="patch")
    gel = simulate_skin_pk("Drug_A", dose_mg=10.0, route="gel")
    assert gel.tmax_h < patch.tmax_h


def test_route_stored_in_result():
    patch = simulate_skin_pk("Drug_A", dose_mg=10.0, route="patch")
    gel = simulate_skin_pk("Drug_A", dose_mg=10.0, route="gel")
    assert patch.route == "patch"
    assert gel.route == "gel"


# ---------------------------------------------------------------------------
# logP effect on k_sc_ve
# ---------------------------------------------------------------------------


def test_high_logP_higher_k_sc_ve():
    result_high = simulate_skin_pk("Drug_A", dose_mg=10.0, logP=4.0)
    result_low = simulate_skin_pk("Drug_A", dose_mg=10.0, logP=0.0)
    assert result_high.k_sc_ve_per_h > result_low.k_sc_ve_per_h


def test_k_sc_ve_clamped_low():
    """Very hydrophilic molecule should be clamped to minimum 0.01."""
    result = simulate_skin_pk("Drug_A", dose_mg=10.0, logP=-10.0, mw=1000.0)
    assert result.k_sc_ve_per_h >= 0.01


def test_k_sc_ve_clamped_high():
    """Very lipophilic + low MW should be clamped to maximum 5.0."""
    result = simulate_skin_pk("Drug_A", dose_mg=10.0, logP=10.0, mw=50.0)
    assert result.k_sc_ve_per_h <= 5.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = simulate_skin_pk("Drug_A", dose_mg=10.0)
    r2 = simulate_skin_pk("Drug_A", dose_mg=20.0)
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 0.05


def test_double_dose_doubles_auc():
    r1 = simulate_skin_pk("Drug_A", dose_mg=10.0)
    r2 = simulate_skin_pk("Drug_A", dose_mg=20.0)
    assert abs(r2.auc_mg_h_per_L / r1.auc_mg_h_per_L - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------


def test_lag_time_positive():
    """Transdermal delivery should have a lag before plasma concentrations rise."""
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.lag_time_h > 0.0


def test_lag_time_less_than_tmax():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert result.lag_time_h < result.tmax_h


# ---------------------------------------------------------------------------
# compare_transdermal_routes
# ---------------------------------------------------------------------------


def test_compare_transdermal_routes_returns_dict():
    result = compare_transdermal_routes("Drug_A", dose_mg=10.0)
    assert isinstance(result, dict)


def test_compare_has_required_keys():
    result = compare_transdermal_routes("Drug_A", dose_mg=10.0)
    assert "patch_result" in result
    assert "gel_result" in result
    assert "gel_faster" in result


def test_compare_gel_faster_true():
    result = compare_transdermal_routes("Drug_A", dose_mg=10.0)
    assert result["gel_faster"] is True


def test_compare_notes_nonempty():
    result = compare_transdermal_routes("Drug_A", dose_mg=10.0)
    assert isinstance(result["notes"], str)
    assert len(result["notes"]) > 0


def test_notes_nonempty():
    result = simulate_skin_pk("Drug_A", dose_mg=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_skin_pk("Drug_A", dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_skin_pk("Drug_A", dose_mg=-5.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        simulate_skin_pk("Drug_A", dose_mg=10.0, route="cream")
