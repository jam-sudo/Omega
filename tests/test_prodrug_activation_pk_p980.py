"""Tests for prodrug activation PK model (Phase 980)."""

import math
import pytest

from omega_pbpk.prediction.prodrug_activation_pk import (
    ProdrugActivationResult,
    compare_prodrug_routes,
    simulate_prodrug_activation,
)


# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------

def test_returns_result_type():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert isinstance(result, ProdrugActivationResult)


def test_prodrug_name_stored():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.prodrug_name == "Enalapril"


def test_active_name_stored():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.active_drug_name == "Enalaprilat"


def test_route_stored():
    result = simulate_prodrug_activation("X", "Y", 10.0, route="iv")
    assert result.route == "iv"


def test_activation_site_stored():
    result = simulate_prodrug_activation("X", "Y", 10.0, activation_site="plasma")
    assert result.activation_site == "plasma"


# ---------------------------------------------------------------------------
# Positive concentration checks
# ---------------------------------------------------------------------------

def test_cmax_active_positive():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.cmax_active_mg_L > 0


def test_cmax_prodrug_positive():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.cmax_prodrug_mg_L > 0


def test_tmax_active_positive():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.tmax_active_h > 0


def test_auc_active_positive():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.auc_active_mg_h_per_L > 0


def test_auc_prodrug_positive():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert result.auc_prodrug_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Conversion efficiency
# ---------------------------------------------------------------------------

def test_conversion_efficiency_in_range():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0)
    assert 0 < result.conversion_efficiency <= 1


# ---------------------------------------------------------------------------
# Oral: active starts at zero
# ---------------------------------------------------------------------------

def test_oral_active_starts_at_zero():
    result = simulate_prodrug_activation("Enalapril", "Enalaprilat", 10.0, route="oral")
    assert math.isclose(result.c_active_mg_L[0], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Higher f_conversion -> higher AUC active
# ---------------------------------------------------------------------------

def test_higher_f_conversion_higher_auc_active():
    r_low = simulate_prodrug_activation("X", "Y", 10.0, f_conversion=0.3)
    r_high = simulate_prodrug_activation("X", "Y", 10.0, f_conversion=0.9)
    assert r_high.auc_active_mg_h_per_L > r_low.auc_active_mg_h_per_L


# ---------------------------------------------------------------------------
# compare_prodrug_routes
# ---------------------------------------------------------------------------

def test_compare_routes_returns_two():
    results = compare_prodrug_routes("Enalapril", "Enalaprilat", 10.0)
    assert len(results) == 2


def test_compare_routes_sorted_by_auc_active():
    results = compare_prodrug_routes("Enalapril", "Enalaprilat", 10.0)
    assert results[0].auc_active_mg_h_per_L >= results[1].auc_active_mg_h_per_L


def test_compare_routes_contains_both_routes():
    results = compare_prodrug_routes("Enalapril", "Enalaprilat", 10.0)
    routes = {r.route for r in results}
    assert routes == {"oral", "iv"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_prodrug_activation("X", "Y", 0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_prodrug_activation("X", "Y", -5.0)


def test_invalid_k_conv_raises():
    with pytest.raises(ValueError, match="k_conv_per_h"):
        simulate_prodrug_activation("X", "Y", 10.0, k_conv_per_h=0.0)


def test_invalid_f_conversion_zero_raises():
    with pytest.raises(ValueError, match="f_conversion"):
        simulate_prodrug_activation("X", "Y", 10.0, f_conversion=0.0)


def test_invalid_f_conversion_above_one_raises():
    with pytest.raises(ValueError, match="f_conversion"):
        simulate_prodrug_activation("X", "Y", 10.0, f_conversion=1.5)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_prodrug_activation("X", "Y", 10.0, route="subcutaneous")


# ---------------------------------------------------------------------------
# IV vs oral
# ---------------------------------------------------------------------------

def test_iv_prodrug_nonzero_at_t0():
    result = simulate_prodrug_activation("X", "Y", 10.0, route="iv")
    assert result.c_prodrug_mg_L[0] > 0
