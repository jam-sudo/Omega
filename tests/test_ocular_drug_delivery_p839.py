"""Tests for Phase 839 — Ocular Drug Delivery Extended model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.ocular_drug_delivery_p839 import (
    OcularDeliveryResult,
    compare_ocular_routes,
    simulate_ocular_delivery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(route: str = "subconjunctival") -> OcularDeliveryResult:
    return simulate_ocular_delivery(
        drug_name="TestDrug",
        dose_mg=1.0,
        route=route,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_subconjunctival():
    result = _default_result("subconjunctival")
    assert isinstance(result, OcularDeliveryResult)


def test_return_type_suprachoroidal():
    result = _default_result("suprachoroidal")
    assert isinstance(result, OcularDeliveryResult)


def test_return_type_intravitreal():
    result = _default_result("intravitreal")
    assert isinstance(result, OcularDeliveryResult)


# ---------------------------------------------------------------------------
# List length consistency
# ---------------------------------------------------------------------------


def test_lists_same_length():
    result = _default_result()
    n = len(result.times_h)
    assert n > 0
    assert len(result.c_injection_site_mg_L) == n
    assert len(result.c_choroid_mg_L) == n
    assert len(result.c_vitreous_mg_L) == n
    assert len(result.c_retina_mg_L) == n
    assert len(result.c_systemic_mg_L) == n


def test_time_starts_at_zero():
    result = _default_result()
    assert result.times_h[0] == 0.0


def test_time_ends_near_t_end():
    result = simulate_ocular_delivery("TestDrug", 1.0, "subconjunctival", t_end_h=48.0, dt_h=1.0)
    assert result.times_h[-1] == pytest.approx(48.0, abs=1.0)


# ---------------------------------------------------------------------------
# Intravitreal: initial dose in vitreous
# ---------------------------------------------------------------------------


def test_intravitreal_cmax_vitreous_positive():
    result = _default_result("intravitreal")
    assert result.cmax_vitreous_mg_L > 0.0


def test_intravitreal_initial_vitreous_positive():
    result = _default_result("intravitreal")
    # At t=0, vitreous should already have drug
    assert result.c_vitreous_mg_L[0] > 0.0


def test_intravitreal_initial_choroid_zero():
    result = _default_result("intravitreal")
    assert result.c_choroid_mg_L[0] == 0.0


def test_intravitreal_injection_site_zero():
    """For IVT, depot_mg=0 so injection site conc = 0 at t=0."""
    result = _default_result("intravitreal")
    assert result.c_injection_site_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Route comparisons
# ---------------------------------------------------------------------------


def test_suprachoroidal_greater_vitreous_than_subconjunctival():
    """Suprachoroidal has much higher ka_to_choroid → more drug reaches vitreous."""
    sc = _default_result("subconjunctival")
    supra = _default_result("suprachoroidal")
    assert supra.auc_vitreous_mg_h_per_L > sc.auc_vitreous_mg_h_per_L


def test_suprachoroidal_lower_systemic_than_subconjunctival():
    """Suprachoroidal has lower ka_to_systemic (0.1) vs subconjunctival (0.5)."""
    sc = _default_result("subconjunctival")
    supra = _default_result("suprachoroidal")
    assert supra.cmax_systemic_mg_L < sc.cmax_systemic_mg_L


def test_intravitreal_highest_vitreous_auc():
    """Direct IVT injection typically gives highest vitreous exposure."""
    sc = _default_result("subconjunctival")
    ivt = _default_result("intravitreal")
    assert ivt.auc_vitreous_mg_h_per_L > sc.auc_vitreous_mg_h_per_L


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def test_cmax_vitreous_positive_all_routes():
    for route in ("subconjunctival", "suprachoroidal", "intravitreal"):
        result = _default_result(route)
        assert result.cmax_vitreous_mg_L > 0.0, f"Failed for route {route}"


def test_cmax_retina_positive():
    result = _default_result("suprachoroidal")
    assert result.cmax_retina_mg_L > 0.0


def test_auc_vitreous_positive():
    result = _default_result("intravitreal")
    assert result.auc_vitreous_mg_h_per_L > 0.0


def test_ocular_bioavailability_in_range():
    for route in ("subconjunctival", "suprachoroidal", "intravitreal"):
        result = _default_result(route)
        assert 0.0 <= result.ocular_bioavailability <= 1.0, (
            f"Bioavailability out of range for {route}: {result.ocular_bioavailability}"
        )


def test_tmax_vitreous_within_simulation():
    result = _default_result("suprachoroidal")
    assert 0.0 <= result.tmax_vitreous_h <= 168.0


def test_cmax_systemic_positive():
    result = _default_result("subconjunctival")
    assert result.cmax_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# compare_ocular_routes
# ---------------------------------------------------------------------------


def test_compare_returns_three_results():
    results = compare_ocular_routes("TestDrug", 1.0)
    assert len(results) == 3


def test_compare_all_are_ocular_delivery_results():
    results = compare_ocular_routes("TestDrug", 1.0)
    for r in results:
        assert isinstance(r, OcularDeliveryResult)


def test_compare_sorted_by_auc_vitreous_descending():
    results = compare_ocular_routes("TestDrug", 1.0)
    aucs = [r.auc_vitreous_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_three_routes_present():
    results = compare_ocular_routes("TestDrug", 1.0)
    routes = {r.route for r in results}
    assert routes == {"subconjunctival", "suprachoroidal", "intravitreal"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ocular_delivery("Drug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ocular_delivery("Drug", dose_mg=-1.0)


def test_invalid_cl_sys_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_ocular_delivery("Drug", dose_mg=1.0, cl_sys_L_per_h=0.0)


def test_invalid_vd_sys_raises():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_ocular_delivery("Drug", dose_mg=1.0, vd_sys_L=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_ocular_delivery("Drug", dose_mg=1.0, route="eye_drop")
