"""Tests for Phase 765: Transdermal skin-layer permeation PK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.transdermal_layers_pk import (
    TransdermalLayersPKResult,
    compare_patch_areas,
    simulate_transdermal_layers_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg_per_cm2=0.5,
    patch_area_cm2=10.0,
    logp=2.0,
    mw_da=300.0,
    k_removal_per_h=5.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def _default_result() -> TransdermalLayersPKResult:
    return simulate_transdermal_layers_pk(**DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Basic return-type and structure tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _default_result()
    assert isinstance(result, TransdermalLayersPKResult)


def test_times_list_nonempty():
    result = _default_result()
    assert len(result.times_h) > 0


def test_c_plasma_list_matches_times():
    result = _default_result()
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_a_sc_list_matches_times():
    result = _default_result()
    assert len(result.a_sc_mg_cm2) == len(result.times_h)


def test_a_ve_list_matches_times():
    result = _default_result()
    assert len(result.a_ve_mg_cm2) == len(result.times_h)


# ---------------------------------------------------------------------------
# Scalar metric sanity
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = _default_result()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = _default_result()
    assert result.auc_mg_h_per_L > 0.0


def test_total_dose_correct():
    result = _default_result()
    expected = DEFAULT_KWARGS["dose_mg_per_cm2"] * DEFAULT_KWARGS["patch_area_cm2"]
    assert abs(result.total_dose_mg - expected) < 1e-9


def test_fraction_absorbed_between_0_and_1():
    result = _default_result()
    assert 0.0 <= result.fraction_absorbed <= 1.0


def test_lag_time_nonnegative():
    result = _default_result()
    assert result.lag_time_h >= 0.0


def test_p_sc_positive():
    result = _default_result()
    assert result.p_sc_cm_per_h > 0.0


def test_notes_nonempty():
    result = _default_result()
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Physical / pharmacological behaviour
# ---------------------------------------------------------------------------


def test_higher_logp_gives_higher_cmax():
    """Higher logP → larger SC diffusion coefficient → better penetration."""
    r_low = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "logp": 1.0})
    r_high = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "logp": 4.0})
    assert r_high.cmax_mg_L > r_low.cmax_mg_L


def test_higher_mw_gives_lower_cmax():
    """Higher MW → smaller diffusion → worse permeation."""
    r_low = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "mw_da": 200.0})
    r_high = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "mw_da": 600.0})
    assert r_low.cmax_mg_L > r_high.cmax_mg_L


def test_larger_patch_area_gives_higher_cmax():
    """Larger patch delivers more drug into systemic circulation."""
    r_small = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "patch_area_cm2": 5.0})
    r_large = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "patch_area_cm2": 20.0})
    assert r_large.cmax_mg_L > r_small.cmax_mg_L


def test_approximate_linear_scaling_with_patch_area():
    """2× patch area → approximately 2× Cmax (linear dose-scaling property)."""
    r1 = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "patch_area_cm2": 10.0})
    r2 = simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "patch_area_cm2": 20.0})
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    # Allow 10% tolerance around 2×
    assert 1.8 <= ratio <= 2.2


# ---------------------------------------------------------------------------
# compare_patch_areas
# ---------------------------------------------------------------------------


def test_compare_patch_areas_returns_list_of_correct_length():
    areas = [5.0, 10.0, 20.0]
    results = compare_patch_areas("TestDrug", 0.5, areas)
    assert len(results) == len(areas)


def test_compare_patch_areas_all_correct_type():
    results = compare_patch_areas("TestDrug", 0.5, [5.0, 15.0])
    for r in results:
        assert isinstance(r, TransdermalLayersPKResult)


def test_compare_patch_areas_preserves_order():
    """Results should be in the same order as the input list."""
    areas = [20.0, 5.0, 10.0]
    results = compare_patch_areas("TestDrug", 0.5, areas)
    for r, area in zip(results, areas):
        assert r.patch_area_cm2 == area


# ---------------------------------------------------------------------------
# Validation (errors)
# ---------------------------------------------------------------------------


def test_validation_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_per_cm2"):
        simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "dose_mg_per_cm2": 0.0})


def test_validation_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_per_cm2"):
        simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "dose_mg_per_cm2": -1.0})


def test_validation_zero_patch_area_raises():
    with pytest.raises(ValueError, match="patch_area_cm2"):
        simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "patch_area_cm2": 0.0})


def test_validation_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "cl_L_per_h": 0.0})


def test_validation_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_transdermal_layers_pk(**{**DEFAULT_KWARGS, "vd_L": 0.0})


# ---------------------------------------------------------------------------
# Initial SC mass conservation
# ---------------------------------------------------------------------------


def test_a_sc_starts_at_dose():
    result = _default_result()
    assert abs(result.a_sc_mg_cm2[0] - DEFAULT_KWARGS["dose_mg_per_cm2"]) < 1e-9


def test_a_ve_starts_at_zero():
    result = _default_result()
    assert result.a_ve_mg_cm2[0] == 0.0


def test_c_plasma_starts_at_zero():
    result = _default_result()
    assert result.c_plasma_mg_L[0] == 0.0
