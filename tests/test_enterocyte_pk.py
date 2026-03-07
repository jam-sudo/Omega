"""Tests for core/enterocyte_pk.py — Phase 657."""

import math

import pytest

from omega_pbpk.core.enterocyte_pk import (
    EnterocytePKResult,
    estimate_fg,
    simulate_enterocyte_pk,
)

# --- Basic return type and structure ---


def test_returns_enterocyte_pk_result():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, EnterocytePKResult)


def test_fa_in_range():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert 0.0 < result.fa <= 1.0


def test_fg_in_range():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert 0.0 < result.fg <= 1.0


def test_fa_fg_equals_product():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert math.isclose(result.fa_fg, result.fa * result.fg, rel_tol=1e-6)


def test_cmax_portal_positive():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_portal > 0.0


def test_auc_portal_positive():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert result.auc_portal > 0.0


def test_cyp3a4_metabolized_nonneg():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert result.cyp3a4_metabolized_mg >= 0.0


def test_pgp_effluxed_nonneg():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert result.pgp_effluxed_mg >= 0.0


def test_list_lengths_consistent():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    n = len(result.times_h)
    assert len(result.a_lumen_mg) == n
    assert len(result.c_portal_mg_L) == n
    assert len(result.c_enterocyte_mg_L) == n


def test_notes_nonempty():
    result = simulate_enterocyte_pk("TestDrug", dose_mg=100.0)
    assert len(result.notes) > 0


# --- Pharmacological sensitivity ---


def test_higher_cyp_lower_fg():
    r_low = simulate_enterocyte_pk("Drug", dose_mg=100.0, k_cyp3a4_per_h=0.1)
    r_high = simulate_enterocyte_pk("Drug", dose_mg=100.0, k_cyp3a4_per_h=2.0)
    assert r_high.fg < r_low.fg


def test_higher_pgp_lower_fg():
    r_low = simulate_enterocyte_pk("Drug", dose_mg=100.0, k_pgp_per_h=0.05)
    r_high = simulate_enterocyte_pk("Drug", dose_mg=100.0, k_pgp_per_h=1.0)
    assert r_high.fg < r_low.fg


def test_zero_cyp_zero_pgp_fg_approx_one():
    result = simulate_enterocyte_pk(
        "Drug", dose_mg=100.0, k_cyp3a4_per_h=0.0, k_pgp_per_h=0.0, k_transport_per_h=2.0
    )
    assert math.isclose(result.fg, 1.0, rel_tol=1e-9)


def test_higher_ka_earlier_tmax():
    r_slow = simulate_enterocyte_pk("Drug", dose_mg=100.0, ka_per_h=0.5)
    r_fast = simulate_enterocyte_pk("Drug", dose_mg=100.0, ka_per_h=3.0)
    assert r_fast.tmax_portal_h <= r_slow.tmax_portal_h


def test_double_dose_double_auc():
    r1 = simulate_enterocyte_pk("Drug", dose_mg=100.0)
    r2 = simulate_enterocyte_pk("Drug", dose_mg=200.0)
    assert math.isclose(r2.auc_portal / r1.auc_portal, 2.0, rel_tol=0.02)


def test_double_dose_double_cmax():
    r1 = simulate_enterocyte_pk("Drug", dose_mg=100.0)
    r2 = simulate_enterocyte_pk("Drug", dose_mg=200.0)
    assert math.isclose(r2.cmax_portal / r1.cmax_portal, 2.0, rel_tol=0.02)


# --- estimate_fg function ---


def test_estimate_fg_returns_float():
    fg = estimate_fg(0.5, 0.2, 2.0)
    assert isinstance(fg, float)


def test_estimate_fg_in_range():
    fg = estimate_fg(0.5, 0.2, 2.0)
    assert 0.0 < fg <= 1.0


def test_estimate_fg_zero_cyp_pgp_is_one():
    fg = estimate_fg(0.0, 0.0, 2.0)
    assert math.isclose(fg, 1.0, rel_tol=1e-9)


def test_estimate_fg_high_cyp_low_fg():
    fg_low = estimate_fg(0.1, 0.0, 2.0)
    fg_high = estimate_fg(5.0, 0.0, 2.0)
    assert fg_high < fg_low


# --- Validation ---


def test_validation_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_enterocyte_pk("Drug", dose_mg=0.0)


def test_validation_zero_ka_raises():
    with pytest.raises(ValueError):
        simulate_enterocyte_pk("Drug", dose_mg=100.0, ka_per_h=0.0)


def test_validation_zero_v_enterocyte_raises():
    with pytest.raises(ValueError):
        simulate_enterocyte_pk("Drug", dose_mg=100.0, v_enterocyte_L=0.0)
