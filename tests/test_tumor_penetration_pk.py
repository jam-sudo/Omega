"""Tests for Phase 803 — Tumor Penetration PK Model."""

import pytest

from omega_pbpk.core.tumor_penetration_pk import (
    TumorPenetrationResult,
    simulate_tumor_penetration,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert isinstance(result, TumorPenetrationResult)


def test_list_fields_are_lists():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_tumor_vascular_mg_L, list)
    assert isinstance(result.c_tumor_interstitium_mg_L, list)


def test_list_fields_same_length():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_tumor_vascular_mg_L) == n
    assert len(result.c_tumor_interstitium_mg_L) == n


def test_scalar_fields_are_floats():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert isinstance(result.cmax_plasma, float)
    assert isinstance(result.cmax_tumor_vasculature, float)
    assert isinstance(result.cmax_tumor_interstitium, float)
    assert isinstance(result.auc_plasma, float)
    assert isinstance(result.auc_tumor, float)
    assert isinstance(result.tumor_plasma_kp, float)
    assert isinstance(result.interstitium_plasma_ratio, float)
    assert isinstance(result.tumor_penetration_score, float)
    assert isinstance(result.transcapillary_perm_limited, bool)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# cmax and auc sanity
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert result.cmax_plasma > 0.0


def test_cmax_tumor_vasculature_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert result.cmax_tumor_vasculature > 0.0


def test_cmax_tumor_interstitium_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert result.cmax_tumor_interstitium > 0.0


def test_auc_plasma_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert result.auc_plasma > 0.0


def test_auc_tumor_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert result.auc_tumor > 0.0


# ---------------------------------------------------------------------------
# tumor_plasma_kp range
# ---------------------------------------------------------------------------


def test_tumor_plasma_kp_in_reasonable_range():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0, ps_tumor_L_per_h=0.5, t_end_h=48.0)
    assert 0.0 < result.tumor_plasma_kp <= 2.0


def test_tumor_plasma_kp_positive():
    result = simulate_tumor_penetration("Drug_A", dose_mg=50.0)
    assert result.tumor_plasma_kp > 0.0


# ---------------------------------------------------------------------------
# Higher PS → higher tumor penetration
# ---------------------------------------------------------------------------


def test_higher_ps_increases_tumor_penetration():
    low_ps = simulate_tumor_penetration("Drug_A", dose_mg=100.0, ps_tumor_L_per_h=0.1, t_end_h=24.0)
    high_ps = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, ps_tumor_L_per_h=2.0, t_end_h=24.0
    )
    assert high_ps.cmax_tumor_interstitium > low_ps.cmax_tumor_interstitium


def test_higher_ps_increases_auc_tumor():
    low_ps = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, ps_tumor_L_per_h=0.05, t_end_h=24.0
    )
    high_ps = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, ps_tumor_L_per_h=2.0, t_end_h=24.0
    )
    assert high_ps.auc_tumor > low_ps.auc_tumor


# ---------------------------------------------------------------------------
# Higher MW → lower penetration
# ---------------------------------------------------------------------------


def test_higher_mw_reduces_interstitium_penetration():
    small = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, mw_kDa=0.3, ps_tumor_L_per_h=0.5, t_end_h=24.0
    )
    large = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, mw_kDa=150.0, ps_tumor_L_per_h=0.5, t_end_h=24.0
    )
    assert small.cmax_tumor_interstitium > large.cmax_tumor_interstitium


def test_higher_mw_reduces_penetration_score():
    small = simulate_tumor_penetration("Drug_A", dose_mg=100.0, mw_kDa=0.5, t_end_h=24.0)
    large = simulate_tumor_penetration("Drug_A", dose_mg=100.0, mw_kDa=200.0, t_end_h=24.0)
    assert small.tumor_penetration_score >= large.tumor_penetration_score


# ---------------------------------------------------------------------------
# IFP factor: lower ifp_factor → less interstitium penetration
# ---------------------------------------------------------------------------


def test_high_ifp_reduces_interstitium_penetration():
    normal = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, ifp_factor=1.0, ps_tumor_L_per_h=0.5, t_end_h=24.0
    )
    high_ifp = simulate_tumor_penetration(
        "Drug_A", dose_mg=100.0, ifp_factor=0.2, ps_tumor_L_per_h=0.5, t_end_h=24.0
    )
    assert normal.cmax_tumor_interstitium > high_ifp.cmax_tumor_interstitium


def test_high_ifp_reduces_auc_interstitium():
    normal = simulate_tumor_penetration("Drug_A", dose_mg=100.0, ifp_factor=1.0, t_end_h=24.0)
    high_ifp = simulate_tumor_penetration("Drug_A", dose_mg=100.0, ifp_factor=0.1, t_end_h=24.0)
    assert normal.auc_tumor >= high_ifp.auc_tumor


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_route_simulation():
    result = simulate_tumor_penetration(
        "Drug_B", dose_mg=200.0, route="oral", f_oral=0.7, ka_per_h=1.2
    )
    assert result.cmax_plasma > 0.0
    assert result.auc_plasma > 0.0


def test_oral_lower_cmax_than_iv():
    iv = simulate_tumor_penetration("Drug_B", dose_mg=100.0, route="iv", t_end_h=24.0)
    oral = simulate_tumor_penetration(
        "Drug_B", dose_mg=100.0, route="oral", f_oral=1.0, ka_per_h=0.5, t_end_h=24.0
    )
    # IV gives immediate spike, oral has slower absorption → typically lower Cmax
    assert iv.cmax_plasma >= oral.cmax_plasma


# ---------------------------------------------------------------------------
# Tumor penetration score range
# ---------------------------------------------------------------------------


def test_penetration_score_range():
    result = simulate_tumor_penetration("Drug_A", dose_mg=100.0)
    assert 0.0 <= result.tumor_penetration_score <= 100.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_tumor_penetration("Drug_A", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_tumor_penetration("Drug_A", dose_mg=-50.0)


def test_negative_ps_raises():
    with pytest.raises(ValueError):
        simulate_tumor_penetration("Drug_A", dose_mg=100.0, ps_tumor_L_per_h=-1.0)
