"""Tests for Phase 1010: tumor_vasculature_pk.py"""

import pytest

from omega_pbpk.core.tumor_vasculature_pk import (
    TumorVasculaturePKResult,
    simulate_tumor_vasculature_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_tumor_vasculature_pk("TestDrug", dose_mg=10.0)
    assert isinstance(result, TumorVasculaturePKResult)


def test_cmax_plasma_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.cmax_plasma_mg_L > 0


def test_cmax_tumor_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.cmax_tumor_mg_L > 0


def test_tmax_tumor_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.tmax_tumor_h > 0


def test_auc_plasma_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_auc_tumor_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.auc_tumor_mg_h_per_L > 0


def test_tumor_to_plasma_ratio_positive():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    assert result.tumor_to_plasma_ratio > 0


# ---------------------------------------------------------------------------
# EPR effect
# ---------------------------------------------------------------------------


def test_large_molecule_has_higher_epr_factor():
    r_large = simulate_tumor_vasculature_pk("mAb", dose_mg=10.0, mw_kDa=100.0)
    r_small = simulate_tumor_vasculature_pk("Small", dose_mg=10.0, mw_kDa=0.3)
    assert r_large.epr_effect_factor > r_small.epr_effect_factor


def test_epr_factor_ge_one():
    for mw in [0.1, 0.3, 1.0, 10.0, 100.0]:
        result = simulate_tumor_vasculature_pk("Drug", dose_mg=10.0, mw_kDa=mw)
        assert result.epr_effect_factor >= 1.0, f"EPR factor < 1 for mw={mw}"


def test_ifp_barrier_factor_in_range():
    for mw in [0.1, 0.3, 1.0, 10.0, 150.0]:
        result = simulate_tumor_vasculature_pk("Drug", dose_mg=10.0, mw_kDa=mw)
        assert 0.0 < result.ifp_barrier_factor <= 1.0, (
            f"IFP factor out of range for mw={mw}: {result.ifp_barrier_factor}"
        )


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_tumor_interstitium_starts_near_zero():
    result = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0, dt_h=0.1)
    assert result.c_tumor_interstitium_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax_plasma():
    r1 = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    r2 = simulate_tumor_vasculature_pk("DrugA", dose_mg=20.0)
    assert r2.cmax_plasma_mg_L == pytest.approx(2.0 * r1.cmax_plasma_mg_L, rel=1e-9)


def test_double_dose_doubles_cmax_tumor():
    r1 = simulate_tumor_vasculature_pk("DrugA", dose_mg=10.0)
    r2 = simulate_tumor_vasculature_pk("DrugA", dose_mg=20.0)
    assert r2.cmax_tumor_mg_L == pytest.approx(2.0 * r1.cmax_tumor_mg_L, rel=1e-6)


# ---------------------------------------------------------------------------
# Small molecule accumulates in tumor faster (relative to plasma half-life)
# ---------------------------------------------------------------------------


def test_small_molecule_tumor_accumulates():
    """Small molecule should show measurable tumor concentration early."""
    result = simulate_tumor_vasculature_pk(
        "SmallDrug",
        dose_mg=100.0,
        mw_kDa=0.3,
        logp=2.0,
        cl_plasma_L_per_h=1.0,
        t_end_h=12.0,
        dt_h=0.5,
    )
    # After a few hours, tumor interstitium should have some drug
    mid_idx = len(result.c_tumor_interstitium_mg_L) // 2
    assert any(c > 0 for c in result.c_tumor_interstitium_mg_L[1 : mid_idx + 1])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_vasculature_pk("X", dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_vasculature_pk("X", dose_mg=-5.0)


def test_invalid_v_plasma_raises():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_tumor_vasculature_pk("X", dose_mg=10.0, v_plasma_L=0.0)


def test_invalid_v_tumor_interstitium_raises():
    with pytest.raises(ValueError, match="v_tumor_interstitium_L"):
        simulate_tumor_vasculature_pk("X", dose_mg=10.0, v_tumor_interstitium_L=-1.0)


def test_invalid_cl_plasma_raises():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_tumor_vasculature_pk("X", dose_mg=10.0, cl_plasma_L_per_h=0.0)
