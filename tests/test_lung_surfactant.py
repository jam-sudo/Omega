"""Tests for lung surfactant interaction model — Phase 231."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.lung_surfactant import (
    SurfactantPKResult,
    compare_logP_effect,
    free_fraction_in_alf,
    simulate_inhaled_surfactant_pk,
    surfactant_partition,
)


# ── surfactant_partition ──────────────────────────────────────────────


def test_surfactant_partition_higher_logP_gives_higher_kp():
    """Higher logP → higher Kp_surf (lipophilic drugs bind more)."""
    kp_low = surfactant_partition(logP=0.0, mw=300.0)
    kp_mid = surfactant_partition(logP=2.0, mw=300.0)
    kp_high = surfactant_partition(logP=5.0, mw=300.0)
    assert kp_low < kp_mid < kp_high


def test_surfactant_partition_higher_mw_gives_lower_kp():
    """Higher MW → lower Kp_surf (MW penalizes binding)."""
    kp_small = surfactant_partition(logP=2.0, mw=100.0)
    kp_large = surfactant_partition(logP=2.0, mw=800.0)
    assert kp_small > kp_large


def test_surfactant_partition_positive():
    """Kp_surf must always be positive."""
    assert surfactant_partition(logP=-3.0, mw=1000.0) > 0.0
    assert surfactant_partition(logP=0.0, mw=300.0) > 0.0
    assert surfactant_partition(logP=5.0, mw=200.0) > 0.0


def test_surfactant_partition_formula():
    """Verify formula: Kp = 10^(0.8*logP - 0.003*mw + 0.5)."""
    logP, mw = 2.0, 300.0
    expected = 10 ** (0.8 * logP - 0.003 * mw + 0.5)
    assert abs(surfactant_partition(logP, mw) - expected) < 1e-10


# ── free_fraction_in_alf ─────────────────────────────────────────────


def test_free_fraction_kp_zero_is_one():
    """Kp_surf = 0 → no surfactant binding → fu = 1.0."""
    fu = free_fraction_in_alf(kp_surf=0.0)
    assert abs(fu - 1.0) < 1e-12


def test_free_fraction_high_kp_approaches_zero():
    """Very high Kp_surf → nearly all drug bound → fu ≈ 0."""
    fu = free_fraction_in_alf(kp_surf=1e6, v_surf_mL=1.0, v_aq_mL=1.0)
    assert fu < 1e-4


def test_free_fraction_in_range():
    """fu must be in (0, 1] for realistic Kp values."""
    fu = free_fraction_in_alf(kp_surf=10.0, v_surf_mL=0.5, v_aq_mL=10.0)
    assert 0.0 < fu <= 1.0


def test_free_fraction_larger_v_surf_lowers_fu():
    """Larger surfactant volume → more binding → lower fu."""
    fu_small = free_fraction_in_alf(kp_surf=5.0, v_surf_mL=0.1, v_aq_mL=10.0)
    fu_large = free_fraction_in_alf(kp_surf=5.0, v_surf_mL=2.0, v_aq_mL=10.0)
    assert fu_small > fu_large


def test_free_fraction_invalid_v_aq():
    """v_aq_mL <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="v_aq_mL"):
        free_fraction_in_alf(kp_surf=1.0, v_aq_mL=0.0)


def test_free_fraction_negative_kp_raises():
    """Negative kp_surf should raise ValueError."""
    with pytest.raises(ValueError, match="kp_surf"):
        free_fraction_in_alf(kp_surf=-1.0)


# ── simulate_inhaled_surfactant_pk ────────────────────────────────────


def test_simulate_fu_surf_in_range():
    """fu_surf must be in (0, 1]."""
    result = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=2.0, mw=300.0)
    assert 0.0 < result.fu_surf <= 1.0


def test_simulate_f_absorbed_in_range():
    """f_absorbed must be in [0, 1]."""
    result = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=2.0, mw=300.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_simulate_effective_ka_leq_ka_aq():
    """effective_ka_per_h <= ka_aq_per_h (fu <= 1)."""
    ka_aq = 3.0
    result = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=2.0, mw=300.0, ka_aq_per_h=ka_aq)
    assert result.effective_ka_per_h <= ka_aq + 1e-10


def test_simulate_high_logP_lower_f_absorbed():
    """High logP → strong surfactant trapping → lower f_absorbed."""
    res_low = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=0.0, mw=300.0)
    res_high = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=5.0, mw=300.0)
    assert res_low.f_absorbed > res_high.f_absorbed


def test_simulate_low_logP_higher_f_absorbed():
    """Low logP → less surfactant binding → higher f_absorbed."""
    res = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=-2.0, mw=300.0)
    assert res.f_absorbed > 0.5


def test_simulate_invalid_dose():
    """dose_mg <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_inhaled_surfactant_pk("Drug", 0.0, logP=2.0, mw=300.0)


def test_simulate_invalid_cl():
    """cl_L_per_h <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_inhaled_surfactant_pk("Drug", 1.0, logP=2.0, mw=300.0, cl_L_per_h=0.0)


def test_simulate_invalid_vd():
    """vd_L <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="vd_L"):
        simulate_inhaled_surfactant_pk("Drug", 1.0, logP=2.0, mw=300.0, vd_L=0.0)


def test_simulate_result_field_types():
    """Result fields have correct types."""
    result = simulate_inhaled_surfactant_pk("Drug", 1.0, logP=1.0, mw=250.0)
    assert isinstance(result, SurfactantPKResult)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.kp_surf, float)
    assert isinstance(result.fu_surf, float)
    assert isinstance(result.cmax_mg_L, float)
    assert isinstance(result.tmax_h, float)
    assert isinstance(result.auc_mg_h_per_L, float)
    assert isinstance(result.f_absorbed, float)
    assert isinstance(result.effective_ka_per_h, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.m_aq_mg, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.notes, list)


# ── compare_logP_effect ───────────────────────────────────────────────


def test_compare_logP_sorted_by_f_absorbed_descending():
    """compare_logP_effect returns results sorted by f_absorbed descending."""
    logP_values = [0.0, 2.0, 4.0, 6.0]
    results = compare_logP_effect("Drug", dose_mg=1.0, mw=300.0, logP_values=logP_values)
    f_abs_values = [r.f_absorbed for r in results]
    assert f_abs_values == sorted(f_abs_values, reverse=True)


def test_compare_logP_correct_count():
    """compare_logP_effect returns one result per logP value."""
    logP_values = [0.0, 2.0, 4.0]
    results = compare_logP_effect("Drug", dose_mg=1.0, mw=300.0, logP_values=logP_values)
    assert len(results) == len(logP_values)


def test_compare_logP_all_f_absorbed_in_range():
    """All results from compare_logP_effect have f_absorbed in [0, 1]."""
    logP_values = [-1.0, 1.0, 3.0, 5.0]
    results = compare_logP_effect("Drug", dose_mg=1.0, mw=300.0, logP_values=logP_values)
    for r in results:
        assert 0.0 <= r.f_absorbed <= 1.0
