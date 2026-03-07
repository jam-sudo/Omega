"""Tests for Phase 1095 — Intravitreal PK (Extended)."""

import math
import pytest

from omega_pbpk.core.intravitreal_pk_p1095 import (
    IntravitreалPKResult,
    compare_molecule_sizes,
    simulate_intravitreal_pk,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def default_result() -> IntravitreалPKResult:
    return simulate_intravitreal_pk("Ranibizumab", dose_mg=0.5, mw_kDa=48.0)


def mab_result() -> IntravitreалPKResult:
    return simulate_intravitreal_pk("Bevacizumab", dose_mg=1.25, mw_kDa=149.0)


def sm_result() -> IntravitreалPKResult:
    return simulate_intravitreal_pk("SmallMol", dose_mg=0.05, mw_kDa=0.5)


# ── Return type ───────────────────────────────────────────────────────────────

def test_return_type():
    res = default_result()
    assert isinstance(res, IntravitreалPKResult)


def test_result_has_expected_fields():
    res = default_result()
    assert hasattr(res, "drug_name")
    assert hasattr(res, "t_half_vitreous_h")
    assert hasattr(res, "systemic_exposure_ratio")
    assert hasattr(res, "dosing_interval_recommended_days")


# ── Vitreous amount ───────────────────────────────────────────────────────────

def test_a_vitreous_starts_at_dose():
    """At t=0, vitreous amount should equal dose_mg."""
    res = simulate_intravitreal_pk("Drug", dose_mg=1.0, mw_kDa=150.0)
    assert res.a_vitreous_mg[0] == pytest.approx(1.0, rel=1e-6)


def test_a_vitreous_decreases_monotonically():
    res = default_result()
    for i in range(1, len(res.a_vitreous_mg)):
        assert res.a_vitreous_mg[i] <= res.a_vitreous_mg[i - 1] + 1e-12


def test_cmax_vitreous_equals_dose():
    res = simulate_intravitreal_pk("Drug", dose_mg=2.0, mw_kDa=100.0)
    assert res.cmax_vitreous_mg == pytest.approx(2.0)


# ── Aqueous humor ─────────────────────────────────────────────────────────────

def test_a_aqueous_starts_at_zero():
    res = default_result()
    assert res.a_aqueous_mg[0] == pytest.approx(0.0)


def test_cmax_aqueous_positive():
    res = default_result()
    assert res.cmax_aqueous_mg > 0.0


def test_tmax_aqueous_positive():
    res = default_result()
    assert res.tmax_aqueous_h > 0.0


# ── Plasma ────────────────────────────────────────────────────────────────────

def test_c_plasma_starts_at_zero():
    res = default_result()
    assert res.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_cmax_plasma_positive():
    res = default_result()
    assert res.cmax_plasma_mg_L > 0.0


# ── Systemic exposure ratio ───────────────────────────────────────────────────

def test_systemic_exposure_ratio_between_0_and_1():
    res = default_result()
    assert 0.0 < res.systemic_exposure_ratio < 1.0


def test_systemic_exposure_ratio_small_molecule():
    res = sm_result()
    assert 0.0 < res.systemic_exposure_ratio < 1.0


# ── MW-dependent half-life ────────────────────────────────────────────────────

def test_larger_mw_longer_t_half():
    """150 kDa mAb should have longer vitreous t½ than 0.5 kDa small molecule."""
    mab = simulate_intravitreal_pk("MAb", dose_mg=1.0, mw_kDa=150.0)
    sm = simulate_intravitreal_pk("SM", dose_mg=1.0, mw_kDa=0.5)
    assert mab.t_half_vitreous_h > sm.t_half_vitreous_h


def test_t_half_matches_formula():
    """t½ = ln(2) / (k_vit_aq + k_vit_elim)."""
    mw = 150.0
    k_vit_aq = 0.003 + 0.05 * math.exp(-mw / 50.0)
    k_vit_elim = 0.001
    expected = math.log(2.0) / (k_vit_aq + k_vit_elim)
    res = simulate_intravitreal_pk("Drug", dose_mg=1.0, mw_kDa=mw)
    assert res.t_half_vitreous_h == pytest.approx(expected, rel=1e-6)


def test_t_half_vitreous_positive():
    res = default_result()
    assert res.t_half_vitreous_h > 0.0


# ── Dosing interval ───────────────────────────────────────────────────────────

def test_dosing_interval_positive():
    res = default_result()
    assert res.dosing_interval_recommended_days > 0.0


def test_dosing_interval_based_on_t_half():
    """Dosing interval = 3 * t½ / 24."""
    res = default_result()
    expected = 3.0 * res.t_half_vitreous_h / 24.0
    assert res.dosing_interval_recommended_days == pytest.approx(expected)


# ── AUC ───────────────────────────────────────────────────────────────────────

def test_auc_vitreous_positive():
    res = default_result()
    assert res.auc_vitreous > 0.0


# ── compare_molecule_sizes ────────────────────────────────────────────────────

def test_compare_molecule_sizes_returns_correct_count():
    mw_list = [0.5, 10.0, 150.0]
    results = compare_molecule_sizes("Drug", dose_mg=1.0, mw_list=mw_list)
    assert len(results) == 3


def test_compare_molecule_sizes_sorted_descending():
    mw_list = [0.5, 10.0, 50.0, 150.0]
    results = compare_molecule_sizes("Drug", dose_mg=1.0, mw_list=mw_list)
    halves = [r.t_half_vitreous_h for r in results]
    assert halves == sorted(halves, reverse=True)


def test_compare_molecule_sizes_all_results_valid():
    mw_list = [1.0, 100.0]
    results = compare_molecule_sizes("Drug", dose_mg=0.5, mw_list=mw_list)
    for r in results:
        assert isinstance(r, IntravitreалPKResult)
        assert r.t_half_vitreous_h > 0.0


# ── Output list lengths ───────────────────────────────────────────────────────

def test_output_lists_same_length():
    res = simulate_intravitreal_pk("Drug", dose_mg=1.0, mw_kDa=150.0, t_end_h=48.0, dt_h=1.0)
    n = len(res.times_h)
    assert len(res.a_vitreous_mg) == n
    assert len(res.a_aqueous_mg) == n
    assert len(res.c_plasma_mg_L) == n


def test_time_length():
    res = simulate_intravitreal_pk("Drug", dose_mg=1.0, mw_kDa=150.0, t_end_h=48.0, dt_h=1.0)
    assert len(res.times_h) == 49  # 0..48 inclusive


# ── Validation errors ─────────────────────────────────────────────────────────

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intravitreal_pk("X", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intravitreal_pk("X", dose_mg=-1.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_intravitreal_pk("X", dose_mg=1.0, mw_kDa=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_intravitreal_pk("X", dose_mg=1.0, vd_sys_L=0.0)


def test_validation_cl_negative():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_intravitreal_pk("X", dose_mg=1.0, cl_sys_L_per_h=-0.5)
