"""Tests for Phase 653 — deep_tissue_pk module."""

import pytest

from omega_pbpk.core.deep_tissue_pk import (
    DeepTissuePKResult,
    compare_tissue_distribution,
    simulate_deep_tissue_pk,
)


def default_result(**kwargs) -> DeepTissuePKResult:
    """Return a result with default parameters, overriding with kwargs."""
    params = dict(drug_name="TestDrug", dose_mg=100.0)
    params.update(kwargs)
    return simulate_deep_tissue_pk(**params)


# ── basic structure ──────────────────────────────────────────────────────────


def test_returns_deep_tissue_pk_result():
    res = default_result()
    assert isinstance(res, DeepTissuePKResult)


def test_cmax_plasma_positive():
    res = default_result()
    assert res.cmax_plasma > 0


def test_auc_plasma_positive():
    res = default_result()
    assert res.auc_plasma > 0


def test_auc_deep_positive():
    res = default_result()
    assert res.auc_deep > 0


def test_deep_tissue_half_life_positive():
    res = default_result()
    assert res.deep_tissue_half_life_h > 0


def test_terminal_phase_t_half_positive():
    res = default_result()
    assert res.terminal_phase_t_half_h > 0


def test_accumulation_in_deep_positive():
    res = default_result()
    assert res.accumulation_in_deep > 0


def test_washout_time_90pct_positive():
    res = default_result()
    assert res.washout_time_90pct_h > 0


# ── concentration list lengths ───────────────────────────────────────────────


def test_length_consistency():
    res = default_result()
    n = len(res.times_h)
    assert n == len(res.c_plasma_mg_L) == len(res.c_deep_mg_L) == len(res.c_shallow_mg_L)


# ── pharmacokinetic behaviour ────────────────────────────────────────────────


def test_higher_q_deep_faster_equilibration():
    """Higher q_deep → deeper tissue fills faster → higher early AUC_deep."""
    res_fast = simulate_deep_tissue_pk("Drug", 100.0, q_deep_L_per_h=5.0, t_end_h=48.0)
    res_slow = simulate_deep_tissue_pk("Drug", 100.0, q_deep_L_per_h=0.2, t_end_h=48.0)
    # Fast equilibration should result in higher deep tissue AUC
    assert res_fast.auc_deep > res_slow.auc_deep


def test_lower_q_deep_slower_filling():
    """Lower q_deep → deep compartment fills more slowly → lower cmax_deep."""
    res_fast = simulate_deep_tissue_pk("Drug", 100.0, q_deep_L_per_h=5.0)
    res_slow = simulate_deep_tissue_pk("Drug", 100.0, q_deep_L_per_h=0.2)
    assert res_fast.auc_deep > res_slow.auc_deep


def test_larger_vd_deep_lower_concentration_ratio():
    """Larger Vd_deep dilutes the drug → lower cmax_deep concentration →
    lower accumulation_in_deep (cmax_deep/cmax_plasma ratio)."""
    res_large = simulate_deep_tissue_pk("Drug", 100.0, vd_deep_L=200.0)
    res_small = simulate_deep_tissue_pk("Drug", 100.0, vd_deep_L=20.0)
    # Smaller Vd_deep → higher deep-tissue concentration per unit mass → higher ratio
    assert res_small.accumulation_in_deep > res_large.accumulation_in_deep


def test_plasma_rises_then_falls():
    """Plasma concentrations rise initially and then decay — last point < first non-zero peak."""
    res = default_result()
    cp = res.c_plasma_mg_L
    # cmax should not be the last point
    assert res.cmax_plasma == max(cp)
    assert cp[-1] < res.cmax_plasma


def test_deep_peaks_later_than_plasma():
    """Deep tissue Tmax should be >= plasma Tmax for IV dosing."""
    res = default_result(route="iv")
    cd = res.c_deep_mg_L
    times = res.times_h
    tmax_deep = times[cd.index(max(cd))]
    assert tmax_deep >= res.tmax_plasma_h


# ── linearity ────────────────────────────────────────────────────────────────


def test_double_dose_doubles_auc_plasma():
    res1 = simulate_deep_tissue_pk("Drug", 100.0)
    res2 = simulate_deep_tissue_pk("Drug", 200.0)
    ratio = res2.auc_plasma / res1.auc_plasma
    assert abs(ratio - 2.0) < 0.05


def test_double_dose_doubles_auc_deep():
    res1 = simulate_deep_tissue_pk("Drug", 100.0)
    res2 = simulate_deep_tissue_pk("Drug", 200.0)
    ratio = res2.auc_deep / res1.auc_deep
    assert abs(ratio - 2.0) < 0.05


# ── compare_tissue_distribution ──────────────────────────────────────────────


def test_compare_returns_correct_length():
    scenarios = [
        {"label": "fast", "q_deep_L_per_h": 3.0, "vd_deep_L": 50.0},
        {"label": "slow", "q_deep_L_per_h": 0.5, "vd_deep_L": 150.0},
        {"label": "medium", "q_deep_L_per_h": 1.0, "vd_deep_L": 100.0},
    ]
    results = compare_tissue_distribution("Drug", 100.0, scenarios)
    assert len(results) == len(scenarios)


# ── oral route ───────────────────────────────────────────────────────────────


def test_oral_cmax_less_than_iv_bolus():
    """Oral Cmax < theoretical IV bolus concentration dose/Vd_plasma."""
    res = simulate_deep_tissue_pk("Drug", 100.0, route="oral", vd_plasma_L=10.0)
    theoretical_iv_cmax = 100.0 / 10.0  # 10 mg/L
    assert res.cmax_plasma < theoretical_iv_cmax


# ── validation ───────────────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_deep_tissue_pk("Drug", dose_mg=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_deep_tissue_pk("Drug", dose_mg=100.0, cl_plasma_L_per_h=0.0)


def test_validation_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_deep_tissue_pk("Drug", dose_mg=100.0, route="oral", f_oral=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_deep_tissue_pk("Drug", dose_mg=100.0, route="sc")


# ── notes ────────────────────────────────────────────────────────────────────


def test_notes_nonempty():
    res = default_result()
    assert res.notes and len(res.notes) > 0
