"""Tests for Phase 168 — Multidrug Resistance (MDR) Modulation."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.mdr_modulation import MDRResult, screen_mdr_inhibitors, simulate_mdr_effect

# ── default parameters for convenience ──────────────────────────────────────

DEFAULTS = dict(
    drug_name="DrugX",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=70.0,
)


# ── basic behaviour ─────────────────────────────────────────────────────────


def test_baseline_fold_change_is_one():
    r = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.0)
    assert r.mdr_fold_change == 1.0


def test_full_inhibition_increases_intracellular():
    base = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.0)
    full = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=1.0)
    assert full.auc_intracellular > base.auc_intracellular


def test_higher_inhibitor_fraction_higher_auc_intra():
    low = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.3)
    high = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.8)
    assert high.auc_intracellular > low.auc_intracellular


def test_mdr_fold_change_greater_than_one_with_inhibitor():
    r = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.5)
    assert r.mdr_fold_change > 1.0


def test_default_parameters_work():
    r = simulate_mdr_effect(**DEFAULTS)
    assert isinstance(r, MDRResult)


def test_c_plasma_starts_at_dose_over_vd():
    r = simulate_mdr_effect(**DEFAULTS)
    expected = DEFAULTS["dose_mg"] / DEFAULTS["vd_L"]
    assert r.c_plasma_mg_L[0] == pytest.approx(expected)


def test_c_intra_starts_at_zero():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.c_intracellular_mg_L[0] == 0.0


def test_times_start_at_zero():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.times_h[0] == 0.0


def test_times_end_near_t_end():
    r = simulate_mdr_effect(**DEFAULTS, t_end_h=12.0)
    assert abs(r.times_h[-1] - 12.0) < 0.11


def test_all_concentrations_non_negative():
    r = simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=0.9)
    assert all(c >= 0 for c in r.c_plasma_mg_L)
    assert all(c >= 0 for c in r.c_intracellular_mg_L)


def test_auc_values_positive():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.auc_plasma > 0
    assert r.auc_intracellular > 0


def test_cmax_plasma_non_negative():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.cmax_plasma >= 0


def test_cmax_intracellular_non_negative():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.cmax_intracellular >= 0


def test_intracellular_to_plasma_ratio_positive():
    r = simulate_mdr_effect(**DEFAULTS)
    assert r.intracellular_to_plasma_ratio > 0


def test_notes_non_empty():
    r = simulate_mdr_effect(**DEFAULTS)
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_higher_efflux_reduces_intracellular():
    low_efflux = simulate_mdr_effect(**DEFAULTS, mdr_efflux_rate_per_h=0.2)
    high_efflux = simulate_mdr_effect(**DEFAULTS, mdr_efflux_rate_per_h=2.0)
    assert low_efflux.auc_intracellular > high_efflux.auc_intracellular


# ── screen_mdr_inhibitors ──────────────────────────────────────────────────


def test_screen_returns_correct_count():
    inhibitors = [
        {"inhibitor_name": "InhA", "inhibitor_fraction": 0.3},
        {"inhibitor_name": "InhB", "inhibitor_fraction": 0.7},
    ]
    results = screen_mdr_inhibitors(**DEFAULTS, inhibitors=inhibitors)
    assert len(results) == 2


def test_screen_sorted_descending():
    inhibitors = [
        {"inhibitor_name": "InhA", "inhibitor_fraction": 0.2},
        {"inhibitor_name": "InhB", "inhibitor_fraction": 0.9},
        {"inhibitor_name": "InhC", "inhibitor_fraction": 0.5},
    ]
    results = screen_mdr_inhibitors(**DEFAULTS, inhibitors=inhibitors)
    ratios = [r.intracellular_to_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_empty_list_raises():
    with pytest.raises(ValueError):
        screen_mdr_inhibitors(**DEFAULTS, inhibitors=[])


# ── validation errors ───────────────────────────────────────────────────────


def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(drug_name="X", dose_mg=0, cl_L_per_h=5, vd_L=70)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(drug_name="X", dose_mg=-10, cl_L_per_h=5, vd_L=70)


def test_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(drug_name="X", dose_mg=100, cl_L_per_h=0, vd_L=70)


def test_vd_zero_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(drug_name="X", dose_mg=100, cl_L_per_h=5, vd_L=0)


def test_inhibitor_fraction_above_one_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=1.5)


def test_inhibitor_fraction_below_zero_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, inhibitor_fraction=-0.1)


def test_t_end_zero_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, t_end_h=0)


def test_t_end_negative_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, t_end_h=-5)


def test_negative_efflux_rate_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, mdr_efflux_rate_per_h=-1)


def test_negative_k_in_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, k_in_per_h=-0.5)


def test_negative_k_out_raises():
    with pytest.raises(ValueError):
        simulate_mdr_effect(**DEFAULTS, k_out_per_h=-0.1)
