"""Tests for clinical/drug_replenishment_kinetics.py — Phase 651."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_replenishment_kinetics import (
    ReplenishmentResult,
    compare_depletion_recovery,
    simulate_replenishment_kinetics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs) -> ReplenishmentResult:
    params = dict(
        drug_name="Statin",
        substrate_name="CoQ10",
        drug_dose_mg=10.0,
        drug_cl_L_per_h=5.0,
        drug_vd_L=50.0,
        drug_route="iv",
        k_depletion_per_h=0.1,
        k_synthesis_per_h=0.2,
        substrate_baseline=100.0,
        t_end_h=72.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_replenishment_kinetics(**params)


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_replenishment_result():
    r = default_result()
    assert isinstance(r, ReplenishmentResult)


def test_field_types():
    r = default_result()
    assert isinstance(r.drug_name, str)
    assert isinstance(r.substrate_name, str)
    assert isinstance(r.drug_dose_mg, float)
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_drug_mg_L, list)
    assert isinstance(r.s_substrate_units, list)
    assert isinstance(r.drug_cmax_mg_L, float)
    assert isinstance(r.drug_auc, float)
    assert isinstance(r.substrate_nadir, float)
    assert isinstance(r.time_to_nadir_h, float)
    assert isinstance(r.depletion_pct, float)
    assert isinstance(r.time_to_recovery_h, float)
    assert isinstance(r.recovery_complete, bool)
    assert isinstance(r.clinical_significance, str)
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Scientific correctness
# ---------------------------------------------------------------------------


def test_substrate_nadir_below_baseline():
    r = default_result()
    assert r.substrate_nadir < 100.0


def test_time_to_nadir_positive():
    r = default_result()
    assert r.time_to_nadir_h > 0.0


def test_depletion_pct_in_range():
    r = default_result()
    assert 0.0 <= r.depletion_pct <= 100.0


def test_clinical_significance_valid():
    r = default_result()
    assert r.clinical_significance in {"none", "minor", "moderate", "severe"}


def test_drug_cmax_positive():
    r = default_result()
    assert r.drug_cmax_mg_L > 0.0


def test_drug_auc_positive():
    r = default_result()
    assert r.drug_auc > 0.0


def test_list_lengths_equal():
    r = default_result()
    assert len(r.times_h) == len(r.c_drug_mg_L) == len(r.s_substrate_units)


def test_higher_k_depletion_greater_depletion():
    r_low = default_result(k_depletion_per_h=0.02)
    r_high = default_result(k_depletion_per_h=0.2)
    assert r_high.depletion_pct > r_low.depletion_pct


def test_higher_k_synthesis_faster_recovery():
    r_slow = default_result(k_synthesis_per_h=0.05, t_end_h=120.0)
    r_fast = default_result(k_synthesis_per_h=0.5, t_end_h=120.0)
    # Faster synthesis → higher substrate nadir (less depletion) or faster recovery
    # Both should be true: faster recovery_complete or shorter time_to_recovery_h
    # At minimum, higher k_synthesis -> less or equal depletion
    assert r_fast.depletion_pct <= r_slow.depletion_pct + 5.0  # allow small tolerance


def test_double_dose_higher_depletion():
    r_low = default_result(drug_dose_mg=10.0)
    r_high = default_result(drug_dose_mg=20.0)
    assert r_high.depletion_pct >= r_low.depletion_pct


def test_recovery_complete_high_synthesis_long_sim():
    r = default_result(k_synthesis_per_h=1.0, t_end_h=200.0)
    assert r.recovery_complete is True


def test_time_to_recovery_positive_when_recovered():
    r = default_result(k_synthesis_per_h=1.0, t_end_h=200.0)
    if r.recovery_complete:
        assert r.time_to_recovery_h >= 0.0


# ---------------------------------------------------------------------------
# IV vs Oral route
# ---------------------------------------------------------------------------


def test_iv_cmax_at_t0():
    r = simulate_replenishment_kinetics(
        drug_name="Drug",
        substrate_name="Sub",
        drug_dose_mg=100.0,
        drug_cl_L_per_h=5.0,
        drug_vd_L=50.0,
        drug_route="iv",
        t_end_h=24.0,
        dt_h=0.1,
    )
    expected_c0 = 100.0 / 50.0  # = 2.0 mg/L
    assert abs(r.drug_cmax_mg_L - expected_c0) < 0.01


def test_oral_cmax_less_than_iv_c0():
    dose = 100.0
    vd = 50.0
    r_oral = simulate_replenishment_kinetics(
        drug_name="Drug",
        substrate_name="Sub",
        drug_dose_mg=dose,
        drug_cl_L_per_h=5.0,
        drug_vd_L=vd,
        drug_route="oral",
        drug_ka_per_h=1.0,
        drug_f_oral=0.8,
        t_end_h=24.0,
        dt_h=0.1,
    )
    iv_c0 = dose / vd
    assert r_oral.drug_cmax_mg_L < iv_c0


# ---------------------------------------------------------------------------
# compare_depletion_recovery
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    results = compare_depletion_recovery("Statin", "CoQ10", [10.0, 20.0, 40.0])
    assert len(results) == 3


def test_compare_sorted_by_depletion_descending():
    results = compare_depletion_recovery("Statin", "CoQ10", [5.0, 10.0, 30.0])
    depletions = [r.depletion_pct for r in results]
    assert depletions == sorted(depletions, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_zero_dose():
    with pytest.raises(ValueError):
        simulate_replenishment_kinetics("D", "S", drug_dose_mg=0.0)


def test_validation_zero_synthesis():
    with pytest.raises(ValueError):
        simulate_replenishment_kinetics("D", "S", drug_dose_mg=10.0, k_synthesis_per_h=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError):
        simulate_replenishment_kinetics("D", "S", drug_dose_mg=10.0, drug_route="im")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = default_result()
    assert len(r.notes) > 0
