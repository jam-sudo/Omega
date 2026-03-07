"""Tests for Phase 1137 — Hemodialysis PK model."""

import math

import pytest

from omega_pbpk.core.hemodialysis_pk import (
    HemodialysisPKResult,
    simulate_hemodialysis_pk,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=500.0,
        mw_Da=300.0,
        fu_plasma=0.9,
        cl_residual_L_per_h=1.0,
        vd_L=30.0,
        dialysis_duration_h=4.0,
        dialysis_start_h=0.0,
        t_end_h=12.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_hemodialysis_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_result_type():
    res = run_default()
    assert isinstance(res, HemodialysisPKResult)


def test_drug_name_preserved():
    res = run_default(drug_name="Vancomycin")
    assert res.drug_name == "Vancomycin"


def test_dose_preserved():
    res = run_default(dose_mg=250.0)
    assert res.dose_mg == 250.0


def test_mw_preserved():
    res = run_default(mw_Da=500.0)
    assert res.mw_Da == 500.0


# ---------------------------------------------------------------------------
# Concentration profile sanity
# ---------------------------------------------------------------------------

def test_cmax_equals_initial_concentration():
    dose, vd = 600.0, 30.0
    res = run_default(dose_mg=dose, vd_L=vd, dialysis_start_h=0.0)
    expected_c0 = dose / vd
    assert abs(res.cmax_plasma_mg_L - expected_c0) < 1e-6


def test_concentration_non_negative():
    res = run_default()
    assert all(c >= 0 for c in res.c_plasma_mg_L)


def test_concentration_monotonically_decreasing_during_dialysis():
    res = run_default(dialysis_start_h=0.0)
    # During dialysis window (first 4 h), concentrations should decrease
    n_dial = int(4.0 / 0.1)
    dial_conc = res.c_plasma_mg_L[: n_dial + 1]
    for i in range(1, len(dial_conc)):
        assert dial_conc[i] <= dial_conc[i - 1] + 1e-10


def test_time_vector_length_matches_concentration():
    res = run_default(t_end_h=10.0, dt_h=0.1)
    assert len(res.times_h) == len(res.c_plasma_mg_L)


def test_time_starts_at_zero():
    res = run_default()
    assert res.times_h[0] == 0.0


def test_time_ends_at_t_end():
    res = run_default(t_end_h=12.0, dt_h=0.1)
    assert abs(res.times_h[-1] - 12.0) < 0.15


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def test_auc_positive():
    res = run_default()
    assert res.auc_plasma_mg_h_per_L > 0


def test_auc_during_dialysis_less_than_total():
    res = run_default(dialysis_start_h=0.0, dialysis_duration_h=4.0, t_end_h=12.0)
    assert res.auc_during_dialysis_mg_h_per_L < res.auc_plasma_mg_h_per_L


def test_auc_during_dialysis_non_negative():
    res = run_default()
    assert res.auc_during_dialysis_mg_h_per_L >= 0


# ---------------------------------------------------------------------------
# Small vs large molecule
# ---------------------------------------------------------------------------

def test_small_freely_unbound_drug_high_hd_clearance():
    res_small = run_default(mw_Da=200.0, fu_plasma=0.99)
    res_large = run_default(mw_Da=10000.0, fu_plasma=0.05)
    assert res_small.hd_clearance_L_per_h > res_large.hd_clearance_L_per_h


def test_small_freely_unbound_drug_high_fraction_removed():
    res = run_default(mw_Da=150.0, fu_plasma=1.0, vd_L=10.0, dose_mg=100.0,
                      cl_residual_L_per_h=0.0, dialysis_start_h=0.0,
                      dialysis_duration_h=4.0, t_end_h=8.0, dt_h=0.05)
    assert res.fraction_removed_by_hd > 0.30


def test_large_protein_bound_drug_low_fraction_removed():
    res = run_default(mw_Da=15000.0, fu_plasma=0.01, vd_L=200.0, dose_mg=100.0,
                      dialysis_duration_h=4.0, t_end_h=8.0)
    assert res.fraction_removed_by_hd < 0.30


# ---------------------------------------------------------------------------
# Fraction removed
# ---------------------------------------------------------------------------

def test_fraction_removed_in_unit_interval():
    res = run_default()
    assert 0.0 <= res.fraction_removed_by_hd <= 1.0


def test_dose_supplement_recommended_when_fraction_gt_30pct():
    res = run_default(mw_Da=150.0, fu_plasma=1.0, vd_L=10.0, dose_mg=100.0,
                      cl_residual_L_per_h=0.0, dialysis_start_h=0.0,
                      dialysis_duration_h=4.0, t_end_h=8.0, dt_h=0.05)
    if res.fraction_removed_by_hd > 0.30:
        assert res.dose_supplement_recommended is True
        assert res.supplemental_dose_mg > 0


def test_no_supplement_when_fraction_le_30pct():
    res = run_default(mw_Da=15000.0, fu_plasma=0.01, vd_L=200.0, dose_mg=100.0,
                      dialysis_duration_h=4.0, t_end_h=8.0)
    if res.fraction_removed_by_hd <= 0.30:
        assert res.dose_supplement_recommended is False
        assert res.supplemental_dose_mg == 0.0


def test_supplemental_dose_proportional():
    res = run_default(mw_Da=150.0, fu_plasma=1.0, vd_L=10.0, dose_mg=100.0,
                      cl_residual_L_per_h=0.0, dialysis_start_h=0.0,
                      dialysis_duration_h=4.0, t_end_h=8.0, dt_h=0.05)
    if res.dose_supplement_recommended:
        expected = res.dose_mg * res.fraction_removed_by_hd
        assert abs(res.supplemental_dose_mg - expected) < 1e-6


# ---------------------------------------------------------------------------
# MW class
# ---------------------------------------------------------------------------

def test_mw_class_small():
    res = run_default(mw_Da=300.0)
    assert res.mw_class == "small"


def test_mw_class_medium():
    res = run_default(mw_Da=1000.0)
    assert res.mw_class == "medium"


def test_mw_class_large():
    res = run_default(mw_Da=50000.0)
    assert res.mw_class == "large"


# ---------------------------------------------------------------------------
# HD clearance
# ---------------------------------------------------------------------------

def test_hd_clearance_positive():
    res = run_default()
    assert res.hd_clearance_L_per_h > 0


def test_hd_clearance_decreases_with_higher_mw():
    res_low = run_default(mw_Da=100.0, fu_plasma=1.0)
    res_high = run_default(mw_Da=8000.0, fu_plasma=1.0)
    assert res_low.hd_clearance_L_per_h > res_high.hd_clearance_L_per_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_raises_on_non_positive_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hemodialysis_pk("X", dose_mg=0.0, mw_Da=300.0)


def test_raises_on_negative_cl_residual():
    with pytest.raises(ValueError, match="cl_residual"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=300.0, cl_residual_L_per_h=-1.0)


def test_raises_on_non_positive_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=300.0, vd_L=0.0)


def test_raises_on_non_positive_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=0.0)


def test_raises_on_fu_out_of_range():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=300.0, fu_plasma=0.0)


def test_raises_on_fu_gt_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=300.0, fu_plasma=1.5)


def test_raises_on_non_positive_dialysis_duration():
    with pytest.raises(ValueError, match="dialysis_duration_h"):
        simulate_hemodialysis_pk("X", dose_mg=100.0, mw_Da=300.0, dialysis_duration_h=0.0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------

def test_notes_contains_mw_class():
    res = run_default(mw_Da=300.0)
    assert "small" in res.notes
