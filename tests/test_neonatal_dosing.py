"""Tests for Phase 939 — neonatal_dosing module."""

import pytest
from omega_pbpk.clinical.neonatal_dosing import (
    NeonatalDosingResult,
    simulate_neonatal_dosing,
    compare_age_groups,
)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------

def test_return_type_iv():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert isinstance(r, NeonatalDosingResult)


def test_cmax_positive():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert r.cmax_mg_L > 0


def test_auc_positive():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert r.auc_mg_h_per_L > 0


def test_t_half_positive():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert r.t_half_h > 0


# ---------------------------------------------------------------------------
# Clinical correctness
# ---------------------------------------------------------------------------

def test_t_half_fold_increase_greater_than_1_all_groups():
    """All neonatal age groups should have longer t_half than adult."""
    groups = ["premature", "term_neonate", "infant_1_6m", "infant_6_12m", "toddler_1_2y"]
    for ag in groups:
        r = simulate_neonatal_dosing("drug_A", 5.0, ag)
        assert r.t_half_fold_increase > 1.0, f"Expected fold > 1 for {ag}, got {r.t_half_fold_increase}"


def test_premature_longer_than_toddler():
    """Premature neonate has longest t_half due to most immature CL."""
    r_premature = simulate_neonatal_dosing("drug_A", 5.0, "premature")
    r_toddler = simulate_neonatal_dosing("drug_A", 5.0, "toddler_1_2y")
    assert r_premature.t_half_h > r_toddler.t_half_h


def test_patient_weight_positive():
    r = simulate_neonatal_dosing("drug_A", 5.0, "infant_1_6m")
    assert r.patient_weight_kg > 0


def test_dose_mg_matches_weight():
    r = simulate_neonatal_dosing("drug_A", 5.0, "infant_1_6m")
    assert abs(r.dose_mg - 5.0 * r.patient_weight_kg) < 1e-9


def test_cl_effective_less_than_adult():
    """Effective CL in neonates should be less than adult CL."""
    cl_adult = 10.0
    groups = ["premature", "term_neonate", "infant_1_6m", "infant_6_12m", "toddler_1_2y"]
    for ag in groups:
        r = simulate_neonatal_dosing("drug_A", 5.0, ag, cl_adult_L_per_h=cl_adult)
        assert r.cl_effective_L_per_h < cl_adult, f"CL not reduced for {ag}"


def test_adult_t_half_positive():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert r.adult_t_half_h > 0


# ---------------------------------------------------------------------------
# compare_age_groups
# ---------------------------------------------------------------------------

def test_compare_age_groups_returns_5():
    results = compare_age_groups("drug_A", 5.0)
    assert len(results) == 5


def test_compare_age_groups_sorted_descending():
    results = compare_age_groups("drug_A", 5.0)
    t_halves = [r.t_half_h for r in results]
    assert t_halves == sorted(t_halves, reverse=True)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_nonempty():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate")
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Route-specific behaviour
# ---------------------------------------------------------------------------

def test_iv_initial_concentration_positive():
    """IV bolus: first concentration should be > 0."""
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", route="iv")
    assert r.c_plasma_mg_L[0] > 0


def test_oral_initial_concentration_zero():
    """Oral route: concentration at t=0 should be 0 (drug in gut, not plasma)."""
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", route="oral")
    assert r.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Multi-dose
# ---------------------------------------------------------------------------

def test_n_doses_3_runs():
    """n_doses=3 should run without error."""
    r = simulate_neonatal_dosing(
        "drug_A", 5.0, "term_neonate", n_doses=3, dosing_interval_h=12.0, t_end_h=48.0
    )
    assert r.cmax_mg_L > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_dose_mg_per_kg():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_neonatal_dosing("drug_A", -1.0, "term_neonate")


def test_zero_dose_mg_per_kg():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_neonatal_dosing("drug_A", 0.0, "term_neonate")


def test_invalid_cl_adult():
    with pytest.raises(ValueError, match="cl_adult_L_per_h"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", cl_adult_L_per_h=-1.0)


def test_zero_cl_adult():
    with pytest.raises(ValueError, match="cl_adult_L_per_h"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", cl_adult_L_per_h=0.0)


def test_invalid_vd_adult():
    with pytest.raises(ValueError, match="vd_adult_L"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", vd_adult_L=-5.0)


def test_zero_vd_adult():
    with pytest.raises(ValueError, match="vd_adult_L"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", vd_adult_L=0.0)


def test_invalid_age_group():
    with pytest.raises(ValueError, match="age_group"):
        simulate_neonatal_dosing("drug_A", 5.0, "adult")


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", route="subcutaneous")


def test_invalid_n_doses():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", n_doses=0)


# ---------------------------------------------------------------------------
# Vd scaling
# ---------------------------------------------------------------------------

def test_vd_premature_less_than_toddler():
    """Premature (1 kg) has smaller absolute Vd than toddler (12 kg)."""
    r_premature = simulate_neonatal_dosing("drug_A", 5.0, "premature")
    r_toddler = simulate_neonatal_dosing("drug_A", 5.0, "toddler_1_2y")
    assert r_premature.vd_effective_L < r_toddler.vd_effective_L


# ---------------------------------------------------------------------------
# Times list
# ---------------------------------------------------------------------------

def test_times_nonempty():
    r = simulate_neonatal_dosing("drug_A", 5.0, "term_neonate", t_end_h=24.0)
    assert len(r.times_h) > 0
