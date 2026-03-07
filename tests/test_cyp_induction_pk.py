"""Tests for Phase 756 — Drug Interaction with CYP Inducers."""

import math

import pytest

from omega_pbpk.clinical.cyp_induction_pk import (
    CYPInductionPKResult,
    assess_induction_severity,
    simulate_cyp_induction_pk,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = simulate_cyp_induction_pk("MidazolamTest")
    assert isinstance(result, CYPInductionPKResult)


# ---------------------------------------------------------------------------
# AUC decreases with induction
# ---------------------------------------------------------------------------


def test_day_0_auc_greater_than_day_n_auc():
    r = simulate_cyp_induction_pk("Drug", fold_induction=5.0, n_days=14)
    assert r.day_0_auc > r.day_n_auc


def test_auc_ratio_final_less_than_1():
    r = simulate_cyp_induction_pk("Drug2", fold_induction=5.0, n_days=14)
    assert r.auc_ratio_final < 1.0


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------


def test_days_list_length():
    r = simulate_cyp_induction_pk("Drug3", n_days=14)
    assert len(r.days) == 15  # day 0 to day 14


def test_days_list_length_custom():
    r = simulate_cyp_induction_pk("Drug4", n_days=7)
    assert len(r.days) == 8


def test_cl_list_length_matches_days():
    r = simulate_cyp_induction_pk("Drug5", n_days=14)
    assert len(r.cl_per_day) == len(r.days)


def test_auc_list_length_matches_days():
    r = simulate_cyp_induction_pk("Drug6", n_days=14)
    assert len(r.auc_per_day) == len(r.days)


def test_cmax_list_length_matches_days():
    r = simulate_cyp_induction_pk("Drug7", n_days=14)
    assert len(r.cmax_per_day) == len(r.days)


# ---------------------------------------------------------------------------
# CL increases over time
# ---------------------------------------------------------------------------


def test_cl_increases_over_days():
    r = simulate_cyp_induction_pk("Drug8", fold_induction=5.0, n_days=14)
    # CL at day 14 must be higher than at day 0
    assert r.cl_per_day[-1] > r.cl_per_day[0]


def test_cl_day0_equals_baseline():
    r = simulate_cyp_induction_pk("Drug9", cl_baseline_L_per_h=5.0, fold_induction=3.0, n_days=14)
    # At day 0, induction_fraction=0, so CL = baseline
    assert math.isclose(r.cl_per_day[0], 5.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Higher fold induction → lower AUC ratio
# ---------------------------------------------------------------------------


def test_higher_fold_induction_lower_auc_ratio():
    r_low = simulate_cyp_induction_pk("Drug10", fold_induction=2.0, n_days=14)
    r_high = simulate_cyp_induction_pk("Drug10b", fold_induction=10.0, n_days=14)
    assert r_high.auc_ratio_final < r_low.auc_ratio_final


# ---------------------------------------------------------------------------
# time_to_50pct_max_induction_days
# ---------------------------------------------------------------------------


def test_time_to_50pct_positive():
    r = simulate_cyp_induction_pk("Drug11", t_half_enzyme_h=70.0)
    assert r.time_to_50pct_max_induction_days > 0


def test_time_to_50pct_analytical():
    """t50 = ln(2) / (ln(2)/t_half * 24) = t_half / 24."""
    t_half = 70.0
    r = simulate_cyp_induction_pk("Drug12", t_half_enzyme_h=t_half)
    expected = t_half / 24.0
    assert math.isclose(r.time_to_50pct_max_induction_days, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# assess_induction_severity
# ---------------------------------------------------------------------------


def test_assess_severity_returns_str():
    result = assess_induction_severity(0.6)
    assert isinstance(result, str)


def test_assess_severity_none():
    assert assess_induction_severity(0.9) == "none"


def test_assess_severity_mild():
    assert assess_induction_severity(0.65) == "mild"


def test_assess_severity_moderate():
    assert assess_induction_severity(0.3) == "moderate"


def test_assess_severity_severe():
    assert assess_induction_severity(0.1) == "severe"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_fold_induction_eq1_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad", fold_induction=1.0)


def test_validation_fold_induction_less_than_1_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad2", fold_induction=0.5)


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad3", dose_victim_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad4", dose_victim_mg=-50.0)


def test_validation_cl_baseline_zero_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad5", cl_baseline_L_per_h=0.0)


def test_validation_n_days_zero_raises():
    with pytest.raises(ValueError):
        simulate_cyp_induction_pk("Bad6", n_days=0)


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = simulate_cyp_induction_pk("NotesCheck")
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# fold_induction stored
# ---------------------------------------------------------------------------


def test_fold_induction_stored():
    r = simulate_cyp_induction_pk("Drug13", fold_induction=8.0)
    assert math.isclose(r.fold_induction, 8.0)
