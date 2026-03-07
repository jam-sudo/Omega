"""Tests for Phase 940 — perioperative_pk module."""

import pytest
from omega_pbpk.core.perioperative_pk import (
    PerioperativePKResult,
    simulate_perioperative_pk,
    compare_hypothermia_levels,
)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------

def test_return_type():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert isinstance(r, PerioperativePKResult)


def test_cmax_positive():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert r.cmax_mg_L > 0


def test_auc_positive():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert r.auc_mg_h_per_L > 0


def test_auc_intraop_nonnegative():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert r.auc_intraop_mg_h_per_L >= 0


# ---------------------------------------------------------------------------
# Surgical phase effects
# ---------------------------------------------------------------------------

def test_cl_fold_intraop_less_than_1():
    """CL during surgery should be reduced below normal."""
    r = simulate_perioperative_pk("drug_B", 100.0, hypothermia_temp_C=37.0)
    assert r.cl_fold_intraop < 1.0


def test_cl_fold_intraop_normothermia():
    """At 37C, intraop CL fold should be 0.6 (phase multiplier only)."""
    r = simulate_perioperative_pk("drug_B", 100.0, hypothermia_temp_C=37.0)
    assert r.cl_fold_intraop == pytest.approx(0.6, rel=1e-6)


# ---------------------------------------------------------------------------
# Hypothermia
# ---------------------------------------------------------------------------

def test_hypothermia_reduction_nonnegative():
    r = simulate_perioperative_pk("drug_B", 100.0, hypothermia_temp_C=37.0)
    assert r.hypothermia_cl_reduction_pct >= 0


def test_normothermia_zero_reduction():
    """At 37C, no hypothermia CL reduction."""
    r = simulate_perioperative_pk("drug_B", 100.0, hypothermia_temp_C=37.0)
    assert r.hypothermia_cl_reduction_pct == 0.0


def test_mild_hypothermia_14_pct():
    """At 35C (2 degrees below 37), reduction = 14%."""
    r = simulate_perioperative_pk("drug_B", 100.0, hypothermia_temp_C=35.0)
    assert r.hypothermia_cl_reduction_pct == pytest.approx(14.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_hypothermia_levels
# ---------------------------------------------------------------------------

def test_compare_hypothermia_returns_4():
    results = compare_hypothermia_levels("drug_B", 100.0)
    assert len(results) == 4


def test_compare_hypothermia_sorted_by_auc_descending():
    results = compare_hypothermia_levels("drug_B", 100.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_deeper_hypothermia_higher_auc():
    """Deepest hypothermia (32C) should give highest AUC vs normothermia (37C)."""
    results = compare_hypothermia_levels("drug_B", 100.0, temps_C=[37.0, 32.0])
    # sorted descending by AUC, so first = highest AUC = lowest temp
    assert results[0].hypothermia_cl_reduction_pct > results[1].hypothermia_cl_reduction_pct


# ---------------------------------------------------------------------------
# Route-specific behaviour
# ---------------------------------------------------------------------------

def test_iv_bolus_initial_concentration_positive():
    """IV bolus: concentration at t=0 should be > 0."""
    r = simulate_perioperative_pk("drug_B", 100.0, route="iv_bolus")
    assert r.c_plasma_mg_L[0] > 0


def test_iv_infusion_initial_concentration_zero():
    """IV infusion: concentration at t=0 should be 0 (infusion builds from 0)."""
    r = simulate_perioperative_pk(
        "drug_B", 100.0, route="iv_infusion", infusion_rate_mg_h=10.0
    )
    assert r.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Blood loss
# ---------------------------------------------------------------------------

def test_blood_loss_reduces_concentration():
    """Blood loss should produce a lower or equal concentration compared to no blood loss."""
    r_no_loss = simulate_perioperative_pk("drug_B", 100.0, blood_loss_fraction=0.0)
    r_with_loss = simulate_perioperative_pk("drug_B", 100.0, blood_loss_fraction=0.3)
    # AUC should be lower or equal with blood loss
    assert r_with_loss.auc_mg_h_per_L <= r_no_loss.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_nonempty():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# tmax
# ---------------------------------------------------------------------------

def test_tmax_nonnegative():
    r = simulate_perioperative_pk("drug_B", 100.0)
    assert r.tmax_h >= 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_dose_mg():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_perioperative_pk("drug_B", -10.0)


def test_zero_dose_mg():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_perioperative_pk("drug_B", 0.0)


def test_invalid_cl_normal():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        simulate_perioperative_pk("drug_B", 100.0, cl_normal_L_per_h=-1.0)


def test_zero_cl_normal():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        simulate_perioperative_pk("drug_B", 100.0, cl_normal_L_per_h=0.0)


def test_invalid_vd_normal():
    with pytest.raises(ValueError, match="vd_normal_L"):
        simulate_perioperative_pk("drug_B", 100.0, vd_normal_L=-5.0)


def test_zero_vd_normal():
    with pytest.raises(ValueError, match="vd_normal_L"):
        simulate_perioperative_pk("drug_B", 100.0, vd_normal_L=0.0)


def test_blood_loss_fraction_negative():
    with pytest.raises(ValueError, match="blood_loss_fraction"):
        simulate_perioperative_pk("drug_B", 100.0, blood_loss_fraction=-0.1)


def test_blood_loss_fraction_above_0_5():
    with pytest.raises(ValueError, match="blood_loss_fraction"):
        simulate_perioperative_pk("drug_B", 100.0, blood_loss_fraction=0.6)


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_perioperative_pk("drug_B", 100.0, route="oral")
