"""Tests for core/hepatic_first_pass.py — Phase 457."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.hepatic_first_pass import (
    HepaticFirstPassResult,
    simulate_hepatic_first_pass,
)

# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert isinstance(result, HepaticFirstPassResult)


def test_drug_name_preserved():
    result = simulate_hepatic_first_pass("TestDrug", dose_mg=50.0)
    assert result.drug_name == "TestDrug"


def test_dose_preserved():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=200.0)
    assert result.dose_mg == 200.0


# ---------------------------------------------------------------------------
# Bounds and physical constraints
# ---------------------------------------------------------------------------


def test_f_hepatic_between_0_and_1():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert 0.0 < result.f_hepatic <= 1.0


def test_extraction_ratio_between_0_and_1():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert 0.0 <= result.extraction_ratio < 1.0


def test_f_hepatic_plus_extraction_ratio_equals_1():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert math.isclose(result.f_hepatic + result.extraction_ratio, 1.0, abs_tol=1e-9)


def test_cmax_systemic_positive():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert result.cmax_systemic_mg_L > 0.0


def test_auc_systemic_positive():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# High vs low extraction
# ---------------------------------------------------------------------------


def test_high_clint_low_f_hepatic():
    """Very high intrinsic clearance -> high extraction -> low F_h."""
    result = simulate_hepatic_first_pass(
        "HighExt", dose_mg=100.0, cl_int_L_per_h=1000.0, fu_plasma=0.5
    )
    assert result.f_hepatic < 0.2


def test_low_clint_high_f_hepatic():
    """Very low intrinsic clearance -> low extraction -> high F_h."""
    result = simulate_hepatic_first_pass("LowExt", dose_mg=100.0, cl_int_L_per_h=0.1, fu_plasma=0.1)
    assert result.f_hepatic > 0.9


def test_high_extraction_high_extraction_ratio():
    result = simulate_hepatic_first_pass(
        "HighExt", dose_mg=100.0, cl_int_L_per_h=5000.0, fu_plasma=1.0
    )
    assert result.extraction_ratio > 0.9


def test_low_fu_reduces_extraction():
    """Lower fu -> less free drug -> lower hepatic clearance -> higher F_h."""
    r_high_fu = simulate_hepatic_first_pass("D", dose_mg=100.0, fu_plasma=0.8)
    r_low_fu = simulate_hepatic_first_pass("D", dose_mg=100.0, fu_plasma=0.05)
    assert r_low_fu.f_hepatic > r_high_fu.f_hepatic


# ---------------------------------------------------------------------------
# Linearity (2x dose -> 2x AUC, 2x Cmax)
# ---------------------------------------------------------------------------


def test_linearity_auc():
    r1 = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    r2 = simulate_hepatic_first_pass("DrugA", dose_mg=200.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert math.isclose(ratio, 2.0, rel_tol=0.01)


def test_linearity_cmax():
    r1 = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    r2 = simulate_hepatic_first_pass("DrugA", dose_mg=200.0)
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert math.isclose(ratio, 2.0, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------


def test_times_length_matches_portal():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_portal_mg_L)


def test_times_length_matches_hv():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_hepatic_vein_mg_L)


def test_times_length_matches_systemic():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_systemic_mg_L)


def test_times_starts_at_zero():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert result.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert len(result.notes) > 0


def test_notes_contains_extraction():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert "E=" in result.notes


# ---------------------------------------------------------------------------
# Hepatic clearance value
# ---------------------------------------------------------------------------


def test_hepatic_clearance_positive():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0)
    assert result.hepatic_clearance_L_per_h > 0.0


def test_hepatic_clearance_less_than_qh():
    result = simulate_hepatic_first_pass("DrugA", dose_mg=100.0, Q_h_L_per_h=90.0)
    assert result.hepatic_clearance_L_per_h < 90.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=-50.0)


def test_invalid_ka_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, ka_per_h=0.0)


def test_invalid_qh_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, Q_h_L_per_h=0.0)


def test_invalid_fu_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, fu_plasma=0.0)


def test_invalid_fu_above_one_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, fu_plasma=1.5)


def test_invalid_clint_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, cl_int_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_first_pass("DrugA", dose_mg=100.0, vd_sys_L=0.0)
