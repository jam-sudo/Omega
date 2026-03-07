"""Tests for Phase 245 — Pharmacokinetic Drug Interaction Index.

Covers:
  - R1 formula: 1 + c_inlet / ki
  - FDA action thresholds: no_action / evaluate_in_vivo / significant_ddi
  - R1_gut formula
  - R2 when c_ss_uM is provided
  - composite_score >= r1
  - composite_score amplified by victim TI
  - ddi_risk_matrix: count, sort order
  - Input validation errors
  - Result field types and values
"""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ddi_index import (
    DDIIndexResult,
    compute_ddi_index,
    ddi_risk_matrix,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    perpetrator_name="itraconazole",
    victim_name="midazolam",
    ki_uM=1.0,
    c_inlet_uM=1.0,
    dose_mg=100.0,
    ka_per_h=1.0,
    fg=0.5,
    cl_int_gut_L_per_h=1.0,
)


def _make(**kwargs):
    kw = {**BASE_KWARGS, **kwargs}
    return compute_ddi_index(**kw)


# ---------------------------------------------------------------------------
# R1 formula tests
# ---------------------------------------------------------------------------


def test_r1_formula_basic():
    """R1 = 1 + c_inlet / ki."""
    r = _make(c_inlet_uM=1.0, ki_uM=1.0)
    assert r.r1 == pytest.approx(2.0)


def test_r1_formula_zero_inlet():
    """c_inlet=0 → R1 = 1.0 (no inhibition)."""
    r = _make(c_inlet_uM=0.0, ki_uM=1.0)
    assert r.r1 == pytest.approx(1.0)


def test_r1_formula_low_ratio():
    """Small c_inlet/ki → R1 just above 1."""
    r = _make(c_inlet_uM=0.05, ki_uM=1.0)
    assert r.r1 == pytest.approx(1.05)


def test_r1_formula_high_ratio():
    """c_inlet = 10*ki → R1 = 11."""
    r = _make(c_inlet_uM=10.0, ki_uM=1.0)
    assert r.r1 == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# FDA action thresholds
# ---------------------------------------------------------------------------


def test_fda_action_no_action():
    """c_inlet very low → R1 < 1.1 → 'no_action'."""
    r = _make(c_inlet_uM=0.05, ki_uM=1.0)
    assert r.fda_action == "no_action"


def test_fda_action_evaluate_in_vivo_low_boundary():
    """R1 exactly at 1.1 → 'evaluate_in_vivo'."""
    r = _make(c_inlet_uM=0.1, ki_uM=1.0)
    assert r.r1 == pytest.approx(1.1)
    assert r.fda_action == "evaluate_in_vivo"


def test_fda_action_evaluate_in_vivo_midrange():
    """1.1 <= R1 < 2 → 'evaluate_in_vivo'."""
    r = _make(c_inlet_uM=0.8, ki_uM=1.0)
    assert 1.1 <= r.r1 < 2.0
    assert r.fda_action == "evaluate_in_vivo"


def test_fda_action_significant_ddi_boundary():
    """R1 = 2.0 exactly → 'significant_ddi'."""
    r = _make(c_inlet_uM=1.0, ki_uM=1.0)
    assert r.r1 == pytest.approx(2.0)
    assert r.fda_action == "significant_ddi"


def test_fda_action_significant_ddi_high():
    """R1 = 11 → 'significant_ddi'."""
    r = _make(c_inlet_uM=10.0, ki_uM=1.0)
    assert r.r1 == pytest.approx(11.0)
    assert r.fda_action == "significant_ddi"


# ---------------------------------------------------------------------------
# R1_gut
# ---------------------------------------------------------------------------


def test_r1_gut_formula():
    """R1_gut = 1 + ka * dose * fg / (CL_int_gut * Ki)."""
    ka, dose, fg, cl, ki = 1.0, 100.0, 0.5, 1.0, 1.0
    expected = 1.0 + (ka * dose * fg) / (cl * ki)
    r = _make(ka_per_h=ka, dose_mg=dose, fg=fg, cl_int_gut_L_per_h=cl, ki_uM=ki)
    assert r.r1_gut == pytest.approx(expected)


def test_r1_gut_high_inhibition():
    """Large dose / small Ki → R1_gut > R1."""
    r = _make(
        ka_per_h=2.0, dose_mg=500.0, fg=0.9, cl_int_gut_L_per_h=0.5, ki_uM=1.0, c_inlet_uM=0.1
    )
    assert r.r1_gut > r.r1


def test_r1_gut_is_not_none():
    """R1_gut is always computed (not None)."""
    r = _make()
    assert r.r1_gut is not None


# ---------------------------------------------------------------------------
# R2 (dynamic index)
# ---------------------------------------------------------------------------


def test_r2_none_when_no_css():
    """Without c_ss_uM, R2 should be None."""
    r = _make()
    assert r.r2 is None


def test_r2_formula():
    """R2 = 1 + c_ss / ki when c_ss_uM provided."""
    r = _make(c_ss_uM=2.0, ki_uM=1.0)
    assert r.r2 == pytest.approx(3.0)


def test_r2_css_equals_ki():
    """c_ss = ki → R2 = 2.0."""
    r = _make(c_ss_uM=1.0, ki_uM=1.0)
    assert r.r2 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_composite_score_gte_r1():
    """composite_score >= r1 always."""
    r = _make(c_inlet_uM=1.0, ki_uM=1.0)
    assert r.composite_score >= r.r1


def test_composite_score_amplified_by_ti():
    """Narrow TI victim → higher composite_score than without TI."""
    r_no_ti = _make(c_inlet_uM=1.0, ki_uM=1.0)
    r_with_ti = _make(c_inlet_uM=1.0, ki_uM=1.0, victim_ti=2.0)
    assert r_with_ti.composite_score > r_no_ti.composite_score


def test_composite_score_no_ti_factor():
    """Without victim_ti, composite_score = max(r1, r1_gut)."""
    r = _make(c_inlet_uM=0.0, ki_uM=1.0)
    assert r.composite_score == pytest.approx(max(r.r1, r.r1_gut))


# ---------------------------------------------------------------------------
# ddi_risk_matrix
# ---------------------------------------------------------------------------


def test_ddi_risk_matrix_count():
    """Matrix produces n_perp * n_victim results."""
    perps = [
        {"name": "drugA", "ki_uM": 1.0, "c_inlet_uM": 1.0, "dose_mg": 100.0, "ka_per_h": 1.0},
        {"name": "drugB", "ki_uM": 5.0, "c_inlet_uM": 0.5, "dose_mg": 200.0, "ka_per_h": 0.5},
    ]
    victims = [
        {"name": "sub1"},
        {"name": "sub2"},
        {"name": "sub3"},
    ]
    results = ddi_risk_matrix(perps, victims)
    assert len(results) == 6


def test_ddi_risk_matrix_sorted_descending():
    """Results are sorted by composite_score descending."""
    perps = [
        {"name": "weak", "ki_uM": 100.0, "c_inlet_uM": 0.01, "dose_mg": 10.0, "ka_per_h": 0.5},
        {"name": "strong", "ki_uM": 0.1, "c_inlet_uM": 10.0, "dose_mg": 500.0, "ka_per_h": 2.0},
    ]
    victims = [{"name": "sub"}]
    results = ddi_risk_matrix(perps, victims)
    scores = [r.composite_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ddi_risk_matrix_single_pair():
    """Single perpetrator × single victim → 1 result."""
    perps = [{"name": "p", "ki_uM": 1.0, "c_inlet_uM": 1.0, "dose_mg": 100.0, "ka_per_h": 1.0}]
    victims = [{"name": "v"}]
    results = ddi_risk_matrix(perps, victims)
    assert len(results) == 1
    assert isinstance(results[0], DDIIndexResult)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validation_ki_zero():
    with pytest.raises(ValueError, match="ki_uM"):
        _make(ki_uM=0.0)


def test_validation_ki_negative():
    with pytest.raises(ValueError, match="ki_uM"):
        _make(ki_uM=-1.0)


def test_validation_c_inlet_negative():
    with pytest.raises(ValueError, match="c_inlet_uM"):
        _make(c_inlet_uM=-0.5)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _make(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _make(dose_mg=-100.0)


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_field_types():
    r = _make(c_ss_uM=1.0, victim_ti=3.0)
    assert isinstance(r, DDIIndexResult)
    assert isinstance(r.perpetrator_name, str)
    assert isinstance(r.victim_name, str)
    assert isinstance(r.ki_uM, float)
    assert isinstance(r.c_inlet_uM, float)
    assert isinstance(r.r1, float)
    assert isinstance(r.r1_gut, float)
    assert isinstance(r.r2, float)
    assert isinstance(r.composite_score, float)
    assert isinstance(r.fda_action, str)
    assert isinstance(r.notes, list)
