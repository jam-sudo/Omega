"""Tests for Phase 586 - Hepatic Blood Flow Effect on Drug Clearance."""

import pytest

from omega_pbpk.core.hepatic_flow_clearance_p586 import (
    HepaticFlowClearanceResult,
    predict_hepatic_flow_cl,
    sweep_flow_effects,
)

# --- Basic return type and field tests ---


def test_return_type():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.5)
    assert isinstance(r, HepaticFlowClearanceResult)


def test_all_fields_present():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.5)
    assert r.drug_name == "DrugA"
    assert r.q_hepatic_L_per_h == 90.0
    assert r.cl_intrinsic_L_per_h == 100.0
    assert r.fup == 0.5
    assert isinstance(r.cl_hepatic_L_per_h, float)
    assert isinstance(r.extraction_ratio, float)
    assert isinstance(r.flow_sensitivity_index, float)
    assert isinstance(r.cl_class, str)
    assert isinstance(r.dose_dependency_class, str)
    assert isinstance(r.notes, str)


def test_frozen_dataclass():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.5)
    with pytest.raises(AttributeError):
        r.drug_name = "changed"


# --- Extraction ratio tests ---


def test_extraction_ratio_in_range():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.5)
    assert 0.0 <= r.extraction_ratio <= 1.0


def test_extraction_ratio_calculation():
    """E = CL_h / Q = (Q * fu*CLint / (Q + fu*CLint)) / Q."""
    q, clint, fup = 90.0, 100.0, 0.5
    fu_clint = fup * clint  # 50
    cl_h = q * fu_clint / (q + fu_clint)  # 90*50/140 = 32.14
    expected_e = cl_h / q
    r = predict_hepatic_flow_cl("DrugA", q, clint, fup)
    assert abs(r.extraction_ratio - expected_e) < 1e-10


# --- Flow sensitivity index = E ---


def test_fsi_equals_e():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.5)
    assert abs(r.flow_sensitivity_index - r.extraction_ratio) < 1e-10


# --- Well-stirred model ---


def test_well_stirred_cl():
    q, clint, fup = 72.0, 200.0, 0.3
    fu_clint = fup * clint  # 60
    expected_cl = q * fu_clint / (q + fu_clint)  # 72*60/132 = 32.727
    r = predict_hepatic_flow_cl("DrugB", q, clint, fup)
    assert abs(r.cl_hepatic_L_per_h - expected_cl) < 1e-6


# --- Flow-limited drug (high E) ---


def test_flow_limited_class():
    """CLint=10000, fup=0.1 => fu*CLint=1000 >> Q=72 => high E."""
    r = predict_hepatic_flow_cl("HighE", 72.0, 10000.0, 0.1)
    assert r.extraction_ratio > 0.7
    assert r.cl_class == "flow_limited"
    assert r.dose_dependency_class == "highly_variable_with_flow"


# --- Capacity-limited drug (low E) ---


def test_capacity_limited_class():
    """CLint=10, fup=0.1 => fu*CLint=1 << Q=90 => low E."""
    r = predict_hepatic_flow_cl("LowE", 90.0, 10.0, 0.1)
    assert r.extraction_ratio < 0.3
    assert r.cl_class == "capacity_limited"
    assert r.dose_dependency_class == "flow_independent"


# --- Intermediate ---


def test_intermediate_class():
    """Find parameters giving 0.3 < E < 0.7."""
    # fu*CLint = 50, Q = 72 => E = 50/122 * ... let's calculate
    # CL = 72*50/122 = 29.5, E = 29.5/72 = 0.41 => intermediate
    r = predict_hepatic_flow_cl("MidE", 72.0, 100.0, 0.5)
    assert 0.3 <= r.extraction_ratio <= 0.7
    assert r.cl_class == "intermediate"
    assert r.dose_dependency_class == "moderately_variable"


# --- CL approaches Q for high extraction ---


def test_cl_approaches_q_for_high_e():
    r = predict_hepatic_flow_cl("HighE", 72.0, 100000.0, 1.0)
    # fu*CLint = 100000, CL ≈ Q
    assert r.cl_hepatic_L_per_h > 0.99 * r.q_hepatic_L_per_h


# --- CL approaches fu*CLint for low extraction ---


def test_cl_approaches_fu_clint_for_low_e():
    q, clint, fup = 90.0, 5.0, 0.1
    fu_clint = fup * clint  # 0.5
    r = predict_hepatic_flow_cl("LowE", q, clint, fup)
    assert abs(r.cl_hepatic_L_per_h - fu_clint) / fu_clint < 0.01


# --- Validation tests ---


def test_invalid_q_zero():
    with pytest.raises(ValueError, match="q_hepatic"):
        predict_hepatic_flow_cl("DrugA", 0.0, 100.0, 0.5)


def test_invalid_q_negative():
    with pytest.raises(ValueError, match="q_hepatic"):
        predict_hepatic_flow_cl("DrugA", -10.0, 100.0, 0.5)


def test_invalid_clint_zero():
    with pytest.raises(ValueError, match="cl_intrinsic"):
        predict_hepatic_flow_cl("DrugA", 90.0, 0.0, 0.5)


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 0.0)


def test_invalid_fup_above_1():
    with pytest.raises(ValueError, match="fup"):
        predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 1.5)


def test_fup_exactly_1():
    r = predict_hepatic_flow_cl("DrugA", 90.0, 100.0, 1.0)
    assert r.fup == 1.0


# --- sweep_flow_effects tests ---


def test_sweep_returns_list():
    results = sweep_flow_effects("DrugA", 100.0, 0.5)
    assert isinstance(results, list)
    assert len(results) == 6  # default 6 values


def test_sweep_sorted_ascending():
    results = sweep_flow_effects("DrugA", 100.0, 0.5)
    q_values = [r.q_hepatic_L_per_h for r in results]
    assert q_values == sorted(q_values)


def test_sweep_custom_q_values():
    results = sweep_flow_effects("DrugA", 100.0, 0.5, q_values=[10.0, 50.0, 100.0])
    assert len(results) == 3
    assert results[0].q_hepatic_L_per_h == 10.0
    assert results[-1].q_hepatic_L_per_h == 100.0


def test_sweep_cl_increases_with_q():
    """CL should increase as Q increases (for any drug)."""
    results = sweep_flow_effects("DrugA", 100.0, 0.5)
    cls = [r.cl_hepatic_L_per_h for r in results]
    for i in range(1, len(cls)):
        assert cls[i] > cls[i - 1]
