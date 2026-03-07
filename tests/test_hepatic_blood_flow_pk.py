"""Tests for hepatic blood flow-limited vs enzyme-limited clearance model."""

import pytest

from omega_pbpk.core.hepatic_blood_flow_pk import (
    HepaticClearanceResult,
    calculate_hepatic_clearance,
)

# ---------------------------------------------------------------------------
# Basic return-type and frozen tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert isinstance(result, HepaticClearanceResult)


def test_result_is_frozen():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "changed"  # type: ignore[misc]


def test_drug_name_preserved():
    result = calculate_hepatic_clearance("Warfarin", cl_int_L_per_h=10.0, fup=0.01)
    assert result.drug_name == "Warfarin"


def test_inputs_preserved():
    result = calculate_hepatic_clearance(
        "DrugB", cl_int_L_per_h=30.0, fup=0.5, q_hepatic_L_per_h=80.0
    )
    assert result.cl_int_L_per_h == 30.0
    assert result.fup == 0.5
    assert result.q_hepatic_L_per_h == 80.0


# ---------------------------------------------------------------------------
# Extraction ratio range tests
# ---------------------------------------------------------------------------


def test_er_well_stirred_in_range():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert 0.0 <= result.er_well_stirred < 1.0


def test_er_parallel_tube_in_range():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert 0.0 <= result.er_parallel_tube < 1.0


def test_er_dispersion_in_range():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert 0.0 <= result.er_dispersion < 1.0


# ---------------------------------------------------------------------------
# High CLint (flow-limited) tests
# ---------------------------------------------------------------------------


def test_high_clint_er_close_to_1():
    """Very high CLint >> Qh should yield ER close to 1."""
    result = calculate_hepatic_clearance(
        "HighER", cl_int_L_per_h=10000.0, fup=1.0, q_hepatic_L_per_h=97.0
    )
    assert result.er_well_stirred > 0.99
    assert result.er_parallel_tube > 0.99


def test_high_clint_flow_limited_label():
    result = calculate_hepatic_clearance("HighER", cl_int_L_per_h=10000.0, fup=1.0)
    assert result.cl_sensitivity_to_flow == "flow-limited"


# ---------------------------------------------------------------------------
# Low CLint (enzyme-limited) tests
# ---------------------------------------------------------------------------


def test_low_clint_er_proportional():
    """Low CLint << Qh: WS ER ≈ fup*CLint/Qh."""
    fup = 0.5
    cl_int = 1.0
    qh = 97.0
    result = calculate_hepatic_clearance(
        "LowER", cl_int_L_per_h=cl_int, fup=fup, q_hepatic_L_per_h=qh
    )
    expected_er_approx = fup * cl_int / qh
    assert abs(result.er_well_stirred - expected_er_approx) < 0.001


def test_low_clint_enzyme_limited_label():
    result = calculate_hepatic_clearance(
        "LowER", cl_int_L_per_h=0.5, fup=0.1, q_hepatic_L_per_h=97.0
    )
    assert result.cl_sensitivity_to_flow == "enzyme-limited"


# ---------------------------------------------------------------------------
# Model comparison: PT generally >= WS for intermediate drugs
# ---------------------------------------------------------------------------


def test_parallel_tube_gte_well_stirred():
    """For any finite CLint, PT ER >= WS ER (theoretical property)."""
    result = calculate_hepatic_clearance(
        "Mid", cl_int_L_per_h=50.0, fup=0.3, q_hepatic_L_per_h=97.0
    )
    assert result.er_parallel_tube >= result.er_well_stirred - 1e-10


def test_clearance_model_comparison_not_empty():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert isinstance(result.clearance_model_comparison, str)
    assert len(result.clearance_model_comparison) > 0


# ---------------------------------------------------------------------------
# CL and Fh consistency
# ---------------------------------------------------------------------------


def test_cl_ws_equals_qh_times_er():
    result = calculate_hepatic_clearance(
        "DrugX", cl_int_L_per_h=20.0, fup=0.2, q_hepatic_L_per_h=90.0
    )
    assert abs(result.cl_hepatic_ws - result.q_hepatic_L_per_h * result.er_well_stirred) < 1e-9


def test_fh_ws_equals_one_minus_er():
    result = calculate_hepatic_clearance(
        "DrugX", cl_int_L_per_h=20.0, fup=0.2, q_hepatic_L_per_h=90.0
    )
    assert abs(result.fh_well_stirred - (1.0 - result.er_well_stirred)) < 1e-9


def test_fh_pt_equals_one_minus_er_pt():
    result = calculate_hepatic_clearance(
        "DrugX", cl_int_L_per_h=20.0, fup=0.2, q_hepatic_L_per_h=90.0
    )
    assert abs(result.fh_parallel_tube - (1.0 - result.er_parallel_tube)) < 1e-9


def test_fh_disp_equals_one_minus_er_disp():
    result = calculate_hepatic_clearance(
        "DrugX", cl_int_L_per_h=20.0, fup=0.2, q_hepatic_L_per_h=90.0
    )
    assert abs(result.fh_dispersion - (1.0 - result.er_dispersion)) < 1e-9


# ---------------------------------------------------------------------------
# Sensitivity label
# ---------------------------------------------------------------------------


def test_intermediate_sensitivity():
    """fu_clint between 0.1*Qh and 3*Qh → intermediate."""
    qh = 97.0
    # fup*CLint = 20 → between 9.7 and 291
    result = calculate_hepatic_clearance("Mid", cl_int_L_per_h=20.0, fup=1.0, q_hepatic_L_per_h=qh)
    assert result.cl_sensitivity_to_flow == "intermediate"


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_not_empty():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Default Qh
# ---------------------------------------------------------------------------


def test_default_qh_is_97():
    result = calculate_hepatic_clearance("DrugA", cl_int_L_per_h=50.0, fup=0.1)
    assert result.q_hepatic_L_per_h == 97.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_clint_zero_raises():
    with pytest.raises(ValueError, match="cl_int_L_per_h"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=0.0, fup=0.1)


def test_clint_negative_raises():
    with pytest.raises(ValueError, match="cl_int_L_per_h"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=-5.0, fup=0.1)


def test_fup_zero_raises():
    with pytest.raises(ValueError, match="fup"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=50.0, fup=0.0)


def test_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=50.0, fup=1.1)


def test_qh_zero_raises():
    with pytest.raises(ValueError, match="q_hepatic_L_per_h"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=50.0, fup=0.1, q_hepatic_L_per_h=0.0)


def test_qh_negative_raises():
    with pytest.raises(ValueError, match="q_hepatic_L_per_h"):
        calculate_hepatic_clearance("Bad", cl_int_L_per_h=50.0, fup=0.1, q_hepatic_L_per_h=-10.0)
