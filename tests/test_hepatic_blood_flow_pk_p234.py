"""
Tests for Phase 234: Hepatic Blood Flow PK (Well-Stirred Model)
~20 tests covering return types, physical constraints, edge cases, validation.

Imports from core/hepatic_blood_flow_pk_p234.py (suffixed to avoid collision
with the existing core/hepatic_blood_flow_pk.py from an earlier phase).
"""

import pytest

from omega_pbpk.core.hepatic_blood_flow_pk_p234 import (
    HepaticExtractionResult,
    calculate_hepatic_extraction,
    sensitivity_to_blood_flow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic(cl_int=50.0, fu_b=0.5, qh=90.0):
    return calculate_hepatic_extraction(
        drug_name="TestDrug",
        cl_int_L_per_h=cl_int,
        fu_b=fu_b,
        hepatic_blood_flow_L_per_h=qh,
    )


# ---------------------------------------------------------------------------
# Return type and frozen
# ---------------------------------------------------------------------------


def test_return_type():
    result = _basic()
    assert isinstance(result, HepaticExtractionResult)


def test_result_is_frozen():
    result = _basic()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "changed"  # type: ignore[misc]


def test_drug_name_preserved():
    result = calculate_hepatic_extraction("Lidocaine", 50.0, 0.3)
    assert result.drug_name == "Lidocaine"


# ---------------------------------------------------------------------------
# Extraction ratio range
# ---------------------------------------------------------------------------


def test_extraction_ratio_in_range():
    result = _basic()
    assert 0.0 <= result.extraction_ratio <= 1.0


def test_extraction_ratio_high_for_high_clint():
    """fu_b*CLint >> Qh → E close to 1."""
    result = calculate_hepatic_extraction("HighER", 10000.0, 1.0, 90.0)
    assert result.extraction_ratio > 0.99


def test_extraction_ratio_low_for_low_clint():
    """fu_b*CLint << Qh → E close to 0."""
    result = calculate_hepatic_extraction("LowER", 0.1, 0.1, 90.0)
    assert result.extraction_ratio < 0.05


# ---------------------------------------------------------------------------
# CL and F consistency
# ---------------------------------------------------------------------------


def test_cl_hepatic_equals_qh_times_er():
    result = _basic()
    assert (
        abs(result.cl_hepatic_L_per_h - result.hepatic_blood_flow_L_per_h * result.extraction_ratio)
        < 1e-9
    )


def test_cl_hepatic_lte_blood_flow():
    result = _basic()
    assert result.cl_hepatic_L_per_h <= result.hepatic_blood_flow_L_per_h + 1e-9


def test_f_oral_hepatic_equals_one_minus_er():
    result = _basic()
    assert abs(result.f_oral_hepatic - (1.0 - result.extraction_ratio)) < 1e-12


def test_default_blood_flow_is_90():
    result = calculate_hepatic_extraction("Drug", 50.0, 0.5)
    assert result.hepatic_blood_flow_L_per_h == 90.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_low():
    # fu_b*CLint = 0.1*1=0.1, Qh=90 → E≈0.0011 < 0.3 → low
    result = calculate_hepatic_extraction("LowDrug", 0.1, 1.0, 90.0)
    assert result.cl_classification == "low"


def test_classification_high():
    # fu_b*CLint = 1.0*10000=10000, Qh=90 → E≈0.991 > 0.7 → high
    result = calculate_hepatic_extraction("HighDrug", 10000.0, 1.0, 90.0)
    assert result.cl_classification == "high"


def test_classification_intermediate():
    # E = fu_b*CLint / (Qh + fu_b*CLint) with fu_b*CLint = 45, Qh=90 → E=0.333 → intermediate
    result = calculate_hepatic_extraction("MidDrug", 45.0, 1.0, 90.0)
    assert result.cl_classification == "intermediate"


def test_classification_values_valid():
    for cl_int in [0.5, 45.0, 1000.0]:
        result = calculate_hepatic_extraction("Drug", cl_int, 1.0, 90.0)
        assert result.cl_classification in ("low", "intermediate", "high")


# ---------------------------------------------------------------------------
# sensitivity_to_blood_flow
# ---------------------------------------------------------------------------


def test_sensitivity_returns_list():
    results = sensitivity_to_blood_flow("Drug", 50.0, 0.5)
    assert isinstance(results, list)


def test_sensitivity_default_returns_five():
    results = sensitivity_to_blood_flow("Drug", 50.0, 0.5)
    assert len(results) == 5


def test_sensitivity_sorted_ascending_by_er():
    results = sensitivity_to_blood_flow("Drug", 50.0, 0.5)
    er_values = [r.extraction_ratio for r in results]
    assert er_values == sorted(er_values)


def test_sensitivity_custom_range():
    results = sensitivity_to_blood_flow("Drug", 50.0, 0.5, [60.0, 90.0, 120.0])
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_cl_int_zero_raises():
    with pytest.raises(ValueError, match="cl_int_L_per_h"):
        calculate_hepatic_extraction("Bad", 0.0, 0.5)


def test_cl_int_negative_raises():
    with pytest.raises(ValueError, match="cl_int_L_per_h"):
        calculate_hepatic_extraction("Bad", -10.0, 0.5)


def test_fu_b_zero_raises():
    with pytest.raises(ValueError, match="fu_b"):
        calculate_hepatic_extraction("Bad", 50.0, 0.0)


def test_fu_b_above_one_raises():
    with pytest.raises(ValueError, match="fu_b"):
        calculate_hepatic_extraction("Bad", 50.0, 1.5)


def test_blood_flow_zero_raises():
    with pytest.raises(ValueError, match="hepatic_blood_flow_L_per_h"):
        calculate_hepatic_extraction("Bad", 50.0, 0.5, 0.0)


def test_blood_flow_negative_raises():
    with pytest.raises(ValueError, match="hepatic_blood_flow_L_per_h"):
        calculate_hepatic_extraction("Bad", 50.0, 0.5, -90.0)
