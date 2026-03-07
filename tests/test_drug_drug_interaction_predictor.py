"""Tests for drug_drug_interaction_predictor module (Phase 412)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.drug_drug_interaction_predictor import (
    ddi_matrix,
    predict_ddi_severity,
)

# ---------------------------------------------------------------------------
# predict_ddi_severity — competitive inhibition
# ---------------------------------------------------------------------------


def test_competitive_negligible():
    """Low c_plasma relative to Ki → negligible AUCR."""
    result = predict_ddi_severity(
        perpetrator_name="P1",
        victim_name="V1",
        ki_uM=100.0,
        c_plasma_uM=0.1,
        fm_cyp=0.5,
        inhibition_type="competitive",
    )
    assert result.ddi_risk_category == "negligible"
    # R1 = 1 + 0.1/100 = 1.001; AUCR = 0.5*1.001 + 0.5 = 1.0005
    assert result.aucr_estimate < 1.25


def test_competitive_mild():
    """Mild inhibition: 1.25 <= AUCR < 2."""
    result = predict_ddi_severity(
        perpetrator_name="P2",
        victim_name="V2",
        ki_uM=10.0,
        c_plasma_uM=5.0,
        fm_cyp=1.0,
        inhibition_type="competitive",
    )
    # R1 = 1 + 0.5 = 1.5; AUCR = 1.0*1.5 + 0 = 1.5 → mild
    assert result.ddi_risk_category == "mild"
    assert 1.25 <= result.aucr_estimate < 2.0


def test_competitive_moderate():
    """Moderate inhibition: 2 <= AUCR <= 5."""
    result = predict_ddi_severity(
        perpetrator_name="P3",
        victim_name="V3",
        ki_uM=1.0,
        c_plasma_uM=2.0,
        fm_cyp=1.0,
        inhibition_type="competitive",
    )
    # R1 = 3; AUCR = 3 → moderate
    assert result.ddi_risk_category == "moderate"


def test_competitive_severe():
    """Severe inhibition: AUCR > 5."""
    result = predict_ddi_severity(
        perpetrator_name="P4",
        victim_name="V4",
        ki_uM=1.0,
        c_plasma_uM=10.0,
        fm_cyp=1.0,
        inhibition_type="competitive",
    )
    # R1 = 11; AUCR = 11 → severe
    assert result.ddi_risk_category == "severe"
    assert result.aucr_estimate > 5.0


def test_competitive_r_value_formula():
    """R1 = 1 + C/Ki for competitive."""
    result = predict_ddi_severity("P", "V", ki_uM=4.0, c_plasma_uM=4.0, fm_cyp=1.0)
    assert math.isclose(result.r_value, 2.0, rel_tol=1e-6)


def test_competitive_fm_cyp_partial():
    """Partial fm_cyp dampens AUCR."""
    result_full = predict_ddi_severity("P", "V", 1.0, 9.0, 1.0)
    result_half = predict_ddi_severity("P", "V", 1.0, 9.0, 0.5)
    # full fm → AUCR=10; half fm → AUCR = 0.5*10 + 0.5 = 5.5
    assert result_half.aucr_estimate < result_full.aucr_estimate


# ---------------------------------------------------------------------------
# predict_ddi_severity — mechanism-based inhibition
# ---------------------------------------------------------------------------


def test_mbi_r_value():
    """MBI: R2 = 1 + 5*C/Ki."""
    result = predict_ddi_severity(
        "P", "V", ki_uM=5.0, c_plasma_uM=2.0, fm_cyp=1.0, inhibition_type="mechanism_based"
    )
    expected_r = 1.0 + 5.0 * 2.0 / 5.0
    assert math.isclose(result.r_value, expected_r, rel_tol=1e-6)


def test_mbi_higher_aucr_than_competitive():
    """MBI yields higher AUCR than competitive for same parameters."""
    comp = predict_ddi_severity("P", "V", 1.0, 1.0, 1.0, inhibition_type="competitive")
    mbi = predict_ddi_severity("P", "V", 1.0, 1.0, 1.0, inhibition_type="mechanism_based")
    assert mbi.aucr_estimate > comp.aucr_estimate


# ---------------------------------------------------------------------------
# predict_ddi_severity — induction
# ---------------------------------------------------------------------------


def test_induction_decreases_aucr():
    """Induction should yield AUCR < 1 (increased victim CL)."""
    result = predict_ddi_severity(
        "P",
        "V",
        ki_uM=1.0,
        c_plasma_uM=5.0,
        fm_cyp=1.0,
        inhibition_type="induction",
        induction_emax=10.0,
        induction_ec50_uM=1.0,
    )
    assert result.aucr_estimate < 1.0


def test_induction_zero_emax_no_effect():
    """Emax=0 → no induction → AUCR = 1.0 for full fm."""
    result = predict_ddi_severity(
        "P",
        "V",
        ki_uM=1.0,
        c_plasma_uM=5.0,
        fm_cyp=1.0,
        inhibition_type="induction",
        induction_emax=0.0,
        induction_ec50_uM=1.0,
    )
    # ind_factor = 0; r_value = 1; AUCR = 1/(1*1+0) = 1
    assert math.isclose(result.aucr_estimate, 1.0, rel_tol=1e-6)


def test_induction_ind_factor_formula():
    """Induction fold = Emax * C / (EC50 + C)."""
    emax, ec50, c = 4.0, 2.0, 2.0
    expected_ind = emax * c / (ec50 + c)  # = 2.0
    result = predict_ddi_severity(
        "P",
        "V",
        ki_uM=1.0,
        c_plasma_uM=c,
        fm_cyp=1.0,
        inhibition_type="induction",
        induction_emax=emax,
        induction_ec50_uM=ec50,
    )
    # r_value = 1 + ind_factor
    assert math.isclose(result.r_value, 1.0 + expected_ind, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# predict_ddi_severity — mixed inhibition
# ---------------------------------------------------------------------------


def test_mixed_uses_max_r():
    """Mixed: R = max(R1, R2)."""
    result = predict_ddi_severity(
        "P", "V", ki_uM=1.0, c_plasma_uM=1.0, fm_cyp=1.0, inhibition_type="mixed"
    )
    r1 = 1.0 + 1.0 / 1.0  # 2
    r2 = 1.0 + 5.0 * 1.0 / 1.0  # 6
    assert math.isclose(result.r_value, max(r1, r2), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# predict_ddi_severity — result fields
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    """DDIPredictionResult is immutable."""
    result = predict_ddi_severity("P", "V", 1.0, 1.0, 0.5)
    with pytest.raises(AttributeError):
        result.aucr_estimate = 99.0  # type: ignore[misc]


def test_fold_change_cl_reciprocal_aucr():
    """fold_change_cl = 1 / aucr."""
    result = predict_ddi_severity("P", "V", 1.0, 1.0, 1.0)
    assert math.isclose(result.fold_change_cl, 1.0 / result.aucr_estimate, rel_tol=1e-9)


def test_notes_not_empty():
    """Notes string should not be empty."""
    result = predict_ddi_severity("P", "V", 5.0, 1.0, 0.5)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# predict_ddi_severity — validation errors
# ---------------------------------------------------------------------------


def test_invalid_perpetrator_name():
    with pytest.raises(ValueError, match="perpetrator_name"):
        predict_ddi_severity("", "V", 1.0, 1.0, 0.5)


def test_invalid_victim_name():
    with pytest.raises(ValueError, match="victim_name"):
        predict_ddi_severity("P", "", 1.0, 1.0, 0.5)


def test_invalid_ki():
    with pytest.raises(ValueError, match="ki_uM"):
        predict_ddi_severity("P", "V", 0.0, 1.0, 0.5)


def test_invalid_c_plasma():
    with pytest.raises(ValueError, match="c_plasma"):
        predict_ddi_severity("P", "V", 1.0, -0.1, 0.5)


def test_invalid_fm_cyp():
    with pytest.raises(ValueError, match="fm_cyp"):
        predict_ddi_severity("P", "V", 1.0, 1.0, 1.5)


def test_invalid_inhibition_type():
    with pytest.raises(ValueError, match="inhibition_type"):
        predict_ddi_severity("P", "V", 1.0, 1.0, 0.5, inhibition_type="unknown")


def test_invalid_induction_emax():
    with pytest.raises(ValueError, match="induction_emax"):
        predict_ddi_severity(
            "P", "V", 1.0, 1.0, 0.5, inhibition_type="induction", induction_emax=-1.0
        )


def test_invalid_ec50():
    with pytest.raises(ValueError, match="induction_ec50"):
        predict_ddi_severity(
            "P", "V", 1.0, 1.0, 0.5, inhibition_type="induction", induction_ec50_uM=0.0
        )


# ---------------------------------------------------------------------------
# ddi_matrix tests
# ---------------------------------------------------------------------------


def test_ddi_matrix_shape():
    """Matrix is n×n."""
    compounds = [
        {"name": "A", "ki_uM": 1.0, "c_plasma_uM": 1.0, "fm_cyp": 0.9},
        {"name": "B", "ki_uM": 10.0, "c_plasma_uM": 0.1, "fm_cyp": 0.5},
        {"name": "C", "ki_uM": 5.0, "c_plasma_uM": 2.0, "fm_cyp": 0.8},
    ]
    matrix = ddi_matrix(compounds)
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)


def test_ddi_matrix_diagonal_self():
    """Diagonal entries are 'self'."""
    compounds = [
        {"name": "X", "ki_uM": 2.0, "c_plasma_uM": 1.0, "fm_cyp": 0.5},
        {"name": "Y", "ki_uM": 2.0, "c_plasma_uM": 1.0, "fm_cyp": 0.5},
    ]
    matrix = ddi_matrix(compounds)
    assert matrix[0][0] == "self"
    assert matrix[1][1] == "self"


def test_ddi_matrix_off_diagonal_categories():
    """Off-diagonal entries are valid risk categories."""
    valid = {"negligible", "mild", "moderate", "severe", "self", "unknown"}
    compounds = [
        {"name": "A", "ki_uM": 1.0, "c_plasma_uM": 5.0, "fm_cyp": 0.9},
        {"name": "B", "ki_uM": 100.0, "c_plasma_uM": 0.01, "fm_cyp": 0.1},
    ]
    matrix = ddi_matrix(compounds)
    for row in matrix:
        for cell in row:
            assert cell in valid, f"Unexpected category: {cell}"


def test_ddi_matrix_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ddi_matrix([])


def test_ddi_matrix_single_raises():
    with pytest.raises(ValueError, match="at least 2"):
        ddi_matrix([{"name": "A", "ki_uM": 1.0, "c_plasma_uM": 1.0, "fm_cyp": 0.5}])


def test_ddi_matrix_severe_perpetrator():
    """Strong inhibitor produces 'severe' risk against high fm victim."""
    compounds = [
        {"name": "StrongInh", "ki_uM": 0.1, "c_plasma_uM": 5.0, "fm_cyp": 1.0},
        {"name": "HighFm", "ki_uM": 10.0, "c_plasma_uM": 0.01, "fm_cyp": 0.99},
    ]
    matrix = ddi_matrix(compounds)
    # StrongInh (row 0) → HighFm (col 1)
    assert matrix[0][1] == "severe"
