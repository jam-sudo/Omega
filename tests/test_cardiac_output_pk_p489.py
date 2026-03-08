"""Tests for Phase 489 — Cardiac Output Effect on PK."""

import pytest

from omega_pbpk.core.cardiac_output_pk_p489 import (
    CardiacOutputPKResult,
    compare_co_conditions,
    predict_cardiac_output_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
FU = 0.3
CLINT = 20.0  # L/h
VD = 50.0


def normal_result():
    return predict_cardiac_output_pk(DRUG, "normal", FU, CLINT, VD)


def hf_severe_result():
    return predict_cardiac_output_pk(DRUG, "heart_failure_severe", FU, CLINT, VD)


def exercise_intense_result():
    return predict_cardiac_output_pk(DRUG, "exercise_intense", FU, CLINT, VD)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = normal_result()
    assert isinstance(result, CardiacOutputPKResult)


def test_result_is_frozen():
    result = normal_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Normal condition sanity
# ---------------------------------------------------------------------------


def test_normal_condition_auc_ratio():
    result = normal_result()
    assert abs(result.auc_ratio_vs_normal - 1.0) < 1e-6


def test_normal_condition_co():
    result = normal_result()
    assert result.cardiac_output_L_per_min == 5.0


def test_normal_condition_pk_change_no_change():
    result = normal_result()
    assert result.pk_change == "no_change"


# ---------------------------------------------------------------------------
# Heart failure → lower CL vs normal
# ---------------------------------------------------------------------------


def test_heart_failure_severe_lower_hepatic_flow():
    normal = normal_result()
    hf = hf_severe_result()
    assert hf.Q_hepatic_L_per_h < normal.Q_hepatic_L_per_h


def test_heart_failure_severe_lower_total_cl():
    normal = normal_result()
    hf = hf_severe_result()
    assert hf.cl_total_L_per_h < normal.cl_total_L_per_h


def test_heart_failure_severe_auc_ratio_gt_1():
    hf = hf_severe_result()
    assert hf.auc_ratio_vs_normal > 1.0


def test_heart_failure_mild_auc_ratio_gt_1():
    result = predict_cardiac_output_pk(DRUG, "heart_failure_mild", FU, CLINT, VD)
    assert result.auc_ratio_vs_normal > 1.0


def test_heart_failure_severe_auc_greater_than_mild():
    mild = predict_cardiac_output_pk(DRUG, "heart_failure_mild", FU, CLINT, VD)
    severe = hf_severe_result()
    assert severe.auc_ratio_vs_normal > mild.auc_ratio_vs_normal


# ---------------------------------------------------------------------------
# t_half = 0.693 * Vd / CL (exact)
# ---------------------------------------------------------------------------


def test_t_half_formula_normal():
    result = normal_result()
    expected = 0.693 * result.vd_L / result.cl_total_L_per_h
    assert abs(result.t_half_h - expected) < 1e-4


def test_t_half_formula_hf_severe():
    result = hf_severe_result()
    expected = 0.693 * result.vd_L / result.cl_total_L_per_h
    assert abs(result.t_half_h - expected) < 1e-4


def test_t_half_formula_exercise_intense():
    result = exercise_intense_result()
    expected = 0.693 * result.vd_L / result.cl_total_L_per_h
    assert abs(result.t_half_h - expected) < 1e-4


def test_t_half_vd_effect():
    r1 = predict_cardiac_output_pk(DRUG, "normal", FU, CLINT, 50.0)
    r2 = predict_cardiac_output_pk(DRUG, "normal", FU, CLINT, 100.0)
    assert r2.t_half_h > r1.t_half_h


# ---------------------------------------------------------------------------
# GFR scales with CO
# ---------------------------------------------------------------------------


def test_gfr_normal_is_120():
    result = normal_result()
    assert abs(result.GFR_mL_per_min - 120.0) < 1e-4


def test_gfr_lower_in_heart_failure():
    normal = normal_result()
    hf = hf_severe_result()
    assert hf.GFR_mL_per_min < normal.GFR_mL_per_min


# ---------------------------------------------------------------------------
# Exercise conditions
# ---------------------------------------------------------------------------


def test_exercise_intense_high_co():
    result = exercise_intense_result()
    assert result.cardiac_output_L_per_min == 20.0


def test_exercise_moderate_co():
    result = predict_cardiac_output_pk(DRUG, "exercise_moderate", FU, CLINT, VD)
    assert result.cardiac_output_L_per_min == 12.0


# ---------------------------------------------------------------------------
# pk_change classification
# ---------------------------------------------------------------------------


def test_pk_change_categories_valid():
    valid = {"no_change", "minor", "moderate", "major"}
    for cond in [
        "normal",
        "heart_failure_mild",
        "heart_failure_severe",
        "exercise_moderate",
        "exercise_intense",
    ]:
        result = predict_cardiac_output_pk(DRUG, cond, FU, CLINT, VD)
        assert result.pk_change in valid


def test_pk_change_heart_failure_severe_moderate_or_major():
    hf = hf_severe_result()
    assert hf.pk_change in {"moderate", "major"}


# ---------------------------------------------------------------------------
# compare_co_conditions
# ---------------------------------------------------------------------------


def test_compare_co_conditions_returns_5():
    results = compare_co_conditions(DRUG, FU, CLINT, VD)
    assert len(results) == 5


def test_compare_co_conditions_sorted_descending():
    results = compare_co_conditions(DRUG, FU, CLINT, VD)
    for i in range(len(results) - 1):
        assert results[i].cl_total_L_per_h >= results[i + 1].cl_total_L_per_h


def test_compare_co_conditions_all_result_types():
    results = compare_co_conditions(DRUG, FU, CLINT, VD)
    for r in results:
        assert isinstance(r, CardiacOutputPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_condition_raises():
    with pytest.raises(ValueError, match="co_condition"):
        predict_cardiac_output_pk(DRUG, "coma", FU, CLINT, VD)


def test_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_cardiac_output_pk(DRUG, "normal", 0.0, CLINT, VD)


def test_fu_gt_1_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_cardiac_output_pk(DRUG, "normal", 1.5, CLINT, VD)


def test_clint_zero_raises():
    with pytest.raises(ValueError, match="cl_int"):
        predict_cardiac_output_pk(DRUG, "normal", FU, 0.0, VD)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        predict_cardiac_output_pk(DRUG, "normal", FU, CLINT, 0.0)


def test_compare_invalid_fu_raises():
    with pytest.raises(ValueError):
        compare_co_conditions(DRUG, 0.0, CLINT, VD)
