"""
Tests for Phase 247: Micro-dose PK predictor (prediction/micro_dose_pk_predictor.py).
"""

import math

import pytest

from omega_pbpk.prediction.micro_dose_pk_predictor import (
    MicrodosePKResult,
    compare_microdose_levels,
    predict_microdose_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        microdose_ug=100.0,
        conventional_dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        drug_class="standard",
    )
    defaults.update(kwargs)
    return predict_microdose_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_result()
    assert isinstance(result, MicrodosePKResult)


# ---------------------------------------------------------------------------
# Computed values: cmax and auc > 0
# ---------------------------------------------------------------------------


def test_cmax_microdose_positive():
    result = _make_result()
    assert result.cmax_microdose_ng_L > 0.0


def test_auc_microdose_positive():
    result = _make_result()
    assert result.auc_microdose_ng_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Exact arithmetic checks
# ---------------------------------------------------------------------------


def test_dose_ratio_exact():
    result = predict_microdose_pk(
        drug_name="D", microdose_ug=10.0, conventional_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0
    )
    # microdose_mg = 0.01, conventional = 100 mg -> ratio = 10000
    expected = 100.0 / (10.0 / 1000.0)
    assert math.isclose(result.dose_ratio, expected, rel_tol=1e-9)


def test_cmax_ng_L_formula():
    microdose_ug = 50.0
    vd_L = 20.0
    result = predict_microdose_pk(
        drug_name="D",
        microdose_ug=microdose_ug,
        conventional_dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=vd_L,
    )
    expected_ng_L = (microdose_ug / 1000.0) / vd_L * 1e6
    assert math.isclose(result.cmax_microdose_ng_L, expected_ng_L, rel_tol=1e-9)


def test_auc_ng_h_per_L_formula():
    microdose_ug = 50.0
    cl_L_per_h = 8.0
    result = predict_microdose_pk(
        drug_name="D",
        microdose_ug=microdose_ug,
        conventional_dose_mg=10.0,
        cl_L_per_h=cl_L_per_h,
        vd_L=50.0,
    )
    expected = (microdose_ug / 1000.0) / cl_L_per_h * 1e6
    assert math.isclose(result.auc_microdose_ng_h_per_L, expected, rel_tol=1e-9)


def test_auc_conventional_formula():
    conventional_dose_mg = 200.0
    cl_L_per_h = 10.0
    result = predict_microdose_pk(
        drug_name="D",
        microdose_ug=100.0,
        conventional_dose_mg=conventional_dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=50.0,
    )
    expected = conventional_dose_mg / cl_L_per_h
    assert math.isclose(result.auc_conventional_predicted_mg_h_per_L, expected, rel_tol=1e-9)


def test_cmax_scaling_factor_equals_dose_ratio():
    result = _make_result()
    assert math.isclose(result.cmax_scaling_factor, result.dose_ratio, rel_tol=1e-9)


def test_t_half_formula():
    cl_L_per_h = 5.0
    vd_L = 50.0
    result = predict_microdose_pk(
        drug_name="D",
        microdose_ug=10.0,
        conventional_dose_mg=10.0,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
    )
    expected_t_half = 0.693147 / (cl_L_per_h / vd_L)
    assert math.isclose(result.t_half_h, expected_t_half, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Higher CL -> lower AUC
# ---------------------------------------------------------------------------


def test_higher_cl_lower_auc():
    result_low = predict_microdose_pk(
        drug_name="D", microdose_ug=10.0, conventional_dose_mg=10.0, cl_L_per_h=2.0, vd_L=50.0
    )
    result_high = predict_microdose_pk(
        drug_name="D", microdose_ug=10.0, conventional_dose_mg=10.0, cl_L_per_h=20.0, vd_L=50.0
    )
    assert result_low.auc_microdose_ng_h_per_L > result_high.auc_microdose_ng_h_per_L


# ---------------------------------------------------------------------------
# Linearity flag
# ---------------------------------------------------------------------------


def test_linearity_expected_standard():
    result = _make_result(drug_class="standard")
    assert result.linearity_expected is True


def test_linearity_expected_false_tmdd():
    result = _make_result(drug_class="TMDD")
    assert result.linearity_expected is False


def test_linearity_expected_false_nti():
    result = _make_result(drug_class="NTI")
    assert result.linearity_expected is False


def test_linearity_expected_false_saturable():
    result = _make_result(drug_class="saturable")
    assert result.linearity_expected is False


# ---------------------------------------------------------------------------
# compare_microdose_levels
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_microdose_levels("D", 10.0, 5.0, 50.0)
    assert len(results) == 4


def test_compare_sorted_ascending():
    results = compare_microdose_levels("D", 10.0, 5.0, 50.0)
    levels = [r.microdose_ug for r in results]
    assert levels == sorted(levels)


def test_compare_default_levels():
    results = compare_microdose_levels("D", 10.0, 5.0, 50.0)
    levels = [r.microdose_ug for r in results]
    assert levels == [1.0, 10.0, 50.0, 100.0]


def test_compare_custom_levels():
    results = compare_microdose_levels("D", 10.0, 5.0, 50.0, microdose_levels_ug=[5.0, 20.0])
    assert len(results) == 2
    assert results[0].microdose_ug == 5.0
    assert results[1].microdose_ug == 20.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_microdose_exceeds_100_ug():
    with pytest.raises(ValueError, match="100"):
        predict_microdose_pk(
            "D", microdose_ug=101.0, conventional_dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0
        )


def test_validation_microdose_zero():
    with pytest.raises(ValueError):
        predict_microdose_pk(
            "D", microdose_ug=0.0, conventional_dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        predict_microdose_pk(
            "D", microdose_ug=10.0, conventional_dose_mg=10.0, cl_L_per_h=0.0, vd_L=50.0
        )


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        predict_microdose_pk(
            "D", microdose_ug=10.0, conventional_dose_mg=10.0, cl_L_per_h=5.0, vd_L=-1.0
        )


def test_validation_invalid_drug_class():
    with pytest.raises(ValueError, match="drug_class"):
        predict_microdose_pk(
            "D",
            microdose_ug=10.0,
            conventional_dose_mg=10.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            drug_class="unknown",
        )


def test_validation_conventional_dose_too_small():
    with pytest.raises(ValueError):
        # conventional (0.005 mg) <= microdose (10 ug = 0.01 mg)
        predict_microdose_pk(
            "D", microdose_ug=10.0, conventional_dose_mg=0.005, cl_L_per_h=5.0, vd_L=50.0
        )
