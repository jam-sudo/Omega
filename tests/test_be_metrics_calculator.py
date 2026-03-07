"""
Tests for Phase 900 — Drug Bioequivalence Metric Calculator
"""

import math
import pytest

from omega_pbpk.clinical.be_metrics_calculator import (
    BEMetricsResult,
    calculate_be_metrics,
    compare_formulation_be,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

REF_TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
REF_CONC   = [0.0, 0.8, 1.2, 1.0, 0.7, 0.5, 0.3, 0.15]

# Test formulation nearly identical to reference (should be BE)
TEST_TIMES_BE = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
TEST_CONC_BE  = [0.0, 0.82, 1.18, 1.0, 0.72, 0.51, 0.31, 0.16]

# Test formulation with much lower exposure (should fail BE)
TEST_TIMES_FAIL = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
TEST_CONC_FAIL  = [0.0, 0.2, 0.3, 0.25, 0.18, 0.12, 0.07, 0.03]


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert isinstance(result, BEMetricsResult)


def test_frozen_dataclass():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_field_names_stored():
    result = calculate_be_metrics(
        "Aspirin", "IR", "ReferenceIR",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert result.drug_name == "Aspirin"
    assert result.test_formulation == "IR"
    assert result.reference_formulation == "ReferenceIR"


# ---------------------------------------------------------------------------
# AUCt computation
# ---------------------------------------------------------------------------


def test_auc_t_positive():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert result.test_auc_t > 0.0
    assert result.ref_auc_t > 0.0


def test_auc_inf_geq_auc_t():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert result.test_auc_inf >= result.test_auc_t
    assert result.ref_auc_inf >= result.ref_auc_t


# ---------------------------------------------------------------------------
# Cmax and Tmax
# ---------------------------------------------------------------------------


def test_cmax_equals_profile_max():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert math.isclose(result.test_cmax, max(TEST_CONC_BE), rel_tol=1e-9)
    assert math.isclose(result.ref_cmax, max(REF_CONC), rel_tol=1e-9)


def test_tmax_correct():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    # Cmax of REF_CONC is 1.2 at index 2 → time 1.0
    assert math.isclose(result.ref_tmax_h, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------


def test_auc_ratio_near_one_for_similar_profiles():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert 0.9 < result.auc_ratio < 1.1


def test_cmax_ratio_near_one_for_similar_profiles():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert 0.9 < result.cmax_ratio < 1.1


def test_auc_ratio_low_for_failing_formulation():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_FAIL, TEST_CONC_FAIL,
        REF_TIMES, REF_CONC,
    )
    assert result.auc_ratio < 0.5


# ---------------------------------------------------------------------------
# 90% CI
# ---------------------------------------------------------------------------


def test_90ci_lower_less_than_upper():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
    )
    assert result.auc_90ci_lower < result.auc_90ci_upper
    assert result.cmax_90ci_lower < result.cmax_90ci_upper


def test_90ci_symmetric_around_ratio():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
        cv_intra_pct=15.0,
    )
    half_width = 1.645 * 15.0 / 100.0
    expected_lower = result.auc_ratio * (1.0 - half_width)
    expected_upper = result.auc_ratio * (1.0 + half_width)
    assert math.isclose(result.auc_90ci_lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(result.auc_90ci_upper, expected_upper, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# BE conclusion
# ---------------------------------------------------------------------------


def test_be_demonstrated_similar_profiles():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
        cv_intra_pct=5.0,  # tight CI
    )
    assert result.be_conclusion == "BE demonstrated"


def test_be_not_demonstrated_very_different():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_FAIL, TEST_CONC_FAIL,
        REF_TIMES, REF_CONC,
    )
    assert result.be_conclusion == "BE not demonstrated"


def test_inconclusive_for_borderline():
    # ratio ~0.85 but CV=20% → CI may straddle 80%
    times = [0.0, 1.0, 2.0, 4.0, 8.0]
    conc_ref  = [0.0, 1.0, 0.8, 0.5, 0.2]
    conc_test = [0.0, 0.85, 0.68, 0.425, 0.17]
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        times, conc_test,
        times, conc_ref,
        cv_intra_pct=20.0,
    )
    # auc_ratio ~0.85; half_width=0.329 → lower ~0.57, upper ~1.13
    # not "not demonstrated" (ratio>0.5), not "demonstrated" (lower<0.8)
    assert result.be_conclusion in {"Inconclusive — additional data needed", "BE demonstrated", "BE not demonstrated"}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_be_demonstrated():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_BE, TEST_CONC_BE,
        REF_TIMES, REF_CONC,
        cv_intra_pct=5.0,
    )
    assert "80-125%" in result.notes


def test_notes_be_not_demonstrated():
    result = calculate_be_metrics(
        "Drug", "Test", "Ref",
        TEST_TIMES_FAIL, TEST_CONC_FAIL,
        REF_TIMES, REF_CONC,
    )
    assert "reformulation" in result.notes.lower() or "additional studies" in result.notes.lower()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mismatched_lengths_test_raises():
    with pytest.raises(ValueError, match="same length"):
        calculate_be_metrics(
            "Drug", "Test", "Ref",
            [0.0, 1.0], [0.0, 1.0, 2.0],
            REF_TIMES, REF_CONC,
        )


def test_mismatched_lengths_ref_raises():
    with pytest.raises(ValueError, match="same length"):
        calculate_be_metrics(
            "Drug", "Test", "Ref",
            REF_TIMES, REF_CONC,
            [0.0, 1.0], [0.0, 1.0, 2.0],
        )


def test_too_few_test_points_raises():
    with pytest.raises(ValueError, match="at least 3"):
        calculate_be_metrics(
            "Drug", "Test", "Ref",
            [0.0, 1.0], [0.0, 0.5],
            REF_TIMES, REF_CONC,
        )


def test_too_few_ref_points_raises():
    with pytest.raises(ValueError, match="at least 3"):
        calculate_be_metrics(
            "Drug", "Test", "Ref",
            REF_TIMES, REF_CONC,
            [0.0, 1.0], [0.0, 0.5],
        )


# ---------------------------------------------------------------------------
# compare_formulation_be
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    test_formulations = [
        {"name": "FormA", "times_h": TEST_TIMES_BE, "conc_mg_L": TEST_CONC_BE},
        {"name": "FormB", "times_h": TEST_TIMES_FAIL, "conc_mg_L": TEST_CONC_FAIL},
    ]
    results = compare_formulation_be("Drug", REF_TIMES, REF_CONC, test_formulations)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_auc_ratio():
    test_formulations = [
        {"name": "Low",  "times_h": TEST_TIMES_FAIL, "conc_mg_L": TEST_CONC_FAIL},
        {"name": "High", "times_h": TEST_TIMES_BE,   "conc_mg_L": TEST_CONC_BE},
    ]
    results = compare_formulation_be("Drug", REF_TIMES, REF_CONC, test_formulations)
    ratios = [r.auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_reference_label():
    test_formulations = [
        {"name": "TestX", "times_h": TEST_TIMES_BE, "conc_mg_L": TEST_CONC_BE},
    ]
    results = compare_formulation_be("Drug", REF_TIMES, REF_CONC, test_formulations)
    assert results[0].reference_formulation == "Reference"
    assert results[0].test_formulation == "TestX"
