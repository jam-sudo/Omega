"""Tests for Phase 490 — Cytokine-Mediated CYP Suppression."""

import pytest

from omega_pbpk.clinical.cytokine_cyp_suppression import (
    CytokineCYPSuppressionResult,
    predict_cytokine_cyp_suppression,
    screen_cyp_states,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
FM3A4 = 0.5
FM1A2 = 0.1
FM2C9 = 0.2
FM2D6 = 0.1  # sum = 0.9, non-CYP = 0.1


def normal_result():
    return predict_cytokine_cyp_suppression(DRUG, "normal", FM3A4, FM1A2, FM2C9, FM2D6)


def sepsis_result():
    return predict_cytokine_cyp_suppression(DRUG, "sepsis", FM3A4, FM1A2, FM2C9, FM2D6)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = normal_result()
    assert isinstance(result, CytokineCYPSuppressionResult)


def test_result_is_frozen():
    result = normal_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Normal state: auc_ratio = 1.0
# ---------------------------------------------------------------------------


def test_normal_auc_ratio_is_one():
    result = normal_result()
    assert abs(result.auc_ratio - 1.0) < 1e-6


def test_normal_cl_ratio_is_one():
    result = normal_result()
    assert abs(result.cl_ratio - 1.0) < 1e-6


def test_normal_dose_adjustment_factor_is_one():
    result = normal_result()
    assert abs(result.dose_adjustment_factor - 1.0) < 1e-6


def test_normal_clinical_significance_none():
    result = normal_result()
    assert result.clinical_significance == "none"


# ---------------------------------------------------------------------------
# Sepsis → highest AUC increase
# ---------------------------------------------------------------------------


def test_sepsis_auc_ratio_highest():
    states = ["IL-6_high", "IL-6_moderate", "sepsis", "COVID-19_severe"]
    results = {
        s: predict_cytokine_cyp_suppression(DRUG, s, FM3A4, FM1A2, FM2C9, FM2D6) for s in states
    }
    sepsis_auc = results["sepsis"].auc_ratio
    for s, r in results.items():
        if s != "sepsis":
            assert sepsis_auc >= r.auc_ratio


def test_sepsis_cl_ratio_lowest():
    sepsis = sepsis_result()
    il6 = predict_cytokine_cyp_suppression(DRUG, "IL-6_high", FM3A4, FM1A2, FM2C9, FM2D6)
    assert sepsis.cl_ratio < il6.cl_ratio


# ---------------------------------------------------------------------------
# fm_cyp3a4=0: IL-6 has less impact
# ---------------------------------------------------------------------------


def test_zero_fm_cyp3a4_reduces_il6_impact():
    # With all fm on CYP3A4, IL-6 has large impact
    r_high_3a4 = predict_cytokine_cyp_suppression(DRUG, "IL-6_high", 0.9, 0.0, 0.0, 0.0)
    # With zero fm_cyp3a4, IL-6 high has less CYP3A4 suppression impact
    r_no_3a4 = predict_cytokine_cyp_suppression(DRUG, "IL-6_high", 0.0, 0.0, 0.0, 0.0)
    # No CYP fraction → cl_ratio = 1.0, no suppression
    assert r_no_3a4.cl_ratio >= r_high_3a4.cl_ratio


def test_zero_fm_cyp3a4_no_suppression_from_cyp3a4():
    result = predict_cytokine_cyp_suppression(DRUG, "IL-6_high", 0.0, 0.0, 0.0, 0.0)
    # No CYP involvement → cl_ratio = 1.0
    assert abs(result.cl_ratio - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# auc_ratio >= 1.0 for all inflammatory states
# ---------------------------------------------------------------------------


def test_auc_ratio_ge_1_all_states():
    states = ["IL-6_high", "IL-6_moderate", "sepsis", "COVID-19_severe", "normal"]
    for state in states:
        result = predict_cytokine_cyp_suppression(DRUG, state, FM3A4, FM1A2, FM2C9, FM2D6)
        assert result.auc_ratio >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# dose_adjustment_factor = 1 / auc_ratio (exact)
# ---------------------------------------------------------------------------


def test_dose_adjustment_factor_exact():
    for state in ["IL-6_high", "IL-6_moderate", "sepsis", "COVID-19_severe", "normal"]:
        result = predict_cytokine_cyp_suppression(DRUG, state, FM3A4, FM1A2, FM2C9, FM2D6)
        expected = 1.0 / result.auc_ratio
        assert abs(result.dose_adjustment_factor - expected) < 1e-5


# ---------------------------------------------------------------------------
# CYP factors stored correctly
# ---------------------------------------------------------------------------


def test_sepsis_cyp3a4_factor():
    result = sepsis_result()
    assert abs(result.cyp3a4_factor - 0.2) < 1e-9


def test_il6_high_cyp3a4_factor():
    result = predict_cytokine_cyp_suppression(DRUG, "IL-6_high", FM3A4, FM1A2, FM2C9, FM2D6)
    assert abs(result.cyp3a4_factor - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# cl_ratio computation correctness
# ---------------------------------------------------------------------------


def test_cl_ratio_manual_calculation():
    # sepsis: 3A4=0.2, 1A2=0.3, 2C9=0.5, 2D6=0.8
    # fm: 0.5, 0.1, 0.2, 0.1; non-CYP = 0.1
    expected_cl_ratio = 0.5 * 0.2 + 0.1 * 0.3 + 0.2 * 0.5 + 0.1 * 0.8 + 0.1 * 1.0
    result = sepsis_result()
    assert abs(result.cl_ratio - expected_cl_ratio) < 1e-5


# ---------------------------------------------------------------------------
# screen_cyp_states
# ---------------------------------------------------------------------------


def test_screen_returns_5():
    results = screen_cyp_states(DRUG, FM3A4, FM1A2, FM2C9, FM2D6)
    assert len(results) == 5


def test_screen_sorted_by_auc_ratio_descending():
    results = screen_cyp_states(DRUG, FM3A4, FM1A2, FM2C9, FM2D6)
    for i in range(len(results) - 1):
        assert results[i].auc_ratio >= results[i + 1].auc_ratio


def test_screen_first_is_sepsis():
    results = screen_cyp_states(DRUG, FM3A4, FM1A2, FM2C9, FM2D6)
    assert results[0].cyp_state == "sepsis"


def test_screen_last_is_normal():
    results = screen_cyp_states(DRUG, FM3A4, FM1A2, FM2C9, FM2D6)
    assert results[-1].cyp_state == "normal"


def test_screen_all_result_types():
    results = screen_cyp_states(DRUG, FM3A4, FM1A2, FM2C9, FM2D6)
    for r in results:
        assert isinstance(r, CytokineCYPSuppressionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_cyp_state_raises():
    with pytest.raises(ValueError, match="cyp_state"):
        predict_cytokine_cyp_suppression(DRUG, "flu", FM3A4, FM1A2, FM2C9, FM2D6)


def test_negative_fm_raises():
    with pytest.raises(ValueError, match="fm values"):
        predict_cytokine_cyp_suppression(DRUG, "normal", -0.1, FM1A2, FM2C9, FM2D6)


def test_fm_sum_gt_1_raises():
    with pytest.raises(ValueError, match="Sum of fm"):
        predict_cytokine_cyp_suppression(DRUG, "normal", 0.5, 0.3, 0.3, 0.2)


def test_screen_invalid_fm_raises():
    with pytest.raises(ValueError):
        screen_cyp_states(DRUG, 0.5, 0.3, 0.3, 0.2)
