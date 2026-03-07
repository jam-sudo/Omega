"""Tests for hepatic_disease_pk module — Phase 703."""

import pytest

from omega_pbpk.clinical.hepatic_disease_pk import (
    HepaticDiseaseAdjResult,
    child_pugh_score,
    estimate_hepatic_pk_adjustment,
    simulate_hepatic_disease_pk_profiles,
)

# ── child_pugh_score tests ────────────────────────────────────────────────────


def test_child_pugh_score_class_a_minimal():
    """Minimal values → Class A (score = 5)."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=4.0,
        pt_seconds_prolonged=2.0,
        ascites="none",
        encephalopathy="none",
    )
    assert score == 5
    assert cp_class == "A"


def test_child_pugh_score_class_a_boundary():
    """Score of 6 → Class A."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=2.5,  # 2 pts
        albumin_g_dL=4.0,  # 1 pt
        pt_seconds_prolonged=2.0,  # 1 pt
        ascites="none",  # 1 pt
        encephalopathy="none",  # 1 pt
    )
    assert score == 6
    assert cp_class == "A"


def test_child_pugh_score_class_b():
    """Score of 7-9 → Class B."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=2.5,  # 2 pts
        albumin_g_dL=3.0,  # 2 pts
        pt_seconds_prolonged=5.0,  # 2 pts
        ascites="none",  # 1 pt
        encephalopathy="none",  # 1 pt
    )
    assert score == 8
    assert cp_class == "B"


def test_child_pugh_score_class_c_worst():
    """Worst values → Class C with score 15."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=4.0,  # 3 pts
        albumin_g_dL=2.5,  # 3 pts
        pt_seconds_prolonged=7.0,  # 3 pts
        ascites="severe",  # 3 pts
        encephalopathy="grade 3-4",  # 3 pts
    )
    assert score == 15
    assert cp_class == "C"


def test_child_pugh_score_mild_ascites():
    """Mild ascites gives 2 points."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=4.0,
        pt_seconds_prolonged=2.0,
        ascites="mild",
        encephalopathy="none",
    )
    assert score == 6
    assert cp_class == "A"


def test_child_pugh_score_moderate_ascites():
    """Moderate ascites gives 3 points."""
    score, cp_class = child_pugh_score(
        bilirubin_mg_dL=1.5,
        albumin_g_dL=4.0,
        pt_seconds_prolonged=2.0,
        ascites="moderate",
        encephalopathy="none",
    )
    assert score == 7
    assert cp_class == "B"


# ── estimate_hepatic_pk_adjustment tests ─────────────────────────────────────


def test_estimate_returns_result_dataclass():
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "A")
    assert isinstance(result, HepaticDiseaseAdjResult)


def test_class_a_cl_reduction():
    """Class A: CL adjusted ≈ 0.75 × normal."""
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "A", hepatic_extraction_ratio=0.3)
    assert abs(result.cl_adjusted_L_per_h - 7.5) < 0.01


def test_class_b_cl_reduction():
    """Class B: CL adjusted ≈ 0.5 × normal."""
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "B", hepatic_extraction_ratio=0.3)
    assert abs(result.cl_adjusted_L_per_h - 5.0) < 0.01


def test_class_c_cl_reduction():
    """Class C: CL adjusted ≈ 0.25 × normal."""
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "C", hepatic_extraction_ratio=0.3)
    assert abs(result.cl_adjusted_L_per_h - 2.5) < 0.01


def test_t_half_adjusted_longer():
    """Reduced CL → longer half-life."""
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "B", hepatic_extraction_ratio=0.3)
    assert result.t_half_adjusted_h > result.t_half_normal_h


def test_dose_adjustment_factor_le_1():
    """dose_adjustment_factor ≤ 1.0 for all CP classes."""
    for cp_class in ["A", "B", "C"]:
        result = estimate_hepatic_pk_adjustment(10.0, 100.0, cp_class)
        assert result.dose_adjustment_factor <= 1.0


def test_recommended_dose_fraction_is_string():
    result = estimate_hepatic_pk_adjustment(10.0, 100.0, "B")
    assert isinstance(result.recommended_dose_fraction, str)
    assert "%" in result.recommended_dose_fraction


def test_fup_adjusted_greater_than_normal():
    """Hypoalbuminemia increases free fraction."""
    for cp_class in ["A", "B", "C"]:
        result = estimate_hepatic_pk_adjustment(10.0, 100.0, cp_class, fu_plasma=0.1)
        assert result.fup_adjusted > result.fup_normal


def test_high_extraction_drug_greater_reduction():
    """High-extraction drug has greater CL reduction."""
    result_low = estimate_hepatic_pk_adjustment(10.0, 100.0, "B", hepatic_extraction_ratio=0.3)
    result_high = estimate_hepatic_pk_adjustment(10.0, 100.0, "B", hepatic_extraction_ratio=0.9)
    assert result_high.cl_adjusted_L_per_h < result_low.cl_adjusted_L_per_h


def test_invalid_cp_class_raises():
    """Invalid child_pugh_class raises ValueError."""
    with pytest.raises(ValueError):
        estimate_hepatic_pk_adjustment(10.0, 100.0, "D")


def test_invalid_er_raises():
    with pytest.raises(ValueError):
        estimate_hepatic_pk_adjustment(10.0, 100.0, "A", hepatic_extraction_ratio=1.5)


def test_invalid_fup_raises():
    with pytest.raises(ValueError):
        estimate_hepatic_pk_adjustment(10.0, 100.0, "A", fu_plasma=0.0)


# ── simulate_hepatic_disease_pk_profiles tests ───────────────────────────────


def test_simulate_returns_dict_with_all_keys():
    result = simulate_hepatic_disease_pk_profiles("DrugX", 100.0, 5.0, 50.0)
    for key in ["normal", "A", "B", "C"]:
        assert key in result


def test_simulate_normal_cl_greater_than_class_c():
    """Normal CL > Class C adjusted CL → normal AUC < Class C AUC."""
    result = simulate_hepatic_disease_pk_profiles("DrugX", 100.0, 5.0, 50.0)
    assert result["normal"]["auc"] < result["C"]["auc"]


def test_simulate_each_profile_has_required_keys():
    result = simulate_hepatic_disease_pk_profiles("DrugX", 100.0, 5.0, 50.0)
    for key in ["normal", "A", "B", "C"]:
        profile = result[key]
        assert "times_h" in profile
        assert "c_plasma_mg_L" in profile
        assert "cmax" in profile
        assert "auc" in profile
        assert "t_half" in profile


def test_simulate_cmax_positive():
    result = simulate_hepatic_disease_pk_profiles("DrugX", 100.0, 5.0, 50.0)
    for key in ["normal", "A", "B", "C"]:
        assert result[key]["cmax"] > 0


def test_simulate_custom_classes():
    """Specifying child_pugh_classes=['A', 'B'] returns only those + normal."""
    result = simulate_hepatic_disease_pk_profiles(
        "DrugX", 100.0, 5.0, 50.0, child_pugh_classes=["A", "B"]
    )
    assert "normal" in result
    assert "A" in result
    assert "B" in result
    assert "C" not in result


def test_simulate_auc_increases_with_severity():
    """AUC: normal < A < B < C (increasing with disease severity)."""
    result = simulate_hepatic_disease_pk_profiles("DrugX", 100.0, 5.0, 50.0)
    assert result["normal"]["auc"] < result["A"]["auc"]
    assert result["A"]["auc"] < result["B"]["auc"]
    assert result["B"]["auc"] < result["C"]["auc"]
