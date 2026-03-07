"""Tests for Phase 350 — Hepatic Blood Flow Effect on CL."""

import pytest

from omega_pbpk.core.hepatic_blood_flow_effect import (
    HepaticFlowEffectResult,
    simulate_hepatic_flow_effect,
)

# --- Helpers ---

VALID_KWARGS_HIGH_E = dict(
    drug_name="HighExtraction",
    dose_mg=100.0,
    cl_intrinsic_L_per_h=5000.0,  # Very high CLint -> high E (E ~ 0.97)
    fu_plasma=0.5,
    vd_L=50.0,
    t_end_h=12.0,
    dt_h=0.1,
    q_hepatic_baseline_L_per_h=90.0,
    flow_change_pct=0.0,
)

VALID_KWARGS_LOW_E = dict(
    drug_name="LowExtraction",
    dose_mg=100.0,
    cl_intrinsic_L_per_h=1.0,  # Very low CLint -> low E
    fu_plasma=0.5,
    vd_L=50.0,
    t_end_h=12.0,
    dt_h=0.1,
    q_hepatic_baseline_L_per_h=90.0,
    flow_change_pct=0.0,
)


def make_result(**kwargs):
    defaults = {**VALID_KWARGS_HIGH_E}
    defaults.update(kwargs)
    return simulate_hepatic_flow_effect(**defaults)


# --- Return type and structure ---


def test_return_type():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert isinstance(result, HepaticFlowEffectResult)


def test_array_lengths_consistent():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert len(result.times_h) == len(result.c_baseline_mg_L)
    assert len(result.times_h) == len(result.c_modified_mg_L)


def test_arrays_non_empty():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert len(result.times_h) > 1


# --- Extraction ratio ---


def test_e_baseline_in_0_1():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert 0.0 < result.e_baseline < 1.0


def test_e_modified_in_0_1():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert 0.0 < result.e_modified < 1.0


def test_high_extraction_e_gt_07():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.e_baseline > 0.7


def test_low_extraction_e_lt_03():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_LOW_E)
    assert result.e_baseline < 0.3


# --- CL values ---


def test_cl_baseline_positive():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.cl_baseline_L_per_h > 0.0


def test_cl_modified_positive():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.cl_modified_L_per_h > 0.0


# --- High extraction: CL sensitive to flow ---


def test_high_extraction_flow_reduction_reduces_cl():
    """Reducing Qh for high-E drug should reduce CL."""
    r_base = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    r_reduced = simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "flow_change_pct": -30.0})
    assert r_reduced.cl_modified_L_per_h < r_base.cl_baseline_L_per_h


# --- Low extraction: CL insensitive to flow ---


def test_low_extraction_flow_reduction_minimal_cl_change():
    """Reducing Qh for low-E drug should have minimal effect on CL."""
    r_base = simulate_hepatic_flow_effect(**VALID_KWARGS_LOW_E)
    r_reduced = simulate_hepatic_flow_effect(**{**VALID_KWARGS_LOW_E, "flow_change_pct": -50.0})
    # CL change should be < 10% for low extraction drug
    pct_change = (
        abs(r_reduced.cl_modified_L_per_h - r_base.cl_baseline_L_per_h) / r_base.cl_baseline_L_per_h
    )
    assert pct_change < 0.10


# --- AUC ratio: reduced flow + high extraction -> higher AUC ---


def test_reduced_flow_high_extraction_auc_ratio_gt_1():
    """Reduced Qh -> lower CL -> higher AUC for high-E drug."""
    result = simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "flow_change_pct": -30.0})
    assert result.auc_ratio > 1.0


def test_auc_ratio_defined():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.auc_ratio > 0.0


# --- Flow sensitivity classification ---

VALID_FLOW_CLASSES = {"flow_sensitive", "intermediate", "flow_insensitive"}


def test_flow_sensitivity_class_valid():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.flow_sensitivity_class in VALID_FLOW_CLASSES


def test_high_extraction_classified_flow_sensitive():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.flow_sensitivity_class == "flow_sensitive"


def test_low_extraction_classified_flow_insensitive():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_LOW_E)
    assert result.flow_sensitivity_class == "flow_insensitive"


# --- Clinical significance ---

VALID_CLIN_SIG = {"major", "moderate", "minor"}


def test_clinical_significance_valid():
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    assert result.clinical_significance in VALID_CLIN_SIG


def test_major_significance_large_flow_change():
    """Large flow reduction in high-E drug -> major significance."""
    result = simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "flow_change_pct": -60.0})
    assert result.clinical_significance == "major"


def test_minor_significance_no_flow_change():
    """No flow change -> auc_ratio=1 -> minor."""
    result = simulate_hepatic_flow_effect(**VALID_KWARGS_HIGH_E)
    # auc_ratio should be exactly 1 with flow_change_pct=0
    assert result.auc_ratio == pytest.approx(1.0, rel=1e-6)
    assert result.clinical_significance == "minor"


# --- Validation errors ---


def test_validation_dose_le_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "dose_mg": 0.0})


def test_validation_cl_intrinsic_le_zero():
    with pytest.raises(ValueError, match="cl_intrinsic"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "cl_intrinsic_L_per_h": 0.0})


def test_validation_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "fu_plasma": 0.0})


def test_validation_fu_gt_1():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "fu_plasma": 1.5})


def test_validation_vd_le_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "vd_L": -10.0})


def test_validation_flow_change_le_minus100():
    with pytest.raises(ValueError, match="flow_change_pct"):
        simulate_hepatic_flow_effect(**{**VALID_KWARGS_HIGH_E, "flow_change_pct": -100.0})
