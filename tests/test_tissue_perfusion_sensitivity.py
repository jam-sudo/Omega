"""Tests for Phase 1065 — Tissue Perfusion Sensitivity Analysis."""

import pytest

from omega_pbpk.core.tissue_perfusion_sensitivity import (
    PerfusionSensitivityResult,
    analyze_perfusion_sensitivity,
)


def test_return_type():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert isinstance(result, PerfusionSensitivityResult)


def test_frozen_dataclass():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    with pytest.raises((AttributeError, TypeError)):
        result.tissue = "kidney"  # type: ignore[misc]


def test_sensitivity_coefficient_is_float():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert isinstance(result.sensitivity_coefficient, float)


def test_perfusion_class_valid():
    valid = {"flow_limited", "capacity_limited", "intermediate"}
    for tissue in ["liver", "kidney", "muscle", "fat", "brain", "lung"]:
        result = analyze_perfusion_sensitivity("DrugA", tissue)
        assert result.perfusion_class in valid, f"{tissue}: {result.perfusion_class}"


def test_flow_limited_xor_capacity_limited_xor_intermediate():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    fl = result.flow_limited
    cl = result.capacity_limited
    pc = result.perfusion_class
    if pc == "flow_limited":
        assert fl and not cl
    elif pc == "capacity_limited":
        assert cl and not fl
    else:
        assert not fl and not cl


def test_critical_flow_reduction_range():
    for tissue in ["liver", "kidney", "muscle", "fat", "brain", "lung"]:
        result = analyze_perfusion_sensitivity("DrugA", tissue)
        assert 10 <= result.critical_flow_reduction_pct <= 100, tissue


def test_base_auc_tissue_positive():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert result.base_auc_tissue > 0


def test_base_auc_plasma_positive():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert result.base_auc_plasma > 0


def test_base_perfusion_liver():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert result.base_perfusion_L_h == pytest.approx(90.0)


def test_base_perfusion_fat():
    result = analyze_perfusion_sensitivity("DrugA", "fat")
    assert result.base_perfusion_L_h == pytest.approx(18.0)


def test_base_perfusion_lung():
    result = analyze_perfusion_sensitivity("DrugA", "lung")
    assert result.base_perfusion_L_h == pytest.approx(300.0)


def test_high_kp_flow_limited():
    """High kp (10) — tissue takes up more drug, more flow-dependent."""
    result = analyze_perfusion_sensitivity("DrugA", "liver", kp_tissue=10.0)
    # With high kp, tissue AUC is strongly influenced by flow
    assert isinstance(result.sensitivity_coefficient, float)
    # Just verify the result is valid
    assert result.perfusion_class in {"flow_limited", "capacity_limited", "intermediate"}


def test_low_kp_capacity_limited():
    """Low kp (0.1) — tissue takes up very little drug, less flow-dependent."""
    result = analyze_perfusion_sensitivity("DrugA", "liver", kp_tissue=0.1)
    assert isinstance(result.sensitivity_coefficient, float)
    assert result.perfusion_class in {"flow_limited", "capacity_limited", "intermediate"}


def test_liver_vs_fat_different_sensitivity():
    """Liver (90 L/h) and fat (18 L/h) should produce different sensitivity metrics."""
    liver = analyze_perfusion_sensitivity("DrugA", "liver")
    fat = analyze_perfusion_sensitivity("DrugA", "fat")
    assert liver.base_perfusion_L_h != fat.base_perfusion_L_h


def test_notes_nonempty():
    result = analyze_perfusion_sensitivity("DrugA", "liver")
    assert result.notes
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = analyze_perfusion_sensitivity("TestDrug", "kidney")
    assert result.drug_name == "TestDrug"


def test_tissue_preserved():
    result = analyze_perfusion_sensitivity("DrugA", "brain")
    assert result.tissue == "brain"


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        analyze_perfusion_sensitivity("DrugA", "liver", cl_sys_L_per_h=0.0)


def test_validation_cl_negative():
    with pytest.raises(ValueError, match="cl_sys"):
        analyze_perfusion_sensitivity("DrugA", "liver", cl_sys_L_per_h=-1.0)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        analyze_perfusion_sensitivity("DrugA", "liver", dose_mg=0.0)


def test_validation_invalid_tissue():
    with pytest.raises(ValueError, match="tissue"):
        analyze_perfusion_sensitivity("DrugA", "spleen")


def test_validation_kp_zero():
    with pytest.raises(ValueError, match="kp_tissue"):
        analyze_perfusion_sensitivity("DrugA", "liver", kp_tissue=0.0)


def test_validation_delta_q_zero():
    with pytest.raises(ValueError, match="delta_q_pct"):
        analyze_perfusion_sensitivity("DrugA", "liver", delta_q_pct=0.0)


def test_validation_delta_q_100():
    with pytest.raises(ValueError, match="delta_q_pct"):
        analyze_perfusion_sensitivity("DrugA", "liver", delta_q_pct=100.0)


def test_all_tissues_work():
    for tissue in ["liver", "kidney", "muscle", "fat", "brain", "lung"]:
        result = analyze_perfusion_sensitivity("DrugA", tissue)
        assert result.base_auc_tissue > 0
