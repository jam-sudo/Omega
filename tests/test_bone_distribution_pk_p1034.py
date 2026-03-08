"""Tests for Phase 1034 — Drug Bone Distribution PK (bone_distribution_pk_p1034.py)."""

import pytest

from omega_pbpk.core.bone_distribution_pk_p1034 import (
    BoneDistributionResult,
    simulate_bone_distribution,
)


def _make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        drug_class="antibiotic",
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
        q_cortical_L_h=0.3,
        q_trabecular_L_h=0.6,
        v_cortical_L=1.0,
        v_trabecular_L=0.5,
        kp_cortical=0.3,
        kp_trabecular=0.8,
        t_end_h=72.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_bone_distribution(**defaults)


# --- Return type ---


def test_return_type():
    result = _make_result()
    assert isinstance(result, BoneDistributionResult)


# --- Initial conditions ---


def test_c_plasma_initial():
    """c_plasma[0] should be dose/vd."""
    result = _make_result(dose_mg=100.0, vd_sys_L=30.0)
    expected = 100.0 / 30.0
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-6


def test_c_cortical_initial_zero():
    result = _make_result()
    assert result.c_cortical_mg_L[0] == 0.0


def test_c_trabecular_initial_zero():
    result = _make_result()
    assert result.c_trabecular_mg_L[0] == 0.0


# --- Peak in bone > 0 ---


def test_c_cortical_has_positive_peak():
    result = _make_result()
    assert max(result.c_cortical_mg_L) > 0


def test_c_trabecular_has_positive_peak():
    result = _make_result()
    assert max(result.c_trabecular_mg_L) > 0


# --- AUCs ---


def test_auc_plasma_positive():
    result = _make_result()
    assert result.auc_plasma > 0


def test_auc_cortical_positive():
    result = _make_result()
    assert result.auc_cortical > 0


def test_auc_trabecular_positive():
    result = _make_result()
    assert result.auc_trabecular > 0


# --- Drug class comparison ---


def test_bisphosphonate_higher_cortical_ratio_than_antibiotic():
    """Bisphosphonates have 10x kp_cortical multiplier, so much higher cortical AUC ratio."""
    bisphos = _make_result(drug_class="bisphosphonate")
    antibiotic = _make_result(drug_class="antibiotic")
    assert bisphos.cortical_to_plasma_auc_ratio > antibiotic.cortical_to_plasma_auc_ratio


def test_bone_affinity_bisphosphonate_is_high():
    """Bisphosphonate with elevated kp values gives 'high' bone affinity.
    The 10x kp_cortical and 5x kp_trabecular multipliers push avg AUC ratio > 2.0."""
    result = _make_result(drug_class="bisphosphonate", kp_cortical=1.0, kp_trabecular=2.0)
    assert result.bone_affinity == "high"


def test_bone_affinity_general_is_low():
    """General class has 0.5x multiplier on kp → low bone affinity."""
    result = _make_result(
        drug_class="general",
        kp_cortical=0.3,
        kp_trabecular=0.8,
    )
    assert result.bone_affinity == "low"


# --- Trabecular vs cortical ratio ---


def test_trabecular_ratio_exceeds_cortical_ratio():
    """Trabecular has higher kp (0.8 vs 0.3), so trabecular AUC ratio > cortical."""
    result = _make_result(
        drug_class="antibiotic",
        kp_cortical=0.3,
        kp_trabecular=0.8,
    )
    assert result.trabecular_to_plasma_auc_ratio > result.cortical_to_plasma_auc_ratio


# --- Cmax and tmax ---


def test_cmax_cortical_positive():
    result = _make_result()
    assert result.cmax_cortical > 0


def test_tmax_cortical_positive():
    result = _make_result()
    assert result.tmax_cortical_h > 0


# --- Notes ---


def test_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


# --- List lengths consistent ---


def test_list_lengths_consistent():
    result = _make_result(t_end_h=10.0, dt_h=0.5)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_cortical_mg_L) == n
    assert len(result.c_trabecular_mg_L) == n


# --- Validation ---


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _make_result(dose_mg=0.0)


def test_validation_invalid_drug_class_raises():
    with pytest.raises(ValueError, match="drug_class"):
        _make_result(drug_class="unknown")


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        _make_result(cl_sys_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        _make_result(vd_sys_L=0.0)


def test_validation_kp_cortical_zero_raises():
    with pytest.raises(ValueError, match="kp_cortical"):
        _make_result(kp_cortical=0.0)
