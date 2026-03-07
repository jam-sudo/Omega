"""Tests for core/bone_marrow_pk.py — Phase 772."""

import math

import pytest

from omega_pbpk.core.bone_marrow_pk import (
    BoneMarrowPKResult,
    assess_hematotoxicity_risk,
    simulate_bone_marrow_pk,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_bone_marrow_pk_result():
    result = simulate_bone_marrow_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, BoneMarrowPKResult)


def test_cmax_plasma_positive():
    result = simulate_bone_marrow_pk("DrugA", dose_mg=50.0)
    assert result.cmax_plasma > 0.0


def test_cmax_bm_positive():
    result = simulate_bone_marrow_pk("DrugA", dose_mg=50.0)
    assert result.cmax_bm > 0.0


def test_auc_plasma_positive():
    result = simulate_bone_marrow_pk("DrugA", dose_mg=50.0)
    assert result.auc_plasma > 0.0


def test_auc_bm_positive():
    result = simulate_bone_marrow_pk("DrugA", dose_mg=50.0)
    assert result.auc_bm > 0.0


def test_kp_marrow_positive():
    result = simulate_bone_marrow_pk("DrugA", dose_mg=100.0, logp=2.0)
    assert result.kp_marrow > 0.0


def test_kp_marrow_bounded():
    """Kp must stay in [0.1, 5.0]."""
    r_low = simulate_bone_marrow_pk("Low", dose_mg=10.0, logp=-10.0)
    r_high = simulate_bone_marrow_pk("High", dose_mg=10.0, logp=100.0)
    assert r_low.kp_marrow >= 0.1
    assert r_high.kp_marrow <= 5.0


# ---------------------------------------------------------------------------
# Pharmacology tests
# ---------------------------------------------------------------------------


def test_high_logp_increases_kp_marrow():
    """Higher logP should produce higher marrow Kp."""
    r_low = simulate_bone_marrow_pk("Low", dose_mg=100.0, logp=0.0)
    r_high = simulate_bone_marrow_pk("High", dose_mg=100.0, logp=4.0)
    assert r_high.kp_marrow > r_low.kp_marrow


def test_high_logp_higher_bm_auc_ratio():
    """Higher logP → more bone marrow accumulation → higher BM/plasma AUC ratio."""
    r_low = simulate_bone_marrow_pk("Low", dose_mg=100.0, logp=0.0)
    r_high = simulate_bone_marrow_pk("High", dose_mg=100.0, logp=5.0)
    assert r_high.bm_to_plasma_auc_ratio > r_low.bm_to_plasma_auc_ratio


def test_hematotox_index_equals_auc_bm_times_sensitivity():
    """hematotox_index must equal auc_bm × drug_sensitivity_factor."""
    dsf = 2.5
    result = simulate_bone_marrow_pk("DrugB", dose_mg=100.0, drug_sensitivity_factor=dsf)
    assert abs(result.hematotox_index - result.auc_bm * dsf) < 1e-9


def test_linear_dose_scaling_cmax():
    """Doubling dose should double cmax_plasma (linearity)."""
    r1 = simulate_bone_marrow_pk("DrugC", dose_mg=100.0)
    r2 = simulate_bone_marrow_pk("DrugC", dose_mg=200.0)
    assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01


def test_linear_dose_scaling_auc():
    """Doubling dose should double auc_plasma."""
    r1 = simulate_bone_marrow_pk("DrugD", dose_mg=100.0)
    r2 = simulate_bone_marrow_pk("DrugD", dose_mg=200.0)
    assert abs(r2.auc_plasma / r1.auc_plasma - 2.0) < 0.05


def test_t_half_plasma_consistent_with_ke():
    """t_half_plasma_h should match ln(2) / ke."""
    cl = 6.0
    vd = 40.0
    ke = cl / vd
    expected_thalf = math.log(2) / ke
    result = simulate_bone_marrow_pk("DrugE", dose_mg=100.0, cl_L_per_h=cl, vd_L=vd)
    assert abs(result.t_half_plasma_h - expected_thalf) < 1e-9


# ---------------------------------------------------------------------------
# Hematotoxicity classification
# ---------------------------------------------------------------------------


def test_hematotox_risk_valid_values():
    result = simulate_bone_marrow_pk("DrugF", dose_mg=100.0)
    assert result.hematotox_risk in ("low", "moderate", "high")


def test_hematotox_risk_low_for_small_auc():
    """Very low dose → low AUC → 'low' hematotox risk."""
    result = simulate_bone_marrow_pk("Safe", dose_mg=1.0, vd_L=50.0, cl_L_per_h=20.0)
    assert result.hematotox_risk == "low"


def test_hematotox_risk_high_for_large_auc():
    """Large dose → high AUC → 'high' hematotox risk."""
    result = simulate_bone_marrow_pk(
        "Toxic", dose_mg=5000.0, cl_L_per_h=1.0, vd_L=10.0, t_end_h=72.0
    )
    assert result.hematotox_risk == "high"


# ---------------------------------------------------------------------------
# Time-series sanity
# ---------------------------------------------------------------------------


def test_times_h_starts_at_zero():
    result = simulate_bone_marrow_pk("DrugG", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_h_ends_near_t_end():
    result = simulate_bone_marrow_pk("DrugG", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert abs(result.times_h[-1] - 10.0) < 0.15


def test_concentration_lists_same_length():
    result = simulate_bone_marrow_pk("DrugH", dose_mg=100.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.c_bm_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_bone_marrow_pk("Bad", dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        simulate_bone_marrow_pk("Bad", dose_mg=-10.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        simulate_bone_marrow_pk("Bad", dose_mg=100.0, vd_L=0.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        simulate_bone_marrow_pk("Bad", dose_mg=100.0, cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# Risk assessment function
# ---------------------------------------------------------------------------


def test_assess_returns_dict_with_risk_level():
    result = simulate_bone_marrow_pk("DrugI", dose_mg=100.0)
    assessment = assess_hematotoxicity_risk(result)
    assert isinstance(assessment, dict)
    assert "risk_level" in assessment


def test_assess_risk_level_valid():
    result = simulate_bone_marrow_pk("DrugI", dose_mg=100.0)
    assessment = assess_hematotoxicity_risk(result)
    assert assessment["risk_level"] in ("low", "moderate", "high")


def test_assess_contains_auc_bm():
    result = simulate_bone_marrow_pk("DrugJ", dose_mg=100.0)
    assessment = assess_hematotoxicity_risk(result)
    assert "auc_bm" in assessment
    assert abs(assessment["auc_bm"] - result.auc_bm) < 1e-9


def test_notes_nonempty():
    result = simulate_bone_marrow_pk("DrugK", dose_mg=100.0)
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = simulate_bone_marrow_pk("CyclophosphamideAnalogue", dose_mg=200.0)
    assert result.drug_name == "CyclophosphamideAnalogue"
