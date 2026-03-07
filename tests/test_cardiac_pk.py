"""Tests for core/cardiac_pk.py — Phase 771."""

import pytest

from omega_pbpk.core.cardiac_pk import (
    CardiacPKResult,
    assess_cardiac_concentration_risk,
    simulate_cardiac_pk,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_cardiac_pk_result():
    result = simulate_cardiac_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, CardiacPKResult)


def test_cmax_plasma_positive():
    result = simulate_cardiac_pk("DrugA", dose_mg=50.0)
    assert result.cmax_plasma > 0.0


def test_cmax_cardiac_positive():
    result = simulate_cardiac_pk("DrugA", dose_mg=50.0)
    assert result.cmax_cardiac > 0.0


def test_auc_plasma_positive():
    result = simulate_cardiac_pk("DrugA", dose_mg=50.0)
    assert result.auc_plasma > 0.0


def test_auc_cardiac_positive():
    result = simulate_cardiac_pk("DrugA", dose_mg=50.0)
    assert result.auc_cardiac > 0.0


def test_kp_cardiac_positive():
    result = simulate_cardiac_pk("DrugA", dose_mg=100.0, logp=2.0)
    assert result.kp_cardiac > 0.0


def test_kp_cardiac_bounded():
    """Kp must stay in [0.1, 10.0]."""
    result_low = simulate_cardiac_pk("Low", dose_mg=10.0, logp=-10.0)
    result_high = simulate_cardiac_pk("High", dose_mg=10.0, logp=100.0)
    assert result_low.kp_cardiac >= 0.1
    assert result_high.kp_cardiac <= 10.0


# ---------------------------------------------------------------------------
# Pharmacology tests
# ---------------------------------------------------------------------------


def test_high_logp_increases_kp_cardiac():
    """Higher logP should produce higher cardiac Kp."""
    r_low = simulate_cardiac_pk("Low", dose_mg=100.0, logp=0.0)
    r_high = simulate_cardiac_pk("High", dose_mg=100.0, logp=4.0)
    assert r_high.kp_cardiac > r_low.kp_cardiac


def test_high_logp_higher_cardiac_auc_ratio():
    """Higher logP → more cardiac accumulation → higher AUC ratio."""
    r_low = simulate_cardiac_pk("Low", dose_mg=100.0, logp=0.0)
    r_high = simulate_cardiac_pk("High", dose_mg=100.0, logp=5.0)
    assert r_high.cardiac_to_plasma_auc_ratio > r_low.cardiac_to_plasma_auc_ratio


def test_basic_charge_reduces_kp():
    """Positive charge should reduce cardiac Kp vs neutral."""
    r_neutral = simulate_cardiac_pk("Neutral", dose_mg=100.0, logp=2.0, charge_at_ph74=0.0)
    r_basic = simulate_cardiac_pk("Basic", dose_mg=100.0, logp=2.0, charge_at_ph74=1.0)
    assert r_neutral.kp_cardiac > r_basic.kp_cardiac


def test_linear_dose_scaling_cmax():
    """Doubling dose should double cmax_plasma (linearity)."""
    r1 = simulate_cardiac_pk("DrugB", dose_mg=100.0)
    r2 = simulate_cardiac_pk("DrugB", dose_mg=200.0)
    assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01


def test_linear_dose_scaling_auc():
    """Doubling dose should double auc_plasma."""
    r1 = simulate_cardiac_pk("DrugC", dose_mg=100.0)
    r2 = simulate_cardiac_pk("DrugC", dose_mg=200.0)
    assert abs(r2.auc_plasma / r1.auc_plasma - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Time-series sanity
# ---------------------------------------------------------------------------


def test_times_h_starts_at_zero():
    result = simulate_cardiac_pk("DrugD", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_h_ends_near_t_end():
    result = simulate_cardiac_pk("DrugD", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert abs(result.times_h[-1] - 10.0) < 0.15


def test_concentration_lists_same_length():
    result = simulate_cardiac_pk("DrugE", dose_mg=100.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.c_cardiac_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_cardiac_pk("Bad", dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        simulate_cardiac_pk("Bad", dose_mg=-10.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        simulate_cardiac_pk("Bad", dose_mg=100.0, vd_plasma_L=0.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        simulate_cardiac_pk("Bad", dose_mg=100.0, cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# Risk assessment tests
# ---------------------------------------------------------------------------


def test_assess_returns_dict_with_risk_level():
    result = simulate_cardiac_pk("DrugF", dose_mg=100.0)
    assessment = assess_cardiac_concentration_risk(result)
    assert isinstance(assessment, dict)
    assert "risk_level" in assessment


def test_risk_level_valid_values():
    result = simulate_cardiac_pk("DrugF", dose_mg=100.0)
    assessment = assess_cardiac_concentration_risk(result)
    assert assessment["risk_level"] in ("low", "moderate", "high")


def test_risk_low_when_ratio_small():
    """Very low logP → minimal cardiac accumulation → 'low' risk."""
    result = simulate_cardiac_pk("Safe", dose_mg=50.0, logp=-3.0, charge_at_ph74=1.0)
    assessment = assess_cardiac_concentration_risk(result)
    assert assessment["risk_level"] == "low"


def test_notes_nonempty():
    result = simulate_cardiac_pk("DrugG", dose_mg=100.0)
    assert len(result.notes) > 0


def test_cardiac_safety_margin_positive():
    result = simulate_cardiac_pk("DrugH", dose_mg=10.0)
    assert result.cardiac_safety_margin > 0.0


def test_drug_name_preserved():
    result = simulate_cardiac_pk("AmiodaroneAnalogue", dose_mg=100.0)
    assert result.drug_name == "AmiodaroneAnalogue"
