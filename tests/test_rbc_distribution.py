"""Tests for core/rbc_distribution.py — Phase 888."""

from __future__ import annotations

import pytest

from omega_pbpk.core.rbc_distribution import (
    RBCDistributionResult,
    predict_rbc_distribution,
    screen_rbc_partitioning,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_basic_result_type():
    result = predict_rbc_distribution("DrugA")
    assert isinstance(result, RBCDistributionResult)
    assert result.drug_name == "DrugA"


def test_default_hematocrit():
    result = predict_rbc_distribution("DrugB")
    assert result.hematocrit == 0.45


def test_bp_ratio_formula():
    result = predict_rbc_distribution("DrugC", logp=2.0, fup=0.5, hematocrit=0.45)
    hct = 0.45
    kp_rbc = result.kp_rbc
    expected_bp = (1.0 - hct) + hct * kp_rbc
    assert abs(result.blood_plasma_ratio - expected_bp) < 1e-12


def test_kp_rbc_formula():
    # kp_rbc = 1.0 + (logp / 5.0) * (1.0 - fup) * 3.0
    logp, fup = 3.0, 0.2
    result = predict_rbc_distribution("DrugD", logp=logp, fup=fup)
    expected = 1.0 + (logp / 5.0) * (1.0 - fup) * 3.0
    assert abs(result.kp_rbc - expected) < 1e-12


def test_kp_rbc_clamped_min():
    # Negative logp, high fup → kp_rbc could go below 1 but clamped at 0.1
    result = predict_rbc_distribution("DrugE", logp=-10.0, fup=0.99)
    assert result.kp_rbc >= 0.1


def test_kp_rbc_clamped_max():
    # Very high logp, very low fup → kp_rbc clamped at 20.0
    result = predict_rbc_distribution("DrugF", logp=50.0, fup=0.01)
    assert result.kp_rbc <= 20.0


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_vd_blood_formula():
    result = predict_rbc_distribution("DrugG", vd_plasma_L=50.0)
    expected = 50.0 * result.blood_plasma_ratio
    assert abs(result.vd_blood_L - expected) < 1e-10


def test_cl_blood_formula():
    result = predict_rbc_distribution("DrugH", cl_plasma_L_per_h=10.0)
    expected = 10.0 / result.blood_plasma_ratio
    assert abs(result.cl_blood_L_per_h - expected) < 1e-10


def test_rbc_bound_fraction_formula():
    result = predict_rbc_distribution("DrugI", hematocrit=0.45)
    hct = 0.45
    kp = result.kp_rbc
    bp = result.blood_plasma_ratio
    expected = hct * kp / bp
    assert abs(result.rbc_bound_fraction - expected) < 1e-12


def test_fu_blood_formula():
    result = predict_rbc_distribution("DrugJ", fup=0.3)
    expected = 0.3 / result.blood_plasma_ratio
    assert abs(result.fu_blood - expected) < 1e-12


def test_t_half_formula():
    result = predict_rbc_distribution("DrugK", vd_plasma_L=60.0, cl_plasma_L_per_h=12.0)
    expected = 0.693 * result.vd_blood_L / result.cl_blood_L_per_h
    assert abs(result.rbc_t_half_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Interpretation strings
# ---------------------------------------------------------------------------


def test_high_rbc_partitioning_interpretation():
    # logp=8, fup=0.01 → kp_rbc well above 5 (pre-clamp: 1 + (8/5)*0.99*3 = ~5.75)
    result = predict_rbc_distribution("DrugL", logp=8.0, fup=0.01)
    assert result.kp_rbc > 5
    assert "High RBC partitioning" in result.interpretation


def test_moderate_rbc_partitioning_interpretation():
    # kp_rbc between 2 and 5
    # 1 + (logp/5)*0.7*3 = 2.5 → logp≈2.38
    result = predict_rbc_distribution("DrugM", logp=4.0, fup=0.3)
    if 2 < result.kp_rbc <= 5:
        assert "Moderate RBC partitioning" in result.interpretation


def test_minimal_rbc_partitioning_interpretation():
    # Low logp, high fup → kp_rbc near 1
    result = predict_rbc_distribution("DrugN", logp=0.5, fup=0.9)
    assert result.kp_rbc <= 2
    assert "Minimal RBC partitioning" in result.interpretation


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        predict_rbc_distribution("DrugO", fup=0.0)


def test_invalid_fup_above_1():
    with pytest.raises(ValueError, match="fup"):
        predict_rbc_distribution("DrugP", fup=1.5)


def test_invalid_hematocrit_zero():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_distribution("DrugQ", hematocrit=0.0)


def test_invalid_hematocrit_one():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_distribution("DrugR", hematocrit=1.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        predict_rbc_distribution("DrugS", vd_plasma_L=0.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        predict_rbc_distribution("DrugT", cl_plasma_L_per_h=-1.0)


# ---------------------------------------------------------------------------
# screen_rbc_partitioning
# ---------------------------------------------------------------------------


def test_screen_returns_sorted():
    candidates = [
        {"name": "Low", "logp": 0.5, "fup": 0.9},
        {"name": "High", "logp": 5.0, "fup": 0.1},
        {"name": "Mid", "logp": 2.5, "fup": 0.4},
    ]
    results = screen_rbc_partitioning(candidates)
    bp_ratios = [r.blood_plasma_ratio for r in results]
    assert bp_ratios == sorted(bp_ratios, reverse=True)


def test_screen_uses_default_fup():
    candidates = [{"name": "DrugU", "logp": 2.0}]
    results = screen_rbc_partitioning(candidates)
    assert len(results) == 1
    # Default fup=0.5 should be used without error
    assert results[0].drug_name == "DrugU"


def test_screen_uses_default_hematocrit():
    candidates = [{"name": "DrugV", "logp": 2.0, "fup": 0.5}]
    results = screen_rbc_partitioning(candidates)
    assert results[0].hematocrit == 0.45


def test_screen_custom_hematocrit():
    candidates = [{"name": "DrugW", "logp": 2.0, "fup": 0.5, "hematocrit": 0.40}]
    results = screen_rbc_partitioning(candidates)
    assert results[0].hematocrit == 0.40


def test_screen_empty_list():
    results = screen_rbc_partitioning([])
    assert results == []


def test_screen_multiple_candidates_correct_names():
    candidates = [
        {"name": "Alpha", "logp": 1.0},
        {"name": "Beta", "logp": 3.0},
        {"name": "Gamma", "logp": 5.0},
    ]
    results = screen_rbc_partitioning(candidates)
    names = [r.drug_name for r in results]
    assert set(names) == {"Alpha", "Beta", "Gamma"}
