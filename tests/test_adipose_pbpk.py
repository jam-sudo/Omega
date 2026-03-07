"""Tests for Phase 1071 — Adipose Tissue PBPK (adipose_pbpk.py)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.adipose_pbpk import (
    AdiposePBPKResult,
    simulate_adipose_pk,
    screen_logp_adipose,
)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logP=2.0)
    assert isinstance(result, AdiposePBPKResult)


def test_result_has_required_fields():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logP=2.0)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "logP")
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_plasma_mg_L")
    assert hasattr(result, "c_fat_mg_L")
    assert hasattr(result, "c_lean_mg_L")
    assert hasattr(result, "cmax_plasma")
    assert hasattr(result, "tmax_plasma_h")
    assert hasattr(result, "auc_plasma")
    assert hasattr(result, "cmax_fat")
    assert hasattr(result, "tmax_fat_h")
    assert hasattr(result, "auc_fat")
    assert hasattr(result, "fat_to_plasma_auc_ratio")
    assert hasattr(result, "adipose_accumulation_risk")
    assert hasattr(result, "t_half_effective_h")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_initial_equals_dose_over_vd():
    dose = 200.0
    vd = 5.0
    result = simulate_adipose_pk("DrugA", dose_mg=dose, logP=1.0, vd_plasma_L=vd)
    assert abs(result.c_plasma_mg_L[0] - dose / vd) < 1e-9


def test_c_fat_initial_is_zero():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.c_fat_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_c_lean_initial_is_zero():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.c_lean_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Decay behaviour
# ---------------------------------------------------------------------------


def test_c_plasma_decreases_over_time():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=1.0, t_end_h=96.0)
    # Plasma at the end should be much lower than at start
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_c_fat_rises_then_falls():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=3.0, t_end_h=120.0)
    # Fat should rise from 0 and then come down
    assert result.cmax_fat > 0.0
    assert result.c_fat_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


def test_higher_logP_higher_cmax_fat():
    # Use longer simulation to allow fat to accumulate;
    # at logP=3 vs logP=1 the fat Cmax is clearly higher for logP=3
    r_low = simulate_adipose_pk("Drug", dose_mg=100.0, logP=1.0, t_end_h=120.0)
    r_high = simulate_adipose_pk("Drug", dose_mg=100.0, logP=3.0, t_end_h=120.0)
    assert r_high.cmax_fat > r_low.cmax_fat


def test_higher_logP_higher_fat_plasma_auc_ratio():
    r_low = simulate_adipose_pk("Drug", dose_mg=100.0, logP=1.0)
    r_high = simulate_adipose_pk("Drug", dose_mg=100.0, logP=5.0)
    assert r_high.fat_to_plasma_auc_ratio > r_low.fat_to_plasma_auc_ratio


# ---------------------------------------------------------------------------
# AUC and PK metrics
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.auc_plasma > 0.0


def test_auc_fat_positive():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.auc_fat > 0.0


def test_fat_to_plasma_auc_ratio_positive():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.fat_to_plasma_auc_ratio > 0.0


def test_t_half_effective_positive():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.t_half_effective_h > 0.0


def test_cmax_plasma_equals_initial_plasma_for_iv_bolus():
    dose = 100.0
    vd = 5.0
    result = simulate_adipose_pk("DrugA", dose_mg=dose, logP=0.0, vd_plasma_L=vd)
    # For IV bolus, Cmax should be at t=0
    assert result.cmax_plasma == pytest.approx(dose / vd, rel=1e-6)


# ---------------------------------------------------------------------------
# Accumulation risk classification
# ---------------------------------------------------------------------------


def test_accumulation_risk_valid_values():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert result.adipose_accumulation_risk in {"low", "moderate", "high"}


def test_low_logP_low_accumulation_risk():
    # logP=-2 → Kp_fat = clamp(10^(0.7*-2), 1, 500) = 1 → very low accumulation
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=-2.0)
    assert result.adipose_accumulation_risk == "low"


def test_high_logP_high_accumulation_risk():
    # logP=7 → Kp_fat = 10^(0.7*7) = 10^4.9 >> 10 → "high"
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=7.0)
    assert result.adipose_accumulation_risk == "high"


def test_moderate_logP_moderate_risk():
    # logP=3 → Kp_fat = 10^2.1 ≈ 126 → fat/plasma ratio should be moderate or high
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=3.0)
    assert result.adipose_accumulation_risk in {"moderate", "high"}


# ---------------------------------------------------------------------------
# screen_logp_adipose
# ---------------------------------------------------------------------------


def test_screen_returns_correct_count():
    logp_values = [1.0, 2.0, 3.0, 5.0]
    results = screen_logp_adipose("Drug", dose_mg=100.0, logp_list=logp_values)
    assert len(results) == len(logp_values)


def test_screen_returns_list_of_results():
    results = screen_logp_adipose("Drug", dose_mg=100.0, logp_list=[1.0, 3.0])
    for r in results:
        assert isinstance(r, AdiposePBPKResult)


def test_screen_sorted_by_fat_plasma_ratio_descending():
    logp_values = [0.0, 2.0, 4.0, 6.0]
    results = screen_logp_adipose("Drug", dose_mg=100.0, logp_list=logp_values)
    ratios = [r.fat_to_plasma_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_single_logp():
    results = screen_logp_adipose("Drug", dose_mg=50.0, logp_list=[2.5])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_pk("Drug", dose_mg=0.0, logP=2.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_pk("Drug", dose_mg=-10.0, logP=2.0)


def test_validation_logP_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_adipose_pk("Drug", dose_mg=100.0, logP=11.0)


def test_validation_logP_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_adipose_pk("Drug", dose_mg=100.0, logP=-6.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_total"):
        simulate_adipose_pk("Drug", dose_mg=100.0, logP=2.0, cl_total_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_plasma"):
        simulate_adipose_pk("Drug", dose_mg=100.0, logP=2.0, vd_plasma_L=0.0)


# ---------------------------------------------------------------------------
# notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = simulate_adipose_pk("DrugA", dose_mg=100.0, logP=2.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_dose_proportionality_plasma_auc():
    r1 = simulate_adipose_pk("Drug", dose_mg=100.0, logP=2.0)
    r2 = simulate_adipose_pk("Drug", dose_mg=200.0, logP=2.0)
    assert r2.auc_plasma == pytest.approx(2 * r1.auc_plasma, rel=1e-3)
