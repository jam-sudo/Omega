"""Tests for Phase 270: core/intranasal_pbpk_p270.py"""

import pytest

from omega_pbpk.core.intranasal_pbpk_p270 import (
    IntranasalPBPKResult,
    compare_doses,
    simulate_intranasal_pbpk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(dose_mg=5.0, cl_L_per_h=5.0, t_end_h=24.0):
    return simulate_intranasal_pbpk(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        t_end_h=t_end_h,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _sim()
    assert isinstance(result, IntranasalPBPKResult)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_nasal_starts_high():
    """Nasal concentration at t=0 should equal dose/V_nasal."""
    result = _sim(dose_mg=5.0)
    # V_nasal = 0.02 L => C_nasal(0) = 5.0/0.02 = 250 mg/mL
    assert result.c_nasal_mg_mL[0] == pytest.approx(250.0, rel=1e-6)


def test_plasma_starts_at_zero():
    result = _sim()
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_brain_starts_at_zero():
    result = _sim()
    assert result.c_brain_mg_L[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Nasal concentration declines
# ---------------------------------------------------------------------------


def test_nasal_declines_over_time():
    result = _sim()
    # Nasal should be strictly decreasing initially (first few steps)
    assert result.c_nasal_mg_mL[0] > result.c_nasal_mg_mL[1]
    assert result.c_nasal_mg_mL[1] > result.c_nasal_mg_mL[2]


def test_nasal_reaches_near_zero_at_end():
    result = _sim(t_end_h=48.0)
    # Combined k = 0.05+0.02+0.3+0.4 = 0.77/h, 48h => very small
    assert result.c_nasal_mg_mL[-1] < result.c_nasal_mg_mL[0] * 1e-5


# ---------------------------------------------------------------------------
# Brain and plasma concentrations
# ---------------------------------------------------------------------------


def test_cmax_brain_positive():
    result = _sim()
    assert result.cmax_brain_mg_L > 0.0


def test_cmax_plasma_positive():
    result = _sim()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_nasal_equals_initial():
    """Maximum nasal concentration occurs at t=0."""
    result = _sim()
    assert result.cmax_nasal_mg_mL == pytest.approx(result.c_nasal_mg_mL[0], rel=1e-6)


# ---------------------------------------------------------------------------
# AUC values
# ---------------------------------------------------------------------------


def test_auc_nasal_positive():
    result = _sim()
    assert result.auc_nasal_mg_h_per_mL > 0.0


def test_auc_plasma_positive():
    result = _sim()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_brain_positive():
    result = _sim()
    assert result.auc_brain_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Nose-to-brain ratio
# ---------------------------------------------------------------------------


def test_nose_to_brain_ratio_positive():
    result = _sim()
    assert result.nose_to_brain_ratio > 0.0


def test_nose_to_brain_ratio_formula():
    """nose_to_brain_ratio == auc_brain / auc_plasma when auc_plasma > 0."""
    result = _sim()
    expected = result.auc_brain_mg_h_per_L / result.auc_plasma_mg_h_per_L
    assert result.nose_to_brain_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Brain targeting score
# ---------------------------------------------------------------------------


def test_brain_targeting_score_in_range():
    result = _sim()
    assert 0.0 <= result.brain_targeting_score <= 100.0


def test_brain_targeting_score_capped_at_100():
    """Score should never exceed 100."""
    result = _sim()
    assert result.brain_targeting_score <= 100.0


def test_brain_targeting_score_formula():
    result = _sim()
    expected = min(100.0, result.nose_to_brain_ratio * 50.0)
    assert result.brain_targeting_score == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Time series length
# ---------------------------------------------------------------------------


def test_time_series_lengths_match():
    result = _sim()
    n = len(result.times_h)
    assert len(result.c_nasal_mg_mL) == n
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_brain_mg_L) == n


def test_time_starts_at_zero():
    result = _sim()
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_ends_at_t_end():
    result = _sim(t_end_h=12.0)
    assert result.times_h[-1] == pytest.approx(12.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_doses
# ---------------------------------------------------------------------------


def test_compare_doses_returns_list():
    results = compare_doses("TestDrug", [1.0, 5.0, 10.0], cl_L_per_h=5.0)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_doses_sorted_by_auc_brain_descending():
    results = compare_doses("TestDrug", [1.0, 5.0, 10.0], cl_L_per_h=5.0)
    aucs = [r.auc_brain_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_doses_higher_dose_higher_auc():
    """Higher dose should yield higher brain AUC (linear system)."""
    results = compare_doses("TestDrug", [1.0, 10.0], cl_L_per_h=5.0)
    # sorted descending so first has higher AUC
    assert results[0].auc_brain_mg_h_per_L > results[1].auc_brain_mg_h_per_L


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intranasal_pbpk("X", dose_mg=0.0, cl_L_per_h=5.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_intranasal_pbpk("X", dose_mg=-1.0, cl_L_per_h=5.0)


def test_invalid_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_intranasal_pbpk("X", dose_mg=5.0, cl_L_per_h=0.0)


def test_invalid_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_intranasal_pbpk("X", dose_mg=5.0, cl_L_per_h=-2.0)
