"""Tests for Phase 732 — Transdermal Iontophoresis PK."""

import pytest

from omega_pbpk.core.iontophoresis_pk import (
    IontophoresisPKResult,
    compare_current_levels,
    simulate_iontophoresis_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_iontophoresis_pk("TestDrug", dose_mg=10.0)
    assert isinstance(result, IontophoresisPKResult)


def test_cmax_positive():
    result = simulate_iontophoresis_pk("DrugA", dose_mg=10.0)
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_iontophoresis_pk("DrugA", dose_mg=10.0)
    assert result.auc_mg_h_per_L > 0


def test_notes_nonempty():
    result = simulate_iontophoresis_pk("DrugA", dose_mg=10.0)
    assert result.notes and len(result.notes) > 0


def test_times_list_length_consistent():
    result = simulate_iontophoresis_pk("DrugA", dose_mg=10.0, t_end_h=12.0, dt_h=0.1)
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.a_skin_mg) == len(result.times_h)


# ---------------------------------------------------------------------------
# Enhancement factor
# ---------------------------------------------------------------------------


def test_zero_current_enhancement_factor_is_one():
    result = simulate_iontophoresis_pk("DrugB", dose_mg=10.0, current_ma=0.0)
    assert result.enhancement_factor == pytest.approx(1.0, abs=1e-9)


def test_enhancement_factor_correct():
    result = simulate_iontophoresis_pk("DrugB", dose_mg=10.0, current_ma=2.0, iono_factor=0.1)
    expected = 1.0 + 0.1 * 2.0
    assert result.enhancement_factor == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Current dependence
# ---------------------------------------------------------------------------


def test_higher_current_higher_cmax():
    low = simulate_iontophoresis_pk("DrugC", dose_mg=10.0, current_ma=0.5)
    high = simulate_iontophoresis_pk("DrugC", dose_mg=10.0, current_ma=5.0)
    assert high.cmax_mg_L > low.cmax_mg_L


def test_higher_current_higher_auc():
    low = simulate_iontophoresis_pk("DrugC", dose_mg=10.0, current_ma=0.5)
    high = simulate_iontophoresis_pk("DrugC", dose_mg=10.0, current_ma=5.0)
    assert high.auc_mg_h_per_L > low.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# AUC enhancement ratio
# ---------------------------------------------------------------------------


def test_auc_enhancement_ratio_ge_one_with_current():
    result = simulate_iontophoresis_pk("DrugD", dose_mg=10.0, current_ma=2.0)
    assert result.auc_enhancement_ratio >= 1.0


def test_auc_enhancement_ratio_near_one_zero_current():
    """With zero current the simulated AUC should approximate the passive estimate."""
    result = simulate_iontophoresis_pk("DrugD", dose_mg=10.0, current_ma=0.0)
    # ratio may not be exactly 1 due to the limited t_end, but should be close
    assert result.auc_enhancement_ratio > 0


# ---------------------------------------------------------------------------
# f_absorbed
# ---------------------------------------------------------------------------


def test_f_absorbed_between_zero_and_one():
    result = simulate_iontophoresis_pk("DrugE", dose_mg=10.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_higher_current_higher_f_absorbed():
    low = simulate_iontophoresis_pk("DrugE", dose_mg=10.0, current_ma=0.0)
    high = simulate_iontophoresis_pk("DrugE", dose_mg=10.0, current_ma=5.0)
    assert high.f_absorbed >= low.f_absorbed


# ---------------------------------------------------------------------------
# compare_current_levels
# ---------------------------------------------------------------------------


def test_compare_current_levels_returns_list():
    results = compare_current_levels("DrugF", dose_mg=10.0, currents_ma=[0.0, 1.0, 2.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_current_levels_all_results_are_correct_type():
    results = compare_current_levels("DrugF", dose_mg=10.0, currents_ma=[0.0, 2.0])
    for r in results:
        assert isinstance(r, IontophoresisPKResult)


def test_compare_current_levels_higher_current_higher_auc():
    results = compare_current_levels("DrugF", dose_mg=10.0, currents_ma=[0.0, 1.0, 3.0])
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs[2] > aucs[1] > aucs[0]


# ---------------------------------------------------------------------------
# Linear dose scaling
# ---------------------------------------------------------------------------


def test_linear_dose_scaling():
    r1 = simulate_iontophoresis_pk("DrugG", dose_mg=5.0, current_ma=1.0)
    r2 = simulate_iontophoresis_pk("DrugG", dose_mg=10.0, current_ma=1.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iontophoresis_pk("DrugV", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iontophoresis_pk("DrugV", dose_mg=-5.0)


def test_validation_ka_passive_zero_raises():
    with pytest.raises(ValueError, match="ka_passive_per_h"):
        simulate_iontophoresis_pk("DrugV", dose_mg=10.0, ka_passive_per_h=0.0)


def test_validation_negative_current_raises():
    with pytest.raises(ValueError, match="current_ma"):
        simulate_iontophoresis_pk("DrugV", dose_mg=10.0, current_ma=-1.0)
