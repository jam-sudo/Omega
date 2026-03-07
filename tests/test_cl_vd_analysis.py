"""Tests for Phase 365 — Clearance-Volume Relationship Analysis."""

import math

import numpy as np
import pytest

from omega_pbpk.analysis.cl_vd_analysis import (
    CLVdAnalysisResult,
    analyze_cl_vd_dataset,
    find_optimal_pk_window,
    plot_cl_vd_summary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NAMES_5 = ["CompA", "CompB", "CompC", "CompD", "CompE"]
# CL in L/h
CL_5 = [5.0, 15.0, 35.0, 50.0, 10.0]
# Vd in L
VD_5 = [20.0, 60.0, 100.0, 200.0, 30.0]


@pytest.fixture()
def result5():
    return analyze_cl_vd_dataset("TestSet", NAMES_5, CL_5, VD_5)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_returns_dataclass(result5):
    assert isinstance(result5, CLVdAnalysisResult)


def test_frozen_dataclass(result5):
    with pytest.raises((AttributeError, TypeError)):
        result5.n_compounds = 99  # type: ignore[misc]


def test_n_compounds(result5):
    assert result5.n_compounds == 5


def test_dataset_name(result5):
    assert result5.dataset_name == "TestSet"


def test_compound_names_preserved(result5):
    assert result5.compound_names == NAMES_5


# ---------------------------------------------------------------------------
# t½ calculation: t½ = 0.693 * Vd / CL
# ---------------------------------------------------------------------------


def test_t_half_formula(result5):
    for i, (cl, vd) in enumerate(zip(CL_5, VD_5)):
        expected = 0.693 * vd / cl
        assert abs(result5.t_half_values_h[i] - expected) < 1e-9


def test_t_half_length(result5):
    assert len(result5.t_half_values_h) == 5


# ---------------------------------------------------------------------------
# CL statistics
# ---------------------------------------------------------------------------


def test_cl_mean(result5):
    assert abs(result5.cl_mean - np.mean(CL_5)) < 1e-9


def test_cl_median(result5):
    assert abs(result5.cl_median - np.median(CL_5)) < 1e-9


def test_cl_range(result5):
    assert result5.cl_range == (min(CL_5), max(CL_5))


def test_cl_cv_pct_positive(result5):
    assert result5.cl_cv_pct > 0


# ---------------------------------------------------------------------------
# Vd statistics
# ---------------------------------------------------------------------------


def test_vd_mean(result5):
    assert abs(result5.vd_mean - np.mean(VD_5)) < 1e-9


def test_vd_median(result5):
    assert abs(result5.vd_median - np.median(VD_5)) < 1e-9


# ---------------------------------------------------------------------------
# t½ statistics
# ---------------------------------------------------------------------------


def test_t_half_mean_positive(result5):
    assert result5.t_half_mean > 0


def test_t_half_cv_pct_non_negative(result5):
    assert result5.t_half_cv_pct >= 0


# ---------------------------------------------------------------------------
# Classification fractions
# ---------------------------------------------------------------------------


def test_high_cl_fraction_in_range(result5):
    assert 0.0 <= result5.high_cl_fraction <= 1.0


def test_high_vd_fraction_in_range(result5):
    assert 0.0 <= result5.high_vd_fraction <= 1.0


def test_long_thalf_fraction_in_range(result5):
    assert 0.0 <= result5.long_thalf_fraction <= 1.0


def test_high_cl_fraction_value(result5):
    # CL > 30: CompC (35) and CompD (50) → 2/5 = 0.4
    assert abs(result5.high_cl_fraction - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Dosing classification counts
# ---------------------------------------------------------------------------


def test_dosing_counts_sum_le_n(result5):
    total = result5.n_once_daily_candidates + result5.n_twice_daily_candidates + result5.n_rapid_clearance
    assert total <= result5.n_compounds


def test_n_once_daily_non_negative(result5):
    assert result5.n_once_daily_candidates >= 0


def test_n_twice_daily_non_negative(result5):
    assert result5.n_twice_daily_candidates >= 0


def test_n_rapid_clearance_non_negative(result5):
    assert result5.n_rapid_clearance >= 0


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


def test_pearson_r_correlated_data():
    # Fully correlated: higher CL → higher Vd
    cl = [5.0, 10.0, 20.0, 40.0, 80.0]
    vd = [10.0, 20.0, 40.0, 80.0, 160.0]
    r = analyze_cl_vd_dataset("corr", ["A", "B", "C", "D", "E"], cl, vd)
    assert r.pearson_r_cl_vd > 0.99


def test_pearson_r_constant_cl():
    # All same CL → correlation undefined / near 0 (pearsonr may return nan for constant array)
    cl = [10.0, 10.0, 10.0, 10.0, 10.0]
    vd = [10.0, 20.0, 30.0, 40.0, 50.0]
    r = analyze_cl_vd_dataset("const_cl", ["A", "B", "C", "D", "E"], cl, vd)
    # pearsonr returns nan for constant input; we accept nan or ~0
    assert math.isnan(r.pearson_r_cl_vd) or abs(r.pearson_r_cl_vd) < 0.05


# ---------------------------------------------------------------------------
# find_optimal_pk_window
# ---------------------------------------------------------------------------


def test_find_optimal_pk_window_returns_subset():
    # CompA: CL=5, Vd=20 → t½=2.772 h
    matches = find_optimal_pk_window(
        target_thalf_h=2.772,
        target_cl_L_per_h=5.0,
        cl_values_L_per_h=CL_5,
        vd_values_L=VD_5,
        compound_names=NAMES_5,
        tolerance_pct=30.0,
    )
    assert "CompA" in matches
    for name in matches:
        assert name in NAMES_5


def test_find_optimal_pk_window_no_match():
    matches = find_optimal_pk_window(
        target_thalf_h=1000.0,
        target_cl_L_per_h=0.001,
        cl_values_L_per_h=CL_5,
        vd_values_L=VD_5,
        compound_names=NAMES_5,
        tolerance_pct=5.0,
    )
    assert matches == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        analyze_cl_vd_dataset("bad", ["A", "B"], [5.0, 10.0, 15.0], [20.0, 30.0])


def test_too_few_compounds_raises():
    with pytest.raises(ValueError, match="3 compounds"):
        analyze_cl_vd_dataset("tiny", ["A", "B"], [5.0, 10.0], [20.0, 30.0])


def test_negative_cl_raises():
    with pytest.raises(ValueError):
        analyze_cl_vd_dataset("neg", ["A", "B", "C"], [-1.0, 5.0, 10.0], [20.0, 30.0, 40.0])


def test_negative_vd_raises():
    with pytest.raises(ValueError):
        analyze_cl_vd_dataset("neg", ["A", "B", "C"], [1.0, 5.0, 10.0], [20.0, -5.0, 40.0])


# ---------------------------------------------------------------------------
# plot_cl_vd_summary
# ---------------------------------------------------------------------------


def test_plot_cl_vd_summary_returns_string(result5):
    out = plot_cl_vd_summary(result5)
    assert isinstance(out, str)
    assert "TestSet" in out
    assert "CompA" in out
