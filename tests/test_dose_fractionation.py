"""Tests for dose_fractionation module (Phase 247)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.dose_fractionation import (
    FractionationOptResult,
    FractionationResult,
    optimize_fractionation,
    simulate_fractionated_dosing,
)

# ---------------------------------------------------------------------------
# Test parameters — simple 1-cpt drug
# ---------------------------------------------------------------------------
DRUG = "TestDrug"
TDD = 100.0  # mg total daily dose
CL = 5.0  # L/h
VD = 50.0  # L


# ---------------------------------------------------------------------------
# Basic structure & types
# ---------------------------------------------------------------------------


def test_returns_fractionation_result():
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD)
    assert isinstance(r, FractionationResult)


def test_result_field_types():
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD)
    assert isinstance(r.drug_name, str)
    assert isinstance(r.total_daily_dose_mg, float)
    assert isinstance(r.n_fractions, int)
    assert isinstance(r.dose_per_fraction_mg, float)
    assert isinstance(r.interval_h, float)
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_plasma_mg_L, list)
    assert isinstance(r.css_max, float)
    assert isinstance(r.css_min, float)
    assert isinstance(r.css_avg, float)
    assert isinstance(r.fluctuation_pct, float)
    assert isinstance(r.accumulation_ratio, float)


def test_dose_per_fraction_correct():
    for n in [1, 2, 4]:
        r = simulate_fractionated_dosing(DRUG, TDD, n, CL, VD)
        assert pytest.approx(r.dose_per_fraction_mg, rel=1e-6) == TDD / n


def test_interval_correct():
    for n in [1, 2, 4]:
        r = simulate_fractionated_dosing(DRUG, TDD, n, CL, VD)
        assert pytest.approx(r.interval_h, rel=1e-6) == 24.0 / n


# ---------------------------------------------------------------------------
# PK properties
# ---------------------------------------------------------------------------


def test_css_max_greater_than_css_min():
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD)
    assert r.css_max > r.css_min


def test_css_max_greater_than_zero():
    r = simulate_fractionated_dosing(DRUG, TDD, 1, CL, VD)
    assert r.css_max > 0.0


def test_accumulation_ratio_at_least_one():
    """At steady state the max concentration should be >= single-dose Cmax."""
    r = simulate_fractionated_dosing(DRUG, TDD, 1, CL, VD)
    assert r.accumulation_ratio >= 1.0


def test_once_daily_highest_fluctuation():
    """n=1 (QD) should have higher fluctuation than n=4 (QID)."""
    r1 = simulate_fractionated_dosing(DRUG, TDD, 1, CL, VD)
    r4 = simulate_fractionated_dosing(DRUG, TDD, 4, CL, VD)
    assert r1.fluctuation_pct > r4.fluctuation_pct


def test_fluctuation_decreases_with_more_fractions():
    """Fluctuation must decrease (or stay equal) as fractions increase."""
    results = [simulate_fractionated_dosing(DRUG, TDD, n, CL, VD) for n in range(1, 5)]
    flucts = [r.fluctuation_pct for r in results]
    for i in range(len(flucts) - 1):
        assert flucts[i] >= flucts[i + 1] - 1e-3  # allow tiny numerical noise


def test_qid_lowest_fluctuation():
    """QID (n=4) should have the lowest fluctuation among 1-4."""
    results = {n: simulate_fractionated_dosing(DRUG, TDD, n, CL, VD) for n in range(1, 5)}
    min_fluct = min(results[n].fluctuation_pct for n in results)
    assert results[4].fluctuation_pct <= min_fluct + 1e-3


def test_double_dose_doubles_css():
    """Linear PK: 2x total dose should yield ~2x Css_max."""
    r1 = simulate_fractionated_dosing(DRUG, 100.0, 2, CL, VD)
    r2 = simulate_fractionated_dosing(DRUG, 200.0, 2, CL, VD)
    assert pytest.approx(r2.css_max / r1.css_max, rel=0.01) == 2.0


def test_fluctuation_pct_formula():
    """Check fluctuation_pct = (css_max - css_min) / css_avg * 100."""
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD)
    expected = (r.css_max - r.css_min) / r.css_avg * 100.0
    assert pytest.approx(r.fluctuation_pct, rel=1e-4) == expected


def test_iv_route():
    """IV route should work without errors."""
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD, route="iv")
    assert r.css_max > 0.0
    assert r.accumulation_ratio >= 1.0


def test_times_and_concentrations_same_length():
    r = simulate_fractionated_dosing(DRUG, TDD, 2, CL, VD)
    assert len(r.times_h) == len(r.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_returns_opt_result():
    opt = optimize_fractionation(DRUG, TDD, CL, VD)
    assert isinstance(opt, FractionationOptResult)


def test_optimize_n_fractions_in_range():
    opt = optimize_fractionation(DRUG, TDD, CL, VD)
    assert 1 <= opt.optimal_n_fractions <= 4


def test_optimize_all_results_length():
    opt = optimize_fractionation(DRUG, TDD, CL, VD)
    assert len(opt.all_results) == 4


def test_optimize_minimize_fluctuation_picks_highest_n():
    """Minimizing fluctuation should pick n=4 (QID)."""
    opt = optimize_fractionation(DRUG, TDD, CL, VD, target="minimize_fluctuation")
    assert opt.optimal_n_fractions == 4


def test_optimize_maximize_trough():
    opt = optimize_fractionation(DRUG, TDD, CL, VD, target="maximize_trough")
    assert 1 <= opt.optimal_n_fractions <= 4


def test_optimize_minimize_peak():
    opt = optimize_fractionation(DRUG, TDD, CL, VD, target="minimize_peak")
    assert 1 <= opt.optimal_n_fractions <= 4


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="total_daily_dose_mg"):
        simulate_fractionated_dosing(DRUG, -50.0, 2, CL, VD)


def test_invalid_n_fractions_raises():
    with pytest.raises(ValueError, match="n_fractions"):
        simulate_fractionated_dosing(DRUG, TDD, 0, CL, VD)


def test_invalid_n_fractions_too_large_raises():
    with pytest.raises(ValueError, match="n_fractions"):
        simulate_fractionated_dosing(DRUG, TDD, 13, CL, VD)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_fractionated_dosing(DRUG, TDD, 2, 0.0, VD)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_fractionated_dosing(DRUG, TDD, 2, CL, -1.0)
