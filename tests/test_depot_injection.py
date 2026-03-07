"""Tests for core/depot_injection.py — Phase 595."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.depot_injection import (
    DepotInjectionResult,
    compare_formulations,
    simulate_depot_injection,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 100.0
CL = 2.0  # L/h
VD = 20.0  # L


def default_sim(**kwargs):
    return simulate_depot_injection(DRUG, DOSE, CL, VD, **kwargs)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_returns_result():
    result = default_sim()
    assert isinstance(result, DepotInjectionResult)


def test_lengths_match():
    result = default_sim(t_end_h=24.0, dt_h=0.5)
    n = len(result.times_h)
    assert len(result.c_depot_mg) == n
    assert len(result.c_plasma_mg_L) == n


def test_notes_nonempty():
    result = default_sim()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Tmax ordering across formulations (single route)
# ---------------------------------------------------------------------------


def test_aqueous_fast_absorption():
    r_aq = default_sim(formulation="aqueous")
    r_sus = default_sim(formulation="suspension")
    assert r_aq.tmax_h < r_sus.tmax_h


def test_oil_slow_tmax():
    r_aq = default_sim(formulation="aqueous", t_end_h=200.0)
    r_oil = default_sim(formulation="oil", t_end_h=200.0)
    assert r_oil.tmax_h > r_aq.tmax_h


def test_polymer_slowest_tmax():
    r_oil = default_sim(formulation="oil", t_end_h=500.0)
    r_poly = default_sim(formulation="polymer_microsphere", t_end_h=500.0)
    assert r_poly.tmax_h > r_oil.tmax_h


# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------


def test_im_faster_than_sc():
    r_sc = default_sim(route="sc", formulation="aqueous")
    r_im = default_sim(route="im", formulation="aqueous")
    # IM absorbs faster → earlier tmax
    assert r_im.tmax_h <= r_sc.tmax_h


# ---------------------------------------------------------------------------
# PK metric positivity
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = default_sim()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = default_sim()
    assert result.auc_mg_h_L > 0.0


def test_f_absorbed_in_range():
    result = default_sim()
    assert 0.0 <= result.f_absorbed_pct <= 100.0


def test_t_half_absorption_positive():
    result = default_sim()
    assert result.t_half_absorption_h > 0.0


def test_t_half_elimination_positive():
    result = default_sim()
    assert result.t_half_elimination_h > 0.0


def test_t_half_elimination_matches_formula():
    result = default_sim()
    expected = math.log(2) * VD / CL
    assert abs(result.t_half_elimination_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_dose_proportionality_cmax():
    r1 = simulate_depot_injection(DRUG, 100.0, CL, VD)
    r2 = simulate_depot_injection(DRUG, 200.0, CL, VD)
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 0.05


def test_dose_proportionality_auc():
    r1 = simulate_depot_injection(DRUG, 100.0, CL, VD)
    r2 = simulate_depot_injection(DRUG, 200.0, CL, VD)
    assert abs(r2.auc_mg_h_L / r1.auc_mg_h_L - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------


def test_lag_time_no_drug_before_lag():
    lag = 5.0
    result = simulate_depot_injection(DRUG, DOSE, CL, VD, lag_time_h=lag, dt_h=0.1, t_end_h=72.0)
    # All plasma concentrations before lag time should be ~0
    for t, c in zip(result.times_h, result.c_plasma_mg_L):
        if t < lag - 0.05:
            assert c < 1e-9, f"Expected ~0 plasma at t={t}, got {c}"


# ---------------------------------------------------------------------------
# Depot depletion
# ---------------------------------------------------------------------------


def test_depot_depletes_over_time():
    result = default_sim(t_end_h=72.0)
    # Depot should decrease monotonically (or at minimum be less at end than start)
    assert result.c_depot_mg[-1] < result.c_depot_mg[0]


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_list():
    results = compare_formulations(DRUG, DOSE, CL, VD)
    assert isinstance(results, list)


def test_compare_formulations_count():
    results = compare_formulations(DRUG, DOSE, CL, VD)
    assert len(results) == 4


def test_compare_formulations_sorted_by_tmax():
    results = compare_formulations(DRUG, DOSE, CL, VD, t_end_h=500.0)
    tmaxes = [r.tmax_h for r in results]
    assert tmaxes == sorted(tmaxes), f"Expected sorted tmax, got {tmaxes}"
    # Verify order: aqueous should be first (fastest), polymer last (slowest)
    formulations = [r.formulation for r in results]
    assert formulations[0] == "aqueous"
    assert formulations[-1] == "polymer_microsphere"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_depot_injection(DRUG, -10.0, CL, VD)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_depot_injection(DRUG, DOSE, 0.0, VD)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        default_sim(route="iv")


def test_invalid_formulation_raises():
    with pytest.raises(ValueError):
        default_sim(formulation="gel")


def test_invalid_f_bioavail_raises():
    with pytest.raises(ValueError):
        default_sim(f_bioavail=0.0)
