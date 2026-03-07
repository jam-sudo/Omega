"""Tests for omega_pbpk.core.intramuscular_pk (Phase 717)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.intramuscular_pk import (
    IntramuscularPKResult,
    compare_im_formulations,
    simulate_im_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_intramuscular_pk_result():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, IntramuscularPKResult)


def test_initial_depot_equals_dose():
    dose = 150.0
    result = simulate_im_pk("TestDrug", dose_mg=dose)
    assert result.a_im_mg[0] == pytest.approx(dose, rel=1e-9)


def test_initial_plasma_conc_zero():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_cmax_positive():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_positive():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert result.tmax_h > 0.0


def test_auc_positive():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_t_half_positive():
    result = simulate_im_pk("TestDrug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    expected_ke = 5.0 / 50.0
    expected_t_half = math.log(2.0) / expected_ke
    assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-6)


def test_lag_time_positive():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert result.lag_time_h >= 0.0


def test_f_absorbed_approaches_one():
    # With long enough simulation, essentially all drug absorbed
    result = simulate_im_pk(
        "TestDrug", dose_mg=100.0, ka_im_per_h=2.0, cl_L_per_h=5.0, vd_L=50.0, t_end_h=48.0
    )
    assert result.f_absorbed > 0.90


def test_notes_nonempty():
    result = simulate_im_pk("TestDrug", dose_mg=100.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Solution vs depot comparison
# ---------------------------------------------------------------------------


def test_solution_tmax_less_than_depot_tmax():
    # Solution: fast release ka=2.0/h → early Tmax
    # Depot: mostly slow release (f_fast=0.2, ka_slow=0.05) → late Tmax
    sol = simulate_im_pk(
        "TestDrug",
        dose_mg=100.0,
        formulation="solution",
        ka_im_per_h=2.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=72.0,
    )
    dep = simulate_im_pk(
        "TestDrug",
        dose_mg=100.0,
        formulation="depot",
        ka_fast_per_h=1.0,
        ka_slow_per_h=0.05,
        f_fast=0.2,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=72.0,
    )
    assert sol.tmax_h < dep.tmax_h


def test_compare_im_formulations_returns_dict():
    result = compare_im_formulations("TestDrug", dose_mg=100.0)
    assert isinstance(result, dict)
    assert "solution" in result
    assert "depot" in result
    assert "tmax_ratio" in result
    assert "auc_ratio" in result
    assert "notes" in result


def test_compare_depot_tmax_greater_than_solution():
    comp = compare_im_formulations("TestDrug", dose_mg=100.0)
    assert comp["tmax_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = simulate_im_pk("TestDrug", dose_mg=100.0, ka_im_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0)
    r2 = simulate_im_pk("TestDrug", dose_mg=200.0, ka_im_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0)
    assert r2.cmax_mg_L == pytest.approx(r1.cmax_mg_L * 2.0, rel=1e-3)


def test_double_dose_doubles_auc():
    r1 = simulate_im_pk("TestDrug", dose_mg=100.0, ka_im_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0)
    r2 = simulate_im_pk("TestDrug", dose_mg=200.0, ka_im_per_h=1.0, cl_L_per_h=5.0, vd_L=50.0)
    assert r2.auc_mg_h_per_L == pytest.approx(r1.auc_mg_h_per_L * 2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Blood flow effect
# ---------------------------------------------------------------------------


def test_higher_blood_flow_faster_absorption():
    r_low = simulate_im_pk("TestDrug", dose_mg=100.0, ka_im_per_h=1.0, muscle_flow_mL_per_min=3.0)
    r_high = simulate_im_pk("TestDrug", dose_mg=100.0, ka_im_per_h=1.0, muscle_flow_mL_per_min=15.0)
    # Higher blood flow → faster ka → earlier Tmax
    assert r_high.tmax_h < r_low.tmax_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_formulation_raises_value_error():
    with pytest.raises(ValueError, match="formulation"):
        simulate_im_pk("TestDrug", dose_mg=100.0, formulation="inhalation")


def test_invalid_dose_zero_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_pk("TestDrug", dose_mg=0.0)


def test_invalid_dose_negative_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_pk("TestDrug", dose_mg=-50.0)


def test_invalid_cl_raises_value_error():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_im_pk("TestDrug", dose_mg=100.0, cl_L_per_h=-1.0)


def test_invalid_vd_raises_value_error():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_im_pk("TestDrug", dose_mg=100.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Depot formulation
# ---------------------------------------------------------------------------


def test_depot_initial_depot_equals_dose():
    dose = 200.0
    result = simulate_im_pk("TestDrug", dose_mg=dose, formulation="depot")
    assert result.a_im_mg[0] == pytest.approx(dose, rel=1e-9)


def test_depot_formulation_name_stored():
    result = simulate_im_pk("TestDrug", dose_mg=100.0, formulation="depot")
    assert result.formulation == "depot"


def test_solution_formulation_name_stored():
    result = simulate_im_pk("TestDrug", dose_mg=100.0, formulation="solution")
    assert result.formulation == "solution"


def test_compare_notes_nonempty():
    comp = compare_im_formulations("TestDrug", dose_mg=100.0)
    assert len(comp["notes"]) > 0
