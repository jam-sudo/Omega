"""Tests for intramuscular depot PK simulation (Phase 1001)."""

import math
import pytest

from omega_pbpk.core.intramuscular_depot_pk import (
    IntramuscularDepotResult,
    simulate_im_depot,
)


def test_returns_correct_type():
    result = simulate_im_depot("TestDrug", dose_mg=100.0)
    assert isinstance(result, IntramuscularDepotResult)


def test_cmax_positive():
    result = simulate_im_depot("DrugA", dose_mg=50.0)
    assert result.cmax_mg_L > 0


def test_tmax_positive():
    result = simulate_im_depot("DrugA", dose_mg=50.0)
    assert result.tmax_h > 0


def test_auc_positive():
    result = simulate_im_depot("DrugA", dose_mg=50.0)
    assert result.auc_mg_h_per_L > 0


def test_f_absorbed_in_range():
    result = simulate_im_depot("DrugA", dose_mg=50.0)
    assert 0 < result.f_absorbed <= 1.0


def test_aqueous_cmax_greater_than_oily():
    aq = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="aqueous")
    oil = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="oily")
    assert aq.cmax_mg_L > oil.cmax_mg_L


def test_microsphere_tmax_greater_than_aqueous():
    aq = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="aqueous", t_end_h=200.0)
    ms = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="microsphere", t_end_h=200.0)
    assert ms.tmax_h > aq.tmax_h


def test_flip_flop_kinetics_is_bool():
    result = simulate_im_depot("DrugA", dose_mg=100.0)
    assert isinstance(result.flip_flop_kinetics, bool)


def test_aqueous_initial_depot_zero():
    result = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="aqueous")
    assert result.a_depot_mg[0] == 0.0


def test_aqueous_initial_im_solution_equals_dose():
    result = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="aqueous")
    assert math.isclose(result.a_im_solution_mg[0], 100.0)


def test_oily_initial_depot_equals_dose():
    result = simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="oily")
    assert math.isclose(result.a_depot_mg[0], 100.0)


def test_microsphere_initial_depot_equals_dose():
    result = simulate_im_depot("DrugA", dose_mg=50.0, formulation_type="microsphere")
    assert math.isclose(result.a_depot_mg[0], 50.0)


def test_nanosuspension_initial_depot_equals_dose():
    result = simulate_im_depot("DrugA", dose_mg=50.0, formulation_type="nanosuspension")
    assert math.isclose(result.a_depot_mg[0], 50.0)


def test_plasma_initial_concentration_near_zero():
    result = simulate_im_depot("DrugA", dose_mg=100.0)
    assert math.isclose(result.c_plasma_mg_L[0], 0.0, abs_tol=1e-9)


def test_formulation_type_preserved():
    for ft in ["aqueous", "oily", "microsphere", "nanosuspension"]:
        result = simulate_im_depot("DrugA", dose_mg=50.0, formulation_type=ft)
        assert result.formulation_type == ft


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=0.0)


def test_invalid_formulation_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=100.0, formulation_type="gel")


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=100.0, cl_L_per_h=-1.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=100.0, vd_L=0.0)


def test_invalid_solubility_raises():
    with pytest.raises(ValueError):
        simulate_im_depot("DrugA", dose_mg=100.0, solubility_mg_mL=0.0)


def test_flip_flop_detected_when_slow_absorption():
    # Very small cl → large ke; or use very low vd and high cl to get ke > k_abs
    # ke = cl/vd; k_abs = 0.8 + 0.1*logp (max 2.0)
    # For flip-flop: k_abs < ke → ke > 2.0 → cl/vd > 2.0 → e.g. cl=100, vd=10
    result = simulate_im_depot(
        "DrugB", dose_mg=100.0, cl_L_per_h=100.0, vd_L=10.0, logp=1.0
    )
    assert result.flip_flop_kinetics is True


def test_no_flip_flop_for_high_vd():
    # ke = cl/vd = 1/100 = 0.01; k_abs = 0.8 → no flip-flop
    result = simulate_im_depot(
        "DrugC", dose_mg=100.0, cl_L_per_h=1.0, vd_L=100.0, logp=1.0
    )
    assert result.flip_flop_kinetics is False


def test_times_list_starts_at_zero():
    result = simulate_im_depot("DrugA", dose_mg=100.0)
    assert result.times_h[0] == 0.0


def test_dose_preserved():
    result = simulate_im_depot("DrugA", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_drug_name_preserved():
    result = simulate_im_depot("MyDrug", dose_mg=100.0)
    assert result.drug_name == "MyDrug"
