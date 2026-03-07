"""Tests for core/iv_bolus_pk.py — Phase 648."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.iv_bolus_pk import (
    IVBolusPKResult,
    analytical_1cpt_iv_bolus,
    analytical_2cpt_iv_bolus,
    simulate_iv_bolus_1cpt,
    simulate_iv_bolus_2cpt,
)

# ---------------------------------------------------------------------------
# 1-compartment tests
# ---------------------------------------------------------------------------


def test_simulate_1cpt_returns_result():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result, IVBolusPKResult)


def test_simulate_1cpt_n_compartments():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert result.n_compartments == 1


def test_1cpt_cmax_equals_dose_over_vd():
    dose = 100.0
    vd = 50.0
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=dose, cl_L_per_h=5.0, vd_L=vd)
    expected_cmax = dose / vd
    assert abs(result.cmax_mg_L - expected_cmax) < 1e-9


def test_1cpt_thalf_approximately_ln2_vd_cl():
    cl = 5.0
    vd = 50.0
    expected_thalf = math.log(2.0) * vd / cl
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=cl, vd_L=vd)
    assert abs(result.t_half_h - expected_thalf) / expected_thalf < 0.10


def test_1cpt_auc_approximately_dose_over_cl():
    dose = 100.0
    cl = 5.0
    result = simulate_iv_bolus_1cpt(
        "DrugA", dose_mg=dose, cl_L_per_h=cl, vd_L=50.0, t_end_h=100.0, dt_h=0.05
    )
    expected_auc = dose / cl
    assert abs(result.auc_mg_h_per_L - expected_auc) / expected_auc < 0.05


def test_1cpt_monoexponential_decay_at_t5():
    dose = 100.0
    cl = 5.0
    vd = 50.0
    t_check = 5.0
    ke = cl / vd
    expected_c = (dose / vd) * math.exp(-ke * t_check)
    ana = analytical_1cpt_iv_bolus(dose, cl, vd, [t_check])
    assert abs(ana[0] - expected_c) < 1e-9


def test_1cpt_vss_equals_vd():
    vd = 50.0
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=vd)
    assert abs(result.vss_L - vd) < 1e-9


def test_1cpt_peripheral_empty():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert result.c_peripheral_mg_L == []


def test_1cpt_times_len_equals_c_plasma_len():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_1cpt_residual_error_pct_below_5():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, dt_h=0.05)
    assert result.residual_error_pct < 5.0


def test_1cpt_notes_nonempty():
    result = simulate_iv_bolus_1cpt("DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert result.notes != ""


# ---------------------------------------------------------------------------
# 2-compartment tests
# ---------------------------------------------------------------------------


def test_simulate_2cpt_returns_result():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert isinstance(result, IVBolusPKResult)


def test_simulate_2cpt_n_compartments():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert result.n_compartments == 2


def test_2cpt_peripheral_nonempty():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert len(result.c_peripheral_mg_L) > 0


def test_2cpt_vss_equals_vd1_plus_vd2():
    vd1, vd2 = 20.0, 80.0
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=vd1, vd2_L=vd2, q12_L_per_h=10.0
    )
    assert abs(result.vss_L - (vd1 + vd2)) < 1e-9


def test_2cpt_cmax_equals_dose_over_vd1():
    dose = 100.0
    vd1 = 20.0
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=dose, cl_L_per_h=5.0, vd1_L=vd1, vd2_L=80.0, q12_L_per_h=10.0
    )
    expected_cmax = dose / vd1
    assert abs(result.cmax_mg_L - expected_cmax) < 1e-9


def test_2cpt_double_dose_doubles_cmax():
    base = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    double = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=200.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert abs(double.cmax_mg_L - 2.0 * base.cmax_mg_L) < 1e-9


def test_2cpt_len_peripheral_equals_len_times():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert len(result.c_peripheral_mg_L) == len(result.times_h)


def test_2cpt_len_times_equals_len_c_plasma():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_2cpt_residual_error_pct_below_5():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0, dt_h=0.05
    )
    assert result.residual_error_pct < 5.0


def test_2cpt_notes_nonempty():
    result = simulate_iv_bolus_2cpt(
        "DrugB", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=80.0, q12_L_per_h=10.0
    )
    assert result.notes != ""


# ---------------------------------------------------------------------------
# analytical_1cpt_iv_bolus
# ---------------------------------------------------------------------------


def test_analytical_1cpt_returns_list():
    result = analytical_1cpt_iv_bolus(100.0, 5.0, 50.0, [0.0, 1.0, 2.0])
    assert isinstance(result, list)
    assert len(result) == 3


def test_analytical_1cpt_at_t0_equals_c0():
    dose = 100.0
    vd = 50.0
    result = analytical_1cpt_iv_bolus(dose, 5.0, vd, [0.0])
    assert abs(result[0] - dose / vd) < 1e-9


def test_analytical_1cpt_matches_expected_at_known_time():
    dose = 200.0
    cl = 10.0
    vd = 40.0
    ke = cl / vd
    t = 3.0
    expected = (dose / vd) * math.exp(-ke * t)
    result = analytical_1cpt_iv_bolus(dose, cl, vd, [t])
    assert abs(result[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# analytical_2cpt_iv_bolus
# ---------------------------------------------------------------------------


def test_analytical_2cpt_returns_positive_values():
    result = analytical_2cpt_iv_bolus(100.0, 5.0, 20.0, 80.0, 10.0, [0.0, 1.0, 5.0])
    assert isinstance(result, list)
    assert all(v > 0 for v in result)


def test_analytical_2cpt_at_t0_equals_dose_over_vd1():
    dose = 100.0
    vd1 = 20.0
    result = analytical_2cpt_iv_bolus(dose, 5.0, vd1, 80.0, 10.0, [0.0])
    expected = dose / vd1
    assert abs(result[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_1cpt_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iv_bolus_1cpt("D", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_1cpt_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_iv_bolus_1cpt("D", dose_mg=100.0, cl_L_per_h=0.0, vd_L=50.0)


def test_2cpt_vd2_zero_raises():
    with pytest.raises(ValueError, match="vd2_L"):
        simulate_iv_bolus_2cpt(
            "D", dose_mg=100.0, cl_L_per_h=5.0, vd1_L=20.0, vd2_L=0.0, q12_L_per_h=10.0
        )
