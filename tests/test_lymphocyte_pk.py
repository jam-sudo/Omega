"""Tests for Phase 473: Lymphocyte PK Model."""

import pytest

from omega_pbpk.core.lymphocyte_pk import (
    LymphocytePKResult,
    compare_p_gp_effect,
    simulate_lymphocyte_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_lymphocyte_pk("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert isinstance(result, LymphocytePKResult)


def test_drug_name_preserved():
    result = simulate_lymphocyte_pk("Cyclosporine", dose_mg=50.0, logP=3.0, cl_sys_L_per_h=2.0)
    assert result.drug_name == "Cyclosporine"


def test_dose_mg_preserved():
    result = simulate_lymphocyte_pk("DrugA", dose_mg=200.0, logP=1.0, cl_sys_L_per_h=3.0)
    assert result.dose_mg == 200.0


def test_p_gp_flag_preserved():
    result = simulate_lymphocyte_pk(
        "DrugB", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, p_gp_substrate=True
    )
    assert result.p_gp_substrate is True


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_dose_over_vd():
    """C_plasma[0] = dose_mg / vd_plasma_L."""
    dose = 100.0
    vd = 5.0
    result = simulate_lymphocyte_pk(
        "DrugA", dose_mg=dose, logP=2.0, cl_sys_L_per_h=5.0, vd_plasma_L=vd
    )
    assert abs(result.c_plasma_mg_L[0] - dose / vd) < 1e-9


def test_c_plasma_custom_vd():
    dose = 200.0
    vd = 8.0
    result = simulate_lymphocyte_pk(
        "DrugA", dose_mg=dose, logP=2.0, cl_sys_L_per_h=5.0, vd_plasma_L=vd
    )
    assert abs(result.c_plasma_mg_L[0] - dose / vd) < 1e-9


def test_c_lymph_starts_at_zero():
    """Lymphocyte concentration starts at 0 (IV bolus)."""
    result = simulate_lymphocyte_pk("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.c_lymph_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# CAR (cellular accumulation ratio)
# ---------------------------------------------------------------------------


def test_higher_logP_gives_higher_car():
    """Higher logP → higher CAR."""
    r_low = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=1.0, cl_sys_L_per_h=5.0)
    r_high = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=4.0, cl_sys_L_per_h=5.0)
    assert r_high.car > r_low.car


def test_negative_logP_gives_car_of_one():
    """logP <= 0 → CAR = 1.0 (before P-gp adjustment)."""
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=-1.0, cl_sys_L_per_h=5.0)
    assert abs(result.car - 1.0) < 1e-9


def test_car_clamped_at_max():
    """Very high logP → CAR clamped at 1000."""
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=20.0, cl_sys_L_per_h=5.0)
    assert result.car <= 1000.0


def test_car_formula_logP2():
    """CAR = 10^(0.5 * 2) = 10 for logP=2, no P-gp."""
    result = simulate_lymphocyte_pk(
        "D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, p_gp_substrate=False
    )
    expected_car = 10 ** (0.5 * 2.0)
    assert abs(result.car - expected_car) < 1e-9


# ---------------------------------------------------------------------------
# P-gp substrate effect
# ---------------------------------------------------------------------------


def test_p_gp_substrate_lower_car_than_non_substrate():
    """P-gp substrate has lower CAR."""
    r_non = simulate_lymphocyte_pk(
        "D", dose_mg=100.0, logP=3.0, cl_sys_L_per_h=5.0, p_gp_substrate=False
    )
    r_pgp = simulate_lymphocyte_pk(
        "D", dose_mg=100.0, logP=3.0, cl_sys_L_per_h=5.0, p_gp_substrate=True
    )
    assert r_non.car > r_pgp.car


def test_p_gp_reduces_car_by_factor_3():
    """P-gp substrate: CAR = base_CAR / 3."""
    r_non = simulate_lymphocyte_pk(
        "D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, p_gp_substrate=False
    )
    r_pgp = simulate_lymphocyte_pk(
        "D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, p_gp_substrate=True
    )
    assert abs(r_non.car / 3.0 - r_pgp.car) < 1e-9


# ---------------------------------------------------------------------------
# compare_p_gp_effect
# ---------------------------------------------------------------------------


def test_compare_p_gp_effect_returns_two_items():
    results = compare_p_gp_effect("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert len(results) == 2


def test_compare_p_gp_effect_sorted_by_car_descending():
    results = compare_p_gp_effect("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert results[0].car >= results[1].car


def test_compare_p_gp_effect_first_is_non_substrate():
    results = compare_p_gp_effect("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert results[0].p_gp_substrate is False
    assert results[1].p_gp_substrate is True


def test_compare_p_gp_effect_returns_lymphocyte_pk_results():
    results = compare_p_gp_effect("DrugA", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    for r in results:
        assert isinstance(r, LymphocytePKResult)


# ---------------------------------------------------------------------------
# AUC and metrics
# ---------------------------------------------------------------------------


def test_auc_lymph_positive():
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.auc_lymph_mg_h_per_L > 0


def test_auc_plasma_positive():
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_lymph_to_plasma_ratio_positive():
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.lymph_to_plasma_ratio > 0


def test_cmax_lymph_positive():
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.cmax_lymph_mg_L > 0


def test_tmax_lymph_nonnegative():
    result = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0)
    assert result.tmax_lymph_h >= 0.0


def test_higher_logP_higher_lymph_to_plasma_ratio():
    """Higher logP → drug accumulates more in lymphocytes."""
    r_low = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=1.0, cl_sys_L_per_h=5.0)
    r_high = simulate_lymphocyte_pk("D", dose_mg=100.0, logP=4.0, cl_sys_L_per_h=5.0)
    assert r_high.lymph_to_plasma_ratio > r_low.lymph_to_plasma_ratio


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphocyte_pk("D", dose_mg=-1.0, logP=2.0, cl_sys_L_per_h=5.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphocyte_pk("D", dose_mg=0.0, logP=2.0, cl_sys_L_per_h=5.0)


def test_negative_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=-1.0)


def test_negative_vd_plasma_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, vd_plasma_L=-1.0)


def test_negative_vd_lymph_raises():
    with pytest.raises(ValueError, match="vd_lymph_L"):
        simulate_lymphocyte_pk("D", dose_mg=100.0, logP=2.0, cl_sys_L_per_h=5.0, vd_lymph_L=-1.0)
