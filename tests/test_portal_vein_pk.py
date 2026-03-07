"""Tests for portal_vein_pk module — Phase 699."""

import pytest

from omega_pbpk.core.portal_vein_pk import (
    PortalVeinPKResult,
    compare_first_pass_effect,
    simulate_portal_vein_pk,
)


def test_returns_portal_vein_pk_result():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, PortalVeinPKResult)


def test_c_portal_starts_at_zero():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.c_portal_mg_L[0] == 0.0


def test_c_sys_starts_at_zero():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.c_sys_mg_L[0] == 0.0


def test_cmax_portal_positive():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_portal_mg_L > 0.0


def test_cmax_sys_positive():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_sys_mg_L > 0.0


def test_portal_to_systemic_ratio_above_one():
    """Portal concentration should be higher than systemic due to first-pass."""
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.portal_to_systemic_ratio > 1.0


def test_auc_sys_positive():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.auc_sys > 0.0


def test_auc_portal_positive():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert result.auc_portal > 0.0


def test_f_bioavailable_formula():
    """f_bioavailable = (1 - ER) * f_abs."""
    f_abs = 0.9
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, f_abs=f_abs)
    expected = (1.0 - result.er) * f_abs
    assert abs(result.f_bioavailable - expected) < 1e-9


def test_higher_cl_int_lower_f_bioavailable():
    """Higher CLint → more first-pass extraction → lower bioavailability."""
    low_cl = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=5.0)
    high_cl = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=100.0)
    assert high_cl.f_bioavailable < low_cl.f_bioavailable


def test_higher_cl_int_higher_er():
    """Higher CLint → higher extraction ratio."""
    low_cl = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=5.0)
    high_cl = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=100.0)
    assert high_cl.er > low_cl.er


def test_low_cl_int_near_zero_er():
    """Very low CLint → ER near 0."""
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=0.01)
    assert result.er < 0.05


def test_high_cl_int_near_unit_er():
    """Very high CLint → ER near 1."""
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=10000.0, fup=1.0)
    assert result.er > 0.95


def test_low_cl_int_f_bioavailable_near_f_abs():
    """Very low CLint → F ≈ f_abs (almost no hepatic extraction)."""
    f_abs = 0.9
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=0.01, f_abs=f_abs)
    assert abs(result.f_bioavailable - f_abs) < 0.05


def test_high_cl_int_f_bioavailable_near_zero():
    """Very high CLint → F near 0."""
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, cl_int_L_per_h=10000.0, fup=1.0)
    assert result.f_bioavailable < 0.05


def test_compare_first_pass_effect_returns_list():
    results = compare_first_pass_effect(
        "TestDrug", dose_mg=100.0, cl_int_scenarios=[1.0, 10.0, 100.0]
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_first_pass_effect_sorted_by_er_ascending():
    results = compare_first_pass_effect(
        "TestDrug", dose_mg=100.0, cl_int_scenarios=[100.0, 1.0, 50.0]
    )
    ers = [r.er for r in results]
    assert ers == sorted(ers)


def test_notes_nonempty():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0)
    assert len(result.notes) > 0


def test_drug_name_stored():
    result = simulate_portal_vein_pk("Aspirin", dose_mg=500.0)
    assert result.drug_name == "Aspirin"


def test_dose_mg_stored():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=250.0)
    assert result.dose_mg == 250.0


def test_times_list_length():
    result = simulate_portal_vein_pk("TestDrug", dose_mg=100.0, t_end_h=12.0, dt_h=0.05)
    # Should have n_steps = int(t_end_h / dt_h) + 1 = 241 steps
    assert len(result.times_h) == len(result.c_portal_mg_L) == len(result.c_sys_mg_L)
    assert len(result.times_h) > 0


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_vein_pk("TestDrug", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_vein_pk("TestDrug", dose_mg=-10.0)


def test_validation_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        simulate_portal_vein_pk("TestDrug", dose_mg=100.0, fup=1.5)


def test_validation_f_abs_zero_raises():
    with pytest.raises(ValueError, match="f_abs"):
        simulate_portal_vein_pk("TestDrug", dose_mg=100.0, f_abs=0.0)
