"""Tests for core/dermal_absorption_pbpk.py — Phase 878."""

import pytest

from omega_pbpk.core.dermal_absorption_pbpk import (
    DermalAbsorptionResult,
    compare_vehicles,
    simulate_dermal_absorption,
)

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_returns_dermal_absorption_result():
    result = simulate_dermal_absorption("estradiol", dose_mg=1.0)
    assert isinstance(result, DermalAbsorptionResult)


def test_result_fields_correct_types():
    result = simulate_dermal_absorption("estradiol", dose_mg=1.0)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.dose_mg, float)
    assert isinstance(result.vehicle, str)
    assert isinstance(result.application_area_cm2, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.a_sc_mg, list)
    assert isinstance(result.a_dermis_mg, list)
    assert isinstance(result.cmax_plasma_mg_L, float)
    assert isinstance(result.tmax_plasma_h, float)
    assert isinstance(result.auc_plasma_mg_h_per_L, float)
    assert isinstance(result.flux_steady_state_ug_cm2_h, float)
    assert isinstance(result.lag_time_h, float)
    assert isinstance(result.f_absorbed, float)
    assert isinstance(result.notes, str)


def test_cmax_positive():
    result = simulate_dermal_absorption("drug_a", dose_mg=10.0)
    assert result.cmax_plasma_mg_L > 0


def test_auc_positive():
    result = simulate_dermal_absorption("drug_a", dose_mg=10.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_times_length_matches_arrays():
    result = simulate_dermal_absorption("drug_a", dose_mg=10.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.a_sc_mg)
    assert len(result.times_h) == len(result.a_dermis_mg)


# ---------------------------------------------------------------------------
# Vehicle-specific tests
# ---------------------------------------------------------------------------


def test_solution_vehicle_stored():
    result = simulate_dermal_absorption("drug", dose_mg=1.0, vehicle="solution")
    assert result.vehicle == "solution"


def test_patch_vehicle_stored():
    result = simulate_dermal_absorption("drug", dose_mg=1.0, vehicle="patch")
    assert result.vehicle == "patch"


def test_patch_gives_higher_auc_than_cream():
    """Patch multiplier (1.5) > cream multiplier (0.6)."""
    patch = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="patch")
    cream = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="cream")
    assert patch.auc_plasma_mg_h_per_L > cream.auc_plasma_mg_h_per_L


def test_solution_higher_auc_than_cream():
    """Solution multiplier (1.0) > cream multiplier (0.6)."""
    sol = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="solution")
    cream = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="cream")
    assert sol.auc_plasma_mg_h_per_L > cream.auc_plasma_mg_h_per_L


def test_patch_higher_auc_than_gel():
    """Patch multiplier (1.5) > gel multiplier (0.8)."""
    patch = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="patch")
    gel = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="gel")
    assert patch.auc_plasma_mg_h_per_L > gel.auc_plasma_mg_h_per_L


# ---------------------------------------------------------------------------
# Pharmacological sensitivity tests
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    res1 = simulate_dermal_absorption("drug", dose_mg=5.0)
    res2 = simulate_dermal_absorption("drug", dose_mg=10.0)
    ratio = res2.cmax_plasma_mg_L / res1.cmax_plasma_mg_L
    assert abs(ratio - 2.0) < 0.1, f"Expected ~2x Cmax, got ratio={ratio:.3f}"


def test_double_dose_doubles_auc():
    res1 = simulate_dermal_absorption("drug", dose_mg=5.0)
    res2 = simulate_dermal_absorption("drug", dose_mg=10.0)
    ratio = res2.auc_plasma_mg_h_per_L / res1.auc_plasma_mg_h_per_L
    assert abs(ratio - 2.0) < 0.1, f"Expected ~2x AUC, got ratio={ratio:.3f}"


def test_higher_kp_gives_higher_auc():
    res_hi = simulate_dermal_absorption("drug", dose_mg=10.0, kp_cm_h=0.01)
    res_lo = simulate_dermal_absorption("drug", dose_mg=10.0, kp_cm_h=0.001)
    assert res_hi.auc_plasma_mg_h_per_L > res_lo.auc_plasma_mg_h_per_L


def test_flux_positive():
    result = simulate_dermal_absorption("drug", dose_mg=10.0)
    assert result.flux_steady_state_ug_cm2_h > 0


def test_lag_time_positive():
    result = simulate_dermal_absorption("drug", dose_mg=10.0)
    assert result.lag_time_h > 0


def test_f_absorbed_nonnegative():
    result = simulate_dermal_absorption("drug", dose_mg=10.0)
    assert result.f_absorbed >= 0


def test_notes_present():
    result = simulate_dermal_absorption("drug", dose_mg=10.0)
    assert len(result.notes) > 0


def test_patch_notes_mention_patch():
    result = simulate_dermal_absorption("drug", dose_mg=10.0, vehicle="patch")
    assert "patch" in result.notes.lower() or "Patch" in result.notes


# ---------------------------------------------------------------------------
# compare_vehicles tests
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_vehicles("drug", dose_mg=10.0)
    assert len(results) == 4


def test_compare_sorted_descending_auc():
    results = compare_vehicles("drug", dose_mg=10.0)
    aucs = [r.auc_plasma_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_contains_all_vehicles():
    results = compare_vehicles("drug", dose_mg=10.0)
    vehicles = {r.vehicle for r in results}
    assert vehicles == {"solution", "cream", "gel", "patch"}


def test_compare_patch_is_first():
    """Patch should yield highest AUC → first in sorted list."""
    results = compare_vehicles("drug", dose_mg=10.0)
    assert results[0].vehicle == "patch"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dermal_absorption("drug", dose_mg=0.0)


def test_validation_area_zero_raises():
    with pytest.raises(ValueError, match="application_area_cm2"):
        simulate_dermal_absorption("drug", dose_mg=1.0, application_area_cm2=0.0)


def test_validation_kp_negative_raises():
    with pytest.raises(ValueError, match="kp_cm_h"):
        simulate_dermal_absorption("drug", dose_mg=1.0, kp_cm_h=-0.001)


def test_validation_invalid_vehicle_raises():
    with pytest.raises(ValueError, match="vehicle"):
        simulate_dermal_absorption("drug", dose_mg=1.0, vehicle="ointment")
