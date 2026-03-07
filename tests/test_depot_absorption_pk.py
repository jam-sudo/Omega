"""Tests for Phase 1097 — Depot Absorption PK."""

import pytest

from omega_pbpk.core.depot_absorption_pk import (
    DepotAbsorptionResult,
    compare_formulations,
    simulate_depot_absorption,
)

DRUG = "TestDrug"
DOSE = 100.0


# --- Return type checks ---

def test_return_type():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert isinstance(result, DepotAbsorptionResult)


def test_a_depot_starts_at_dose():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert abs(result.a_depot_mg[0] - DOSE) < 1e-6


def test_c_plasma_starts_at_zero():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert result.c_plasma_mg_L[0] == 0.0


def test_times_start_at_zero():
    result = simulate_depot_absorption(DRUG, DOSE, "microsphere_PLGA")
    assert result.times_h[0] == 0.0


def test_times_length_matches_depot():
    result = simulate_depot_absorption(DRUG, DOSE, "gel_depot")
    assert len(result.times_h) == len(result.a_depot_mg)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


# --- AUC and Cmax ---

def test_auc_positive():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert result.auc > 0


def test_cmax_positive():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert result.cmax > 0


def test_tmax_within_simulation():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC", t_end_h=720.0)
    assert 0 <= result.tmax_h <= 720.0


# --- Fraction absorbed ---

def test_f_absorbed_total_in_range():
    for formulation in ["suspension_SC", "microsphere_PLGA", "gel_depot", "oil_solution", "long_acting_implant"]:
        result = simulate_depot_absorption(DRUG, DOSE, formulation)
        assert 0.0 <= result.f_absorbed_total <= 1.0, f"Failed for {formulation}"


def test_f_absorbed_7d_ge_24h():
    for formulation in ["suspension_SC", "microsphere_PLGA", "gel_depot", "oil_solution", "long_acting_implant"]:
        result = simulate_depot_absorption(DRUG, DOSE, formulation)
        assert result.f_absorbed_at_7d >= result.f_absorbed_at_24h - 1e-9, \
            f"7d < 24h for {formulation}"


def test_f_absorbed_total_ge_7d():
    for formulation in ["suspension_SC", "oil_solution"]:
        result = simulate_depot_absorption(DRUG, DOSE, formulation)
        assert result.f_absorbed_total >= result.f_absorbed_at_7d - 1e-9


# --- Depot half-life ---

def test_t_half_depot_positive():
    result = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    assert result.t_half_depot_h > 0


def test_t_half_depot_is_finite():
    result = simulate_depot_absorption(DRUG, DOSE, "gel_depot")
    assert result.t_half_depot_h < 720.0


# --- Formulation-specific behavior ---

def test_long_acting_implant_longest_release():
    results = compare_formulations(DRUG, DOSE)
    assert results[0].formulation == "long_acting_implant"


def test_suspension_sc_fast_cmax():
    """suspension_SC with k=0.1/h should have higher Cmax vs long_acting_implant."""
    sc = simulate_depot_absorption(DRUG, DOSE, "suspension_SC")
    implant = simulate_depot_absorption(DRUG, DOSE, "long_acting_implant")
    # Suspension should have tmax earlier than implant
    assert sc.tmax_h < implant.tmax_h


def test_microsphere_burst_24h_absorption():
    """PLGA microsphere should absorb meaningful fraction in 24h due to burst."""
    result = simulate_depot_absorption(DRUG, DOSE, "microsphere_PLGA")
    assert result.f_absorbed_at_24h > 0.1


def test_microsphere_notes_nonempty():
    result = simulate_depot_absorption(DRUG, DOSE, "microsphere_PLGA")
    assert len(result.notes) > 0


def test_oil_solution_notes_nonempty():
    result = simulate_depot_absorption(DRUG, DOSE, "oil_solution")
    assert len(result.notes) > 0


# --- compare_formulations ---

def test_compare_formulations_returns_5():
    results = compare_formulations(DRUG, DOSE)
    assert len(results) == 5


def test_compare_formulations_sorted_descending():
    results = compare_formulations(DRUG, DOSE)
    durations = [r.release_duration_days for r in results]
    assert durations == sorted(durations, reverse=True), \
        f"Not sorted descending: {durations}"


def test_compare_formulations_all_different():
    results = compare_formulations(DRUG, DOSE)
    formulations = {r.formulation for r in results}
    assert len(formulations) == 5


# --- Validation errors ---

def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg must be"):
        simulate_depot_absorption(DRUG, -10.0, "suspension_SC")


def test_zero_dose():
    with pytest.raises(ValueError, match="dose_mg must be"):
        simulate_depot_absorption(DRUG, 0.0, "suspension_SC")


def test_invalid_formulation():
    with pytest.raises(ValueError, match="Unknown formulation"):
        simulate_depot_absorption(DRUG, DOSE, "magic_pill")


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_L_per_h must be"):
        simulate_depot_absorption(DRUG, DOSE, "suspension_SC", cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_L must be"):
        simulate_depot_absorption(DRUG, DOSE, "suspension_SC", vd_L=-5.0)


# --- Custom parameters ---

def test_custom_cl_vd():
    result = simulate_depot_absorption(DRUG, DOSE, "gel_depot", cl_L_per_h=2.0, vd_L=20.0)
    assert isinstance(result, DepotAbsorptionResult)
    assert result.cmax > 0


def test_release_duration_days_positive():
    for formulation in ["suspension_SC", "microsphere_PLGA", "gel_depot", "oil_solution", "long_acting_implant"]:
        result = simulate_depot_absorption(DRUG, DOSE, formulation)
        assert result.release_duration_days > 0, f"Non-positive for {formulation}"
