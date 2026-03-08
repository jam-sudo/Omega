"""Tests for Phase 859 — Dissolution-Absorption Coupling."""

import pytest

from omega_pbpk.core.dissolution_absorption_coupling import (
    DissolutionAbsorptionResult,
    screen_particle_sizes,
    simulate_dissolution_absorption,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert isinstance(result, DissolutionAbsorptionResult)


def test_drug_name_preserved():
    result = simulate_dissolution_absorption("TestDrug", 50.0)
    assert result.drug_name == "TestDrug"


def test_dose_mg_preserved():
    result = simulate_dissolution_absorption("DrugB", 200.0)
    assert result.dose_mg == 200.0


def test_k_diss_preserved():
    result = simulate_dissolution_absorption("DrugC", 100.0, k_diss_per_h=0.8)
    assert result.k_diss_per_h == 0.8


def test_solubility_preserved():
    result = simulate_dissolution_absorption("DrugD", 100.0, solubility_mg_mL=2.0)
    assert result.solubility_mg_mL == 2.0


# ---------------------------------------------------------------------------
# Time series fields
# ---------------------------------------------------------------------------


def test_times_h_is_list():
    result = simulate_dissolution_absorption("DrugA", 100.0, t_end_h=4.0, dt_h=0.1)
    assert isinstance(result.times_h, list)
    assert len(result.times_h) > 0


def test_c_plasma_same_length_as_times():
    result = simulate_dissolution_absorption("DrugA", 100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_a_solid_same_length_as_times():
    result = simulate_dissolution_absorption("DrugA", 100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.a_solid_mg) == len(result.times_h)


def test_a_dissolved_same_length_as_times():
    result = simulate_dissolution_absorption("DrugA", 100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.a_dissolved_mg) == len(result.times_h)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_within_simulation():
    result = simulate_dissolution_absorption("DrugA", 100.0, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


def test_auc_positive():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert result.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


def test_f_dissolved_in_0_1():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert 0.0 <= result.f_dissolved <= 1.0


def test_f_absorbed_in_0_1():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_a_solid_non_negative():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert all(a >= 0.0 for a in result.a_solid_mg)


def test_a_solid_starts_at_dose():
    result = simulate_dissolution_absorption("DrugA", 100.0)
    assert abs(result.a_solid_mg[0] - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Dissolution-limited logic
# ---------------------------------------------------------------------------


def test_low_solubility_dissolution_limited():
    # Very low solubility → dissolution_limited = True
    result = simulate_dissolution_absorption(
        "PoorSolDrug",
        dose_mg=500.0,
        solubility_mg_mL=0.01,
        k_diss_per_h=0.1,
        t_end_h=24.0,
    )
    assert result.dissolution_limited is True


def test_high_solubility_not_dissolution_limited():
    # High solubility + fast dissolution → f_dissolved >= 0.9
    result = simulate_dissolution_absorption(
        "GoodSolDrug",
        dose_mg=10.0,
        solubility_mg_mL=10.0,
        k_diss_per_h=5.0,
        v_gi_mL=250.0,
        t_end_h=24.0,
    )
    assert result.dissolution_limited is False


# ---------------------------------------------------------------------------
# Pharmacological relationships
# ---------------------------------------------------------------------------


def test_higher_k_diss_higher_auc():
    result_slow = simulate_dissolution_absorption(
        "DrugX", 100.0, k_diss_per_h=0.1, solubility_mg_mL=1.0
    )
    result_fast = simulate_dissolution_absorption(
        "DrugX", 100.0, k_diss_per_h=2.0, solubility_mg_mL=1.0
    )
    assert result_fast.auc_mg_h_per_L > result_slow.auc_mg_h_per_L


def test_higher_solubility_higher_f_dissolved():
    result_low = simulate_dissolution_absorption(
        "DrugY", 200.0, solubility_mg_mL=0.1, k_diss_per_h=0.5
    )
    result_high = simulate_dissolution_absorption(
        "DrugY", 200.0, solubility_mg_mL=5.0, k_diss_per_h=0.5
    )
    assert result_high.f_dissolved >= result_low.f_dissolved


# ---------------------------------------------------------------------------
# screen_particle_sizes
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_particle_sizes("DrugA", 100.0, [0.1, 0.5, 1.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_by_auc_descending():
    results = screen_particle_sizes("DrugA", 100.0, [0.1, 0.5, 1.0, 2.0])
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_screen_single_value():
    results = screen_particle_sizes("DrugA", 100.0, [0.5])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_dissolution_absorption("DrugA", 0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_dissolution_absorption("DrugA", -10.0)


def test_validation_k_diss_zero():
    with pytest.raises(ValueError, match="k_diss_per_h must be > 0"):
        simulate_dissolution_absorption("DrugA", 100.0, k_diss_per_h=0.0)


def test_validation_solubility_zero():
    with pytest.raises(ValueError, match="solubility_mg_mL must be > 0"):
        simulate_dissolution_absorption("DrugA", 100.0, solubility_mg_mL=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_dissolution_absorption("DrugA", 100.0, vd_L=0.0)
