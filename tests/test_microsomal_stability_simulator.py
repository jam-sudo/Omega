"""Tests for the microsomal stability assay simulator (Phase 837)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.microsomal_stability_simulator import (
    MicrosomalStabilityResult,
    compare_species,
    simulate_microsomal_stability,
)

# ---------------------------------------------------------------------------
# Basic return-type tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_microsomal_stability("DrugA", t_half_microsomal_min=30.0)
    assert isinstance(result, MicrosomalStabilityResult)


def test_result_is_frozen():
    result = simulate_microsomal_stability("DrugA", t_half_microsomal_min=30.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_drug_name_preserved():
    result = simulate_microsomal_stability("Compound42", t_half_microsomal_min=20.0)
    assert result.drug_name == "Compound42"


def test_default_species_is_human():
    result = simulate_microsomal_stability("DrugA", t_half_microsomal_min=30.0)
    assert result.species == "human"


def test_default_fu_mic_is_one():
    result = simulate_microsomal_stability("DrugA", t_half_microsomal_min=30.0)
    assert result.fu_mic == 1.0


# ---------------------------------------------------------------------------
# CLint in-vitro formula verification
# ---------------------------------------------------------------------------


def test_clint_invitro_formula():
    """CLint_invitro = 0.693 / t_half * (1000_uL / protein_mg_in_incubation).

    protein_mg_in_incubation = protein_conc_mg_per_mL * 1000uL/1000 = protein_conc (mg)
    So CLint_invitro = 0.693 / t_half * (1000 / protein_conc)
    """
    t_half = 20.0
    protein = 0.5
    result = simulate_microsomal_stability(
        "X", t_half_microsomal_min=t_half, microsomal_protein_mg_per_mL=protein
    )
    expected = 0.693 / t_half * (1000.0 / protein)
    assert abs(result.clint_uL_per_min_per_mg - expected) < 1e-6


def test_clint_invitro_doubles_when_protein_halved():
    """Halving protein concentration doubles CLint_invitro."""
    r1 = simulate_microsomal_stability(
        "X", t_half_microsomal_min=30.0, microsomal_protein_mg_per_mL=1.0
    )
    r2 = simulate_microsomal_stability(
        "X", t_half_microsomal_min=30.0, microsomal_protein_mg_per_mL=0.5
    )
    assert abs(r2.clint_uL_per_min_per_mg / r1.clint_uL_per_min_per_mg - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# t_half → stability relationship
# ---------------------------------------------------------------------------


def test_higher_t_half_gives_higher_stability_label():
    """Longer half-life → 'high' metabolic stability."""
    low_t = simulate_microsomal_stability("DrugA", t_half_microsomal_min=5.0)
    high_t = simulate_microsomal_stability("DrugA", t_half_microsomal_min=60.0)
    assert low_t.metabolic_stability == "low"
    assert high_t.metabolic_stability == "high"


def test_higher_t_half_gives_lower_clint():
    """Longer half-life → lower CLint (inverse relationship)."""
    slow = simulate_microsomal_stability("DrugA", t_half_microsomal_min=90.0)
    fast = simulate_microsomal_stability("DrugA", t_half_microsomal_min=5.0)
    assert slow.clint_uL_per_min_per_mg < fast.clint_uL_per_min_per_mg


def test_metabolic_stability_boundaries():
    at_30 = simulate_microsomal_stability("X", t_half_microsomal_min=30.001)
    at_10 = simulate_microsomal_stability("X", t_half_microsomal_min=10.001)
    below_10 = simulate_microsomal_stability("X", t_half_microsomal_min=9.9)
    assert at_30.metabolic_stability == "high"
    assert at_10.metabolic_stability == "moderate"
    assert below_10.metabolic_stability == "low"


def test_valid_metabolic_stability_values():
    for t in [5.0, 15.0, 45.0]:
        r = simulate_microsomal_stability("X", t_half_microsomal_min=t)
        assert r.metabolic_stability in {"high", "moderate", "low"}


# ---------------------------------------------------------------------------
# Hepatic CL bounded by Q_H
# ---------------------------------------------------------------------------


def test_cl_hepatic_bounded_by_q_h_human():
    """CL_H cannot exceed hepatic blood flow Q_H."""
    q_h_human = 20.7
    result = simulate_microsomal_stability("HighCL", t_half_microsomal_min=0.5)
    assert result.cl_hepatic_mL_per_min_per_kg <= q_h_human + 1e-9


def test_cl_hepatic_bounded_by_q_h_rat():
    q_h_rat = 70.0
    result = simulate_microsomal_stability("HighCL", t_half_microsomal_min=0.5, species="rat")
    assert result.cl_hepatic_mL_per_min_per_kg <= q_h_rat + 1e-9


def test_cl_hepatic_positive():
    result = simulate_microsomal_stability("X", t_half_microsomal_min=30.0)
    assert result.cl_hepatic_mL_per_min_per_kg > 0.0


# ---------------------------------------------------------------------------
# Clearance class
# ---------------------------------------------------------------------------


def test_predicted_clearance_class_valid_values():
    for t in [2.0, 20.0, 120.0]:
        r = simulate_microsomal_stability("X", t_half_microsomal_min=t)
        assert r.predicted_clearance_class in {"low", "moderate", "high"}


def test_low_clint_gives_low_clearance_class():
    """Very long t_half → low CLint → low hepatic CL class."""
    result = simulate_microsomal_stability("StableCompound", t_half_microsomal_min=500.0)
    assert result.predicted_clearance_class == "low"


def test_high_clint_gives_high_clearance_class():
    """Very short t_half → high CLint → high hepatic CL class."""
    result = simulate_microsomal_stability("UnstableCompound", t_half_microsomal_min=1.0)
    assert result.predicted_clearance_class == "high"


# ---------------------------------------------------------------------------
# compare_species
# ---------------------------------------------------------------------------


def test_compare_species_returns_four_results():
    results = compare_species("DrugX", t_half_microsomal_min=20.0)
    assert len(results) == 4


def test_compare_species_covers_all_four_species():
    results = compare_species("DrugX", t_half_microsomal_min=20.0)
    species_set = {r.species for r in results}
    assert species_set == {"human", "rat", "mouse", "dog"}


def test_compare_species_sorted_descending():
    results = compare_species("DrugX", t_half_microsomal_min=20.0)
    cl_values = [r.cl_hepatic_mL_per_min_per_kg for r in results]
    assert cl_values == sorted(cl_values, reverse=True)


def test_compare_species_all_correct_type():
    results = compare_species("DrugX", t_half_microsomal_min=15.0)
    for r in results:
        assert isinstance(r, MicrosomalStabilityResult)


# ---------------------------------------------------------------------------
# Scaling factor
# ---------------------------------------------------------------------------


def test_scaling_factor_human():
    """Human scaling factor = 52.5 * 21.4 = 1123.5"""
    result = simulate_microsomal_stability("X", t_half_microsomal_min=30.0, species="human")
    assert abs(result.scaling_factor - 52.5 * 21.4) < 1e-6


def test_scaling_factor_mouse():
    """Mouse scaling factor = 45.0 * 88.0 = 3960.0"""
    result = simulate_microsomal_stability("X", t_half_microsomal_min=30.0, species="mouse")
    assert abs(result.scaling_factor - 45.0 * 88.0) < 1e-6


# ---------------------------------------------------------------------------
# fu_mic effect
# ---------------------------------------------------------------------------


def test_lower_fu_mic_reduces_clint_invivo():
    r1 = simulate_microsomal_stability("X", t_half_microsomal_min=20.0, fu_mic=1.0)
    r2 = simulate_microsomal_stability("X", t_half_microsomal_min=20.0, fu_mic=0.1)
    assert r2.clint_mL_per_min_per_kg < r1.clint_mL_per_min_per_kg


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_t_half():
    with pytest.raises(ValueError, match="t_half"):
        simulate_microsomal_stability("X", t_half_microsomal_min=0.0)


def test_raises_on_negative_t_half():
    with pytest.raises(ValueError):
        simulate_microsomal_stability("X", t_half_microsomal_min=-5.0)


def test_raises_on_zero_protein():
    with pytest.raises(ValueError, match="microsomal_protein"):
        simulate_microsomal_stability(
            "X", t_half_microsomal_min=10.0, microsomal_protein_mg_per_mL=0.0
        )


def test_raises_on_fu_mic_zero():
    with pytest.raises(ValueError, match="fu_mic"):
        simulate_microsomal_stability("X", t_half_microsomal_min=10.0, fu_mic=0.0)


def test_raises_on_fu_mic_greater_than_one():
    with pytest.raises(ValueError, match="fu_mic"):
        simulate_microsomal_stability("X", t_half_microsomal_min=10.0, fu_mic=1.5)


def test_raises_on_invalid_species():
    with pytest.raises(ValueError, match="species"):
        simulate_microsomal_stability("X", t_half_microsomal_min=10.0, species="fish")
